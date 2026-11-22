"""S3:檢查模型用的「音節切分規則」(matcha_patch/.../hakka_syllables.py)和 TextGrid 的拼音音節是否一致。不需要 GPU。

模型推論時只看得到符號序列,音節範圍要用規則推出;唯一有歧義的是「舒聲音節後面單一個鼻音 m/n/ng」
(可能是前一音節的韻尾,也可能是後一音節的聲母)。這裡用 TextGrid 的 words 層(每個 word 就是一個拼音音節,
只用它的分組、不用時間)當標準答案,比較兩種鼻音規則的正確率,決定訓練時用哪一種。

    python s3_syllable_check.py --dialect sixian
"""
import argparse
import ast
import importlib.util
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.structure import base_of, is_pause  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402

PATCH = Path(__file__).resolve().parent.parent / "matcha_patch" / "matcha" / "models" / "components" / "hakka_syllables.py"
PUA_START = 0xE000


def load_rule():
    spec = importlib.util.spec_from_file_location("hakka_syllables", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_symbols(path):
    src = Path(path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "HAKKA_SYMBOLS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path} 沒有 HAKKA_SYMBOLS")


def tg_groups(names, tiers):
    """TextGrid words 層的音節分組:每個音節 = 一組符號位置(names 的索引,不含停頓)。對不上回傳 None。"""
    ph = [(a, b, t.strip()) for a, b, t in find_tier(tiers, "phones") if t.strip().lower() not in C.MFA_SILENCE]
    if any(t.lower() in C.MFA_INVALID for _, _, t in ph):
        return None
    sym = [i for i, n in enumerate(names) if not is_pause(n)]
    if len(sym) != len(ph) or any(base_of(names[i]) != lab for i, (_, _, lab) in zip(sym, ph)):
        return None
    out, k, eps = [], 0, 1e-4
    for wa, wb, wt in find_tier(tiers, "words"):
        if wt.strip().lower() in C.MFA_SILENCE:
            continue
        mem = []
        while k < len(ph) and ph[k][0] >= wa - eps and ph[k][1] <= wb + eps:
            mem.append(sym[k])
            k += 1
        if mem:
            out.append(frozenset(mem))
    return out if k == len(ph) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--lists", nargs="*", help="預設 Matcha 的 data/hakka/joint/<腔>/*.txt(全部去重)")
    ap.add_argument("--symbols", help="預設 ~/Matcha-TTS/matcha/text/symbols.py")
    ap.add_argument("--out")
    ap.add_argument("--min-acc", type=float, default=0.97, help="最佳規則的音節正確率低於此值時結束代碼為 2")
    args = ap.parse_args()
    rule = load_rule()
    syms = load_symbols(args.symbols or C.MATCHA / "matcha" / "text" / "symbols.py")
    table = [None] + list(syms)
    lists = [Path(p) for p in args.lists] if args.lists else sorted((C.MATCHA / "data" / "hakka" / "joint" / args.dialect).glob("*.txt"))
    tgs = {p.stem: p for p in C.textgrid_dir(args.dialect).rglob("*.TextGrid")}
    seen, why = {}, Counter()
    for fl in lists:
        for line in open(fl, encoding="utf-8"):
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 3:
                seen.setdefault(Path(parts[0]).stem, parts[2])
    res = {r: Counter() for r in ("onset", "coda")}
    amb = Counter()
    for stem, text in sorted(seen.items()):
        if stem not in tgs:
            why["沒有 TextGrid"] += 1
            continue
        try:
            ids = [ord(c) - PUA_START + 1 for c in text]
            names = [table[i] for i in ids]
        except (IndexError, TypeError):
            why["文字含非客語符號"] += 1
            continue
        ref = tg_groups(names, read_textgrid(tgs[stem]))
        if ref is None:
            why["TextGrid 與符號對不上"] += 1
            continue
        for r in res:
            syl, _ = rule.segment(ids, table, r)
            grp = {}
            for p, s in enumerate(syl):
                if s >= 0:
                    grp.setdefault(s, set()).add(p)
            got = {frozenset(v) for v in grp.values()}
            hit = sum(g in got for g in ref)
            res[r]["音節"] += len(ref)
            res[r]["正確"] += hit
            res[r]["句"] += 1
            res[r]["整句正確"] += int(hit == len(ref) and len(got) == len(ref))
        # 統計有歧義的位置(舒聲音節後面單一鼻音),依 TextGrid 看實際是韻尾還是聲母
        pos_syl = {p: k for k, g in enumerate(ref) for p in g}
        anchors = [p for p, n in enumerate(names) if rule.classify(n)[0] == "anchor"]
        for a, b in zip(anchors, anchors[1:]):
            mid = [p for p in range(a + 1, b)]
            if len(mid) == 1 and not any(is_pause(names[p]) for p in mid):
                c = rule.classify(names[mid[0]])
                if c[2] and not rule.classify(names[a])[1]:
                    amb["韻尾" if pos_syl.get(mid[0]) == pos_syl.get(a) else "聲母"] += 1
    L = [f"# S3 音節切分規則 vs TextGrid:{args.dialect}\n",
         f"清單:{', '.join(p.name for p in lists)};不重複的句子 {len(seen)};略過:{dict(why) or '無'}\n",
         "| 鼻音規則 | 句數 | 音節正確率 | 整句完全正確 |", "|---|---|---|---|"]
    for r, c in res.items():
        L.append(f"| {r}({'當後一音節的聲母' if r == 'onset' else '當前一音節的韻尾'}) | {c['句']} | "
                 f"{100 * c['正確'] / max(c['音節'], 1):.2f}%({c['正確']}/{c['音節']}) | {100 * c['整句正確'] / max(c['句'], 1):.1f}% |")
    tot = sum(amb.values())
    L.append(f"\n有歧義的位置(舒聲音節後面單一鼻音){tot} 個:實際為韻尾 {amb['韻尾']}、聲母 {amb['聲母']}")
    best = max(res, key=lambda r: res[r]["正確"])
    L.append(f"\n建議:nasal_rule = {best}")
    text = "\n".join(L)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    print(text)
    print(best)
    acc = res[best]["正確"] / max(res[best]["音節"], 1)
    if acc < args.min_acc:
        sys.exit(2)


if __name__ == "__main__":
    main()
