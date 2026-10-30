"""X1:跨腔文本重疊的敏感度分析。不需要 GPU,幾分鐘。

主實驗(Table II、Fig. 2)的 100 句測試句沒有排除「文字也出現在另一腔訓練池」的句子,
而 XAD 的來源模型在另一腔訓練時可能看過這些文字。這裡不重新合成,沿用 D1 的逐音節結果,
標出重疊句,比較「全部句子」與「排除重疊句」兩種情況下的入聲 / 舒聲 ±30 ms 命中率,
以及各系統與 base 的差(命中率、MCD)。兩種情況結論相同,論文就可以寫「排除後結果不變」。

重疊判定與 TAAI 相同:文字 NFKC 正規化、去掉標點與空白後完全相同。
文字欄 = TSV 每列中含漢字最多的欄位;檔名 = 副檔名為 .wav 的欄位(沒有就取第一欄)。

    python x1_overlap.py --dialect hailu --sys base=DIR1,DIR2,DIR3 --sys c6=... --sys 48h=DIR
"""
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]")
TOL_MS = 30


def norm(text):
    """NFKC 後只留文字與數字(去掉標點、空白)。"""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if ch.isalnum())


def tsv_texts(path):
    """{檔名: 正規化文字}。沒有漢字的列(例如表頭)略過。"""
    out = {}
    for row in csv.reader(open(path, encoding="utf-8"), delimiter="\t"):
        if not row:
            continue
        text = max(row, key=lambda c: len(CJK.findall(c)))
        if not CJK.search(text):
            continue
        stem = next((Path(c).stem for c in row if c.lower().endswith((".wav", ".flac"))), Path(row[0]).stem)
        out[stem] = norm(text)
    return out


SYL_CSV = "syllables_wave.csv"      # --csv 可改成 x3_remeasure.py 的 syllables_wave_x3.csv


def hit_counts(dirs, stems, entering):
    """每句的「誤差在 ±30 ms 內的母音核數」與「母音核數」(各訓練種子加總)。"""
    pos = {s: i for i, s in enumerate(stems)}
    hits, n = np.zeros(len(stems)), np.zeros(len(stems))
    for d in dirs:
        c = read_cols(Path(d) / SYL_CSV)
        st = np.asarray(c["stem"]).astype(str)
        m = (boolcol(c["entering"]) == entering) & np.isin(st, stems)
        idx = np.array([pos[s] for s in st[m]], dtype=int)
        ok = (np.abs(c["pred_ms"][m].astype(float) - c["ref_ms"][m].astype(float)) <= TOL_MS).astype(float)
        hits += np.bincount(idx, weights=ok, minlength=len(stems))
        n += np.bincount(idx, minlength=len(stems))
    return hits, n


def checked_rho(dirs, stems):
    """入聲母音核「合成時長 vs 真人時長」的 Pearson ρ,各訓練種子分別算再平均(同 c_seeds.py)。"""
    vals = []
    for d in dirs:
        c = read_cols(Path(d) / SYL_CSV)
        m = boolcol(c["entering"]) & np.isin(np.asarray(c["stem"]).astype(str), stems)
        vals.append(pearson(c["pred_ms"][m].astype(float), c["ref_ms"][m].astype(float)))
    return float(np.mean(vals))


def mcd_per_stem(dirs, stems):
    """每句 MCD(生成種子平均,再對訓練種子平均);沒有 mcd.csv 時回傳 None。"""
    rows = []
    for d in dirs:
        f = Path(d) / "mcd.csv"
        if not f.exists():
            return None
        c = read_cols(f)
        acc = {}
        for s, v in zip(np.asarray(c["stem"]).astype(str), c["mcd"].astype(float)):
            acc.setdefault(s, []).append(v)
        rows.append([np.mean(acc[s]) if s in acc else np.nan for s in stems])
    return np.nanmean(np.array(rows), axis=0)


def rate(hn, sel=None):
    h, n = hn
    if sel is None:
        return h.sum() / max(n.sum(), 1e-9)
    return h[sel].sum(axis=-1) / np.maximum(n[sel].sum(axis=-1), 1e-9)


def ci(point, draws, scale=100.0, digits=1):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return f"{scale * point:+.{digits}f} [{scale * lo:+.{digits}f}, {scale * hi:+.{digits}f}]"


def report(systems, base, stems, n_boot, seed=42):
    """systems: {名稱: [資料夾]};stems:這個句子集合。回傳 Markdown 表格列。"""
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, len(stems), size=(n_boot, len(stems)))
    hit = {k: {et: hit_counts(v, stems, et) for et in (True, False)} for k, v in systems.items()}
    mcd = {k: mcd_per_stem(v, stems) for k, v in systems.items()}
    L = []
    for k in systems:
        cells = [f"{100 * rate(hit[k][True]):.1f}", f"{100 * rate(hit[k][False]):.1f}", f"{checked_rho(systems[k], stems):.3f}"]
        if k == base:
            cells += ["—", "—"]
        else:
            d = rate(hit[k][True]) - rate(hit[base][True])
            cells.append(ci(d, rate(hit[k][True], sel) - rate(hit[base][True], sel)))
            if mcd[k] is not None and mcd[base] is not None:
                diff = mcd[k] - mcd[base]
                ok = np.isfinite(diff)
                dd = np.where(ok, diff, 0.0)
                draws = dd[sel].sum(axis=1) / np.maximum(ok[sel].sum(axis=1), 1)
                cells.append(ci(dd.sum() / max(ok.sum(), 1), draws, scale=1.0, digits=3))
            else:
                cells.append("(沒有 mcd.csv)")
        L.append(f"| {k}({len(systems[k])} 種子) | " + " | ".join(cells) + " |")
    return L


def main():
    global SYL_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--sys", action="append", required=True, metavar="NAME=DIR1,DIR2,...")
    ap.add_argument("--base", default="base")
    ap.add_argument("--test-tsv", help=f"預設 {C.BENCH}/<腔調>/test.tsv")
    ap.add_argument("--src-tsv", help=f"預設 {C.BENCH}/<另一腔>/train_full.tsv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--csv", default=SYL_CSV, help="逐音節檔名(X3 用 syllables_wave_x3.csv)")
    ap.add_argument("--out")
    args = ap.parse_args()
    SYL_CSV = args.csv

    other = [d for d in C.DIALECTS if d != args.dialect][0]
    test = tsv_texts(Path(args.test_tsv or C.BENCH / args.dialect / "test.tsv").expanduser())
    src = set(tsv_texts(Path(args.src_tsv or C.BENCH / other / "train_full.tsv").expanduser()).values())
    if not test or not src:
        sys.exit("★ TSV 裡找不到含漢字的文字欄;請用 head -3 看一下 benchmark/<腔調>/test.tsv 的欄位")
    systems = {}
    for spec in args.sys:
        name, dirs = spec.split("=", 1)
        systems[name] = [Path(x).expanduser() for x in dirs.split(",") if x]
    if args.base not in systems:
        sys.exit(f"★ 沒有基準系統 {args.base}")

    stems = sorted(set(np.asarray(read_cols(systems[args.base][0] / SYL_CSV)["stem"]).astype(str)))
    missing = [s for s in stems if s not in test]
    if len(missing) == len(stems):
        sys.exit(f"★ 評估句的檔名對不到 test.tsv(例如 {stems[:3]} vs {list(test)[:3]})")
    overlap = [s for s in stems if test.get(s) in src]
    clean = [s for s in stems if s not in overlap and s not in missing]
    uniq = set(test.values())

    L = [f"# X1 跨腔文本重疊敏感度:{args.dialect}\n",
         f"整個測試集:{len(uniq & src)} / {len(uniq)} 個不重複文本也出現在{other}訓練池"
         f"(手冊數字:四縣 15/411、海陸 26/409,用來確認文字欄抓對了)。\n",
         f"評估的 {len(stems)} 句中有 {len(overlap)} 句重疊"
         + (f",另有 {len(missing)} 句在 test.tsv 找不到文字" if missing else "") + "。\n",
         f"重疊句:{', '.join(overlap) if overlap else '(無)'}\n",
         f"命中率 = 母音核時長誤差在 ±{TOL_MS} ms 內的比例(%);差 = 系統 − {args.base},CI 以句子重抽 {args.n_boot} 次。\n"]
    head = [f"| 系統 | 入聲 ±30 ms | 舒聲 ±30 ms | 入聲 ρ | 入聲命中率與 {args.base} 的差 | MCD 與 {args.base} 的差 (dB) |",
            "|---|---|---|---|---|---|"]
    L += [f"## 全部 {len(stems)} 句\n"] + head + report(systems, args.base, stems, args.n_boot)
    if overlap:
        L += ["", f"## 排除重疊句後 {len(clean)} 句\n"] + head + report(systems, args.base, clean, args.n_boot)
    else:
        L += ["", "評估句中沒有重疊句,不需要另外分析。"]
    text = "\n".join(L)
    print(text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "x1" / f"overlap_{args.dialect}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
