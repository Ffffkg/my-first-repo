"""S2:真人錄音上,兩套 MFA 對「各種音素邊界」切得多一致?不需要 GPU。

S1 發現入聲母音的錯誤主要來自「母音占音節的比例」。這個比例要當成訓練標籤,所以先檢查它的穩定性:
比較主實驗 MFA(整個腔調的錄音訓練)與 X4 的 1 小時 MFA,在同一批真人錄音上切出的邊界,依邊界種類分開統計。
特別關心「母音 → 塞音韻尾(-b/-d/-g)」:不除阻的塞音幾乎無聲,這個邊界可能最難切。

注意:兩套 MFA 一致只代表「可重現」,不代表邊界正確(兩者可能有共同偏差);而且這裡量的是真人錄音,不是合成語音。

    python s2_boundary_check.py --dialect sixian            # 預設比較 textgrids/<腔> 與 exp_c/x4/<腔>/tg
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402
from hakka_c.structure import has_vowel, is_entering_syllable  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402

NASAL = {"m", "n", "ng"}


def syllables(path):
    """[(拼音, [(音素, 起, 終), ...]), ...];含 spn 等無效音素時回傳 None。"""
    tiers = read_textgrid(path)
    ph = find_tier(tiers, "phones")
    if any(t.strip().lower() in C.MFA_INVALID for _, _, t in ph):
        return None
    ph = [(t.strip(), a, b) for a, b, t in ph if t.strip().lower() not in C.MFA_SILENCE]
    out, k, eps = [], 0, 1e-4
    for wa, wb, wt in find_tier(tiers, "words"):
        w = wt.strip()
        if w.lower() in C.MFA_SILENCE:
            continue
        mem = []
        while k < len(ph) and ph[k][1] >= wa - eps and ph[k][2] <= wb + eps:
            mem.append(ph[k])
            k += 1
        out.append((w, mem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--a-tg", help="預設主實驗 MFA:textgrids/<腔>")
    ap.add_argument("--b-tg", help="預設 X4 的 1 小時 MFA:exp_c/x4/<腔>/tg")
    ap.add_argument("--out")
    args = ap.parse_args()
    A = {p.stem: p for p in (Path(args.a_tg).expanduser() if args.a_tg else C.textgrid_dir(args.dialect)).rglob("*.TextGrid")}
    B = {p.stem: p for p in (Path(args.b_tg).expanduser() if args.b_tg else C.OUT_ROOT / "x4" / args.dialect / "tg").rglob("*.TextGrid")}

    bnd = defaultdict(list)                    # 邊界種類 → |差| (ms)
    dur = defaultdict(lambda: defaultdict(list))   # 入聲/舒聲 → 量 → [(A, B)]
    n_utt = skipped = 0
    for stem in sorted(set(A) & set(B)):
        sa, sb = syllables(A[stem]), syllables(B[stem])
        if sa is None or sb is None or [w for w, _ in sa] != [w for w, _ in sb] or \
                any([p[0] for p in ma] != [p[0] for p in mb] for (_, ma), (_, mb) in zip(sa, sb)):
            skipped += 1
            continue
        n_utt += 1
        for (w, ma), (_, mb) in zip(sa, sb):
            ent = is_entering_syllable(w)
            vow = [i for i, p in enumerate(ma) if has_vowel(p[0])]
            if ent is None or len(vow) != 1:
                continue                       # 成音節鼻音等沒有單一母音核的音節略過
            v = vow[0]
            cls = "入聲" if ent else "舒聲"
            d = lambda x, y: abs(x - y) * 1000  # noqa: E731
            bnd[f"{cls}:音節起點"].append(d(ma[0][1], mb[0][1]))
            bnd[f"{cls}:音節終點"].append(d(ma[-1][2], mb[-1][2]))
            if v > 0:
                bnd[f"{cls}:聲母 → 母音"].append(d(ma[v][1], mb[v][1]))
            if v < len(ma) - 1:
                coda = ma[-1][0]
                kind = "塞音韻尾" if ent else ("鼻音韻尾" if coda in NASAL else "其他韻尾")
                bnd[f"{cls}:母音 → {kind}"].append(d(ma[v][2], mb[v][2]))
            else:
                bnd[f"{cls}:母音 → 音節結束(無韻尾)"].append(d(ma[v][2], mb[v][2]))
            for tag, mm in (("A", ma), ("B", mb)):
                S = (mm[-1][2] - mm[0][1]) * 1000
                V = (mm[v][2] - mm[v][1]) * 1000
                dur[cls][f"S_{tag}"].append(S)
                dur[cls][f"V_{tag}"].append(V)
                dur[cls][f"sh_{tag}"].append(V / S if S > 0 else np.nan)

    L = [f"# S2 真人錄音上兩套 MFA 的邊界一致性:{args.dialect}\n",
         f"A = 主實驗 MFA、B = 1 小時 MFA;比對 {n_utt} 句(略過 {skipped} 句:音素序列不同或含無效音素)\n",
         "## 1. 依邊界種類\n",
         "| 邊界 | 個數 | 平均絕對差 (ms) | 中位數 (ms) | ≤ 10 ms | ≤ 20 ms | > 30 ms |", "|---|---|---|---|---|---|---|"]
    order = ["音節起點", "聲母 → 母音", "母音 → 塞音韻尾", "母音 → 鼻音韻尾", "母音 → 其他韻尾", "母音 → 音節結束(無韻尾)", "音節終點"]
    for cls in ("入聲", "舒聲"):
        for o in order:
            x = np.asarray(bnd.get(f"{cls}:{o}", []))
            if len(x) < 20:
                continue
            L.append(f"| {cls}:{o} | {len(x)} | {x.mean():.1f} | {np.median(x):.1f} | {100 * (x <= 10).mean():.1f}% | "
                     f"{100 * (x <= 20).mean():.1f}% | {100 * (x > 30).mean():.1f}% |")
    L += ["", "## 2. 當成訓練標籤的三個量:兩套 MFA 的一致程度\n",
          "| | 個數 | 平均絕對差 | 相關 r |", "|---|---|---|---|"]
    for cls in ("入聲", "舒聲"):
        dd = {k: np.asarray(v, float) for k, v in dur[cls].items()}
        if not dd:
            continue
        for q, lab, unit in (("S", "音節總長", "ms"), ("V", "母音長度", "ms"), ("sh", "母音占音節比例", "")):
            a, b = dd[f"{q}_A"], dd[f"{q}_B"]
            m = np.isfinite(a) & np.isfinite(b)
            mae = np.abs(a[m] - b[m]).mean()
            L.append(f"| {cls}:{lab} | {m.sum()} | {mae:.3f}{' ' + unit if unit else ''} | {pearson(a[m], b[m]):.3f} |")
    L += ["", "## 判讀方式\n",
          "- 「母音 → 塞音韻尾」若明顯比「聲母 → 母音」「母音 → 鼻音韻尾」不一致,代表入聲母音比例的標籤較不穩定:"
          "第二層的比例損失要降低權重或做平滑,S1 的 (b) 也有一部分可能來自量測誤差。",
          "- 若各種邊界一致程度相近,母音比例可以放心當作訓練標籤。",
          "- 和 S1 的「母音比例平均絕對誤差(合成 vs 真人)」比較:若兩套 MFA 在真人錄音上的比例差,已接近合成與真人的比例差,"
          "代表 S1 看到的比例錯誤有相當部分可能落在量測誤差範圍內(只能作為參考,兩者量的是不同的音檔)。"]
    text = "\n".join(L)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
