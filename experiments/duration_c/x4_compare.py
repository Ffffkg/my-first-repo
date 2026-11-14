"""X4:比較「只用 1 h 訓練的 MFA」和主實驗 MFA 對同一批訓練句的對齊結果(時長標籤品質)。不需要 GPU。

兩邊用同一份發音詞典,所以每句的音素序列相同;逐一比較音素的起訖點與長度,
報告母音與全部音素的平均絕對差、相關係數,以及邊界差在 10 / 20 ms 內的比例。

    python x4_compare.py --dialect sixian --new-tg ~/hakka_tts/exp_c/x4/sixian/tg_train
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402
from hakka_c.structure import has_vowel  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402


def phones(path):
    return [(a, b, t.strip()) for a, b, t in find_tier(read_textgrid(path), "phones")
            if t.strip().lower() not in C.MFA_SILENCE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--new-tg", required=True, help="X4 MFA 的 TextGrid 資料夾")
    ap.add_argument("--ref-tg", help="預設主實驗 MFA 的 textgrids/<腔>")
    args = ap.parse_args()

    ref = {p.stem: p for p in (Path(args.ref_tg).expanduser() if args.ref_tg else C.textgrid_dir(args.dialect)).rglob("*.TextGrid")}
    new = {p.stem: p for p in Path(args.new_tg).expanduser().rglob("*.TextGrid")}
    d_all, d_vow, bnd, skipped = [], [], [], 0
    ref_v, new_v = [], []
    for stem, p in sorted(new.items()):
        if stem not in ref:
            skipped += 1
            continue
        a, b = phones(ref[stem]), phones(p)
        if [x[2] for x in a] != [x[2] for x in b]:
            skipped += 1
            continue
        for (s0, e0, lab), (s1, e1, _) in zip(a, b):
            d = ((e1 - s1) - (e0 - s0)) * 1000
            d_all.append(d)
            bnd += [abs(s1 - s0) * 1000, abs(e1 - e0) * 1000]
            if has_vowel(lab):
                d_vow.append(d)
                ref_v.append((e0 - s0) * 1000)
                new_v.append((e1 - s1) * 1000)
    if not d_all:
        sys.exit(f"★ 沒有可比較的句子(略過 {skipped} 句)")
    d_all, d_vow, bnd = map(np.asarray, (d_all, d_vow, bnd))
    print(f"# X4 時長標籤品質:{args.dialect}(1 h MFA vs 主實驗 MFA)\n")
    print(f"比較 {len(new) - skipped} 句({skipped} 句音素序列不同或缺少,略過)\n")
    print("| | 平均差 (ms) | 平均絕對差 (ms) | r |")
    print("|---|---|---|---|")
    print(f"| 全部音素長度 | {d_all.mean():+.1f} | {np.abs(d_all).mean():.1f} | — |")
    print(f"| 母音長度 | {d_vow.mean():+.1f} | {np.abs(d_vow).mean():.1f} | {pearson(ref_v, new_v):.3f} |")
    print(f"\n音素邊界差在 10 ms 內 {100 * (bnd <= 10).mean():.1f}%、20 ms 內 {100 * (bnd <= 20).mean():.1f}%(1 個音框約 11.6 ms)")


if __name__ == "__main__":
    main()
