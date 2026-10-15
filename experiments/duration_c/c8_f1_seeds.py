"""C8 / F1 精簡版:多個系統 × 多個訓練種子,入聲與舒聲依韻尾、調值、停頓前分組的平均偏差(ms)。不需要 GPU。

每格 = 三個種子各自的分組平均偏差(預測 − 參考)再平均;括號裡是各種子的值,用來看方向是否一致。

    python c8_f1_seeds.py --sys base=$E/base_sixian_step40000,$E/base_sixian_s123_step40000,... \
                          --sys c6=$E/c6_sixian_step40000_cum,... --out $E/c8_f1_sixian.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402

FACTORS = (("coda", "韻尾"), ("tone", "調值"), ("prepausal", "停頓前"))


def show(v):
    if v == "":
        return "(無)"
    return int(v) if isinstance(v, float) and float(v).is_integer() else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", action="append", required=True, metavar="NAME=DIR1,DIR2,...")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--out")
    args = ap.parse_args()
    systems = []
    for spec in args.sys:
        name, dirs = spec.split("=", 1)
        systems.append((name, [read_cols(Path(x).expanduser() / "syllables_wave.csv") for x in dirs.split(",") if x]))
    first = systems[0][1][0]
    L = ["# 分組偏差(預測 − 參考,ms;各種子平均,括號為各種子)\n",
         "| 類別 | 分組 | 值 | n | " + " | ".join(n for n, _ in systems) + " |", "|---|---|---|---|" + "---|" * len(systems)]
    for cls_name, cls_val in (("入聲", True), ("舒聲", False)):
        rows = [(("全部", None),)] + [((lab, key, v),) for key, lab in FACTORS
                                     for v in sorted(set(first[key][boolcol(first["entering"]) == cls_val]))]
        for (spec,) in rows:
            cells, n = [], None
            for _, seeds in systems:
                vals = []
                for c in seeds:
                    m = boolcol(c["entering"]) == cls_val
                    if spec[1] is not None:
                        m &= c[spec[1]] == spec[2]
                    if m.sum() >= args.min_n:
                        vals.append(float((c["pred_ms"][m] - c["ref_ms"][m]).mean()))
                        n = int(m.sum())
                cells.append(f"{np.mean(vals):+.1f}({' / '.join(f'{v:+.0f}' for v in vals)})" if vals else "—")
            if n is None:
                continue
            lab, val = (spec[0], "") if spec[1] is None else (spec[0], show(spec[2]))
            L.append(f"| {cls_name} | {lab} | {val} | {n} | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
