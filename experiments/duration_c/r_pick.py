"""R3:在驗證集上選 Raon-OpenTTS 的訓練步數(不看測試集)。不需要 GPU。

指標:驗證句「所有母音核」(入聲 + 舒聲)從波形量到的長度與真人相差 ≤ 30 ms 的比例;同分取較早的步數。
只用全部母音(不只入聲),避免把選模型的標準調到入聲上。

    python r_pick.py --dirs 2500=DIR 5000=DIR ...      # 最後一行印出選中的步數
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402


def hit_rate(d, tol=30.0):
    c = read_cols(Path(d) / "syllables_wave.csv")
    err = np.abs(c["pred_ms"].astype(float) - c["ref_ms"].astype(float))
    et = boolcol(c["entering"])
    return float((err <= tol).mean()), float((err[et] <= tol).mean()), len(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, metavar="STEP=DIR")
    args = ap.parse_args()
    res = {}
    print("| 步數 | 全部母音命中率 | 入聲命中率(僅供參考) | 音節數 |\n|---|---|---|---|")
    for x in args.dirs:
        st, d = x.split("=", 1)
        if not (Path(d) / "syllables_wave.csv").exists():
            print(f"| {st} | (沒有結果) | | |")
            continue
        a, e, n = hit_rate(d)
        res[int(st)] = a
        print(f"| {st} | {100 * a:.1f}% | {100 * e:.1f}% | {n} |")
    if not res:
        sys.exit("★ 沒有任何驗證結果")
    best = max(res.values())
    print(min(s for s, v in res.items() if v == best))


if __name__ == "__main__":
    main()
