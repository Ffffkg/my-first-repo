"""C3 診斷:停頓前母音核偏長,在不同句子集合(訓練 / 驗證 / 測試)之間為什麼不一樣。不需要 GPU。

把停頓前再分成「句末」(整句最後一個音節)與「句中停頓前」(後面接逗號等停頓符號),
並列出各集合的組成比例與「只看母音核」偏差 N(c3_target_check.py 的 components.csv)。

    C3=~/hakka_tts/exp_c/c3
    python c3_pp_split.py $C3/c2_sixian_ladder_1h/components.csv $C3/c2_sixian_val/components.csv $C3/c2_sixian_test/components.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import read_cols  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, fmt, mean_ci  # noqa: E402


def positions(c):
    """每個音節:0 = 非停頓前、1 = 句中停頓前、2 = 句末。"""
    stem, idx = c["stem"], c["syl_idx"].astype(int)
    last = {}
    for s, i in zip(stem, idx):
        last[s] = max(last.get(s, -1), i)
    final = np.array([i == last[s] for s, i in zip(stem, idx)])
    pp = c["prepausal"] > 0.5
    return np.where(final, 2, np.where(pp, 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="c3_target_check.py 的 components.csv(可多個集合)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    L = ["# 停頓前母音核偏差:句末 vs 句中停頓前\n", "N = 母音核(不含 blank)ceil 後 − 訓練目標,ms;括號內為音節數。\n",
         "| 集合 | 句數 | 舒聲・非停頓前 | 舒聲・句中停頓前 | 舒聲・句末 | 入聲・句中停頓前 | 入聲・句末 | 停頓前裡句末的比例 |",
         "|---|---|---|---|---|---|---|---|"]
    for p in args.csv:
        p = Path(p).expanduser()
        c = read_cols(p)
        pos = positions(c)
        et = c["entering"] > 0.5
        masks = {"n0": ~et & (pos == 0), "n1": ~et & (pos == 1), "n2": ~et & (pos == 2),
                 "e1": et & (pos == 1), "e2": et & (pos == 2)}
        masks = {k: m for k, m in masks.items() if m.sum() > 0}
        t = UttTable(c["stem"], {"N": c["N"]}, masks)
        idx = boot_idx(t.U, args.n_boot)
        cells = [f"{fmt(mean_ci(t, 'N', k, idx))} ({int(masks[k].sum())})" if k in masks else "—"
                 for k in ("n0", "n1", "n2", "e1", "e2")]
        share = (pos == 2).sum() / max(1, (pos > 0).sum())
        L.append(f"| {p.parent.name} | {t.U} | " + " | ".join(cells) + f" | {share:.0%} |")
    L += ["", "讀法:若句末與句中停頓前的偏差差很多,而且各集合的組成比例不同,"
              "就解釋了為什麼在驗證集估的單一縮放係數搬到測試集會失準。"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
