"""在「驗證集」上替每個系統選取整方式(ceil 或 cum),測試集不參與。不需要 GPU。

C4 與 C5 都看到:預測器偏差改變後(資料變多、遷移初始化),ceil 固定加上的 +6–8 ms 不再被抵銷,
整體變長。所以取整方式要跟著系統選,而且要在驗證集上事先選好,避免看了測試集才挑。

規則(事先固定):母音核(不含 blank)「取整後 − 訓練目標」的平均絕對誤差,入聲與舒聲各算一次再平均
(兩類等權,入聲不會被舒聲的數量蓋過);較小者勝,同分取 ceil(Matcha 原本的做法)。

    python c3_target_check.py --exp <實驗> --dialect sixian --filelist <驗證集>_mfadur.txt --name <名稱>
    python c6_pick_rounding.py --val ~/hakka_tts/exp_c/c3/<名稱>/recs.pkl
輸出最後一行是 ceil 或 cum(給腳本讀);表格另存 rounding.md。
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c3_fit_pp import nucleus_errors  # noqa: E402

ROUNDINGS = ("ceil", "cum")


def score(recs, rounding):
    e, et, _ = nucleus_errors(recs, 1.0, rounding)
    return {"et_bias": e[et].mean(), "non_bias": e[~et].mean(),
            "et_mae": np.abs(e[et]).mean(), "non_mae": np.abs(e[~et]).mean(),
            "obj": (np.abs(e[et]).mean() + np.abs(e[~et]).mean()) / 2}


def pick(recs):
    res = {r: score(recs, r) for r in ROUNDINGS}
    best = min(ROUNDINGS, key=lambda r: (round(res[r]["obj"], 6), r != "ceil"))
    return best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="c3_target_check.py 在驗證集輸出的 recs.pkl")
    args = ap.parse_args()
    p = Path(args.val).expanduser()
    recs = pickle.load(open(p, "rb"))["recs"]
    best, res = pick(recs)
    L = [f"# 取整方式(驗證集 {len(recs)} 句):選 {best}\n",
         "母音核(不含 blank)取整後 − 訓練目標,ms。\n",
         "| 取整 | 入聲偏差 | 舒聲偏差 | 入聲平均絕對誤差 | 舒聲平均絕對誤差 | 準則(兩者平均) |", "|---|---|---|---|---|---|"]
    for r in ROUNDINGS:
        s = res[r]
        L.append(f"| {r}{' ★' if r == best else ''} | {s['et_bias']:+.1f} | {s['non_bias']:+.1f} | "
                 f"{s['et_mae']:.1f} | {s['non_mae']:.1f} | {s['obj']:.2f} |")
    text = "\n".join(L)
    (p.parent / "rounding.md").write_text(text, encoding="utf-8")
    print(text)
    print(best)


if __name__ == "__main__":
    main()
