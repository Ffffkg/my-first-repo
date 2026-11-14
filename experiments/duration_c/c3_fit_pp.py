"""C3a:停頓前母音核的推論時縮放 s_pp,在「驗證集」上估,測試集只拿來看。不需要 GPU。

依據(c3_target_check.py):C2 的母音核本身(不含 blank)在非停頓前幾乎沒有偏差,
但停頓前的母音核在訓練句與測試句都被預測得太長,而且波形量測也看得到。
這裡只縮放停頓前音節的母音核 w,讓驗證集上「ceil 後的母音核 − 訓練目標」平均為 0。

    python c3_target_check.py ... --filelist <val>_mfadur.txt --name c2_sixian_val     # 先產生 recs.pkl
    python c3_fit_pp.py --val ~/hakka_tts/exp_c/c3/c2_sixian_val/recs.pkl --test ~/hakka_tts/exp_c/c3/c2_sixian_test/recs.pkl

輸出 s_pp 與 calib_pp.json;合成評估:bash d1_pipeline.sh <腔調> <名稱> <實驗> --which step40000 --s-pp <s_pp>
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import pred_frames  # noqa: E402

GRID = np.round(np.arange(0.50, 1.2001, 0.01), 2)


def nucleus_errors(recs, s_pp=1.0, rounding="ceil"):
    """每個母音核的 (ceil 後 frame − 目標 frame)·HOP_MS,以及 (入聲?, 停頓前?)。"""
    err, et, pp = [], [], []
    for rec in recs:
        fr = pred_frames(rec, rounding, s_pp=s_pp)
        for s in rec["syllables"]:
            p = 2 * s["nuc"] + 1
            err.append((fr[p] - rec["d"][p]) * C.HOP_MS)
            et.append(bool(s["entering"]))
            pp.append(bool(s["prepausal"]))
    return np.array(err), np.array(et), np.array(pp)


def fit_s_pp(recs, rounding="ceil", grid=GRID):
    def bias(s):
        e, _, pp = nucleus_errors(recs, s, rounding)
        return round(abs(e[pp].mean()), 6)
    return float(min(grid, key=lambda s: (bias(s), abs(s - 1.0))))    # 同樣好時取最接近 1(改動最少)


def table(recs, s_pp, rounding):
    L = []
    for label, s in (("原本", 1.0), (f"s_pp = {s_pp:.2f}", s_pp)):
        e, et, pp = nucleus_errors(recs, s, rounding)
        cells = [f"{e[m].mean():+.1f}" if m.sum() else "—"
                 for m in (et & ~pp, et & pp, ~et & ~pp, ~et & pp)]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="c3_target_check.py 在驗證集輸出的 recs.pkl")
    ap.add_argument("--test", help="(選填)測試集的 recs.pkl,只用來看,不參與估計")
    ap.add_argument("--rounding", default="ceil", choices=("ceil", "cum"))
    ap.add_argument("--out", help="calib json(預設放在 --val 旁邊)")
    args = ap.parse_args()

    val = pickle.load(open(Path(args.val).expanduser(), "rb"))["recs"]
    s_pp = fit_s_pp(val, args.rounding)
    head = ["| 設定 | 入聲・非停頓前 | 入聲・停頓前 | 舒聲・非停頓前 | 舒聲・停頓前 |", "|---|---|---|---|---|"]
    L = [f"# C3a 停頓前母音核縮放:s_pp = {s_pp:.2f}(驗證集 {len(val)} 句估計)\n",
         "數值 = 母音核(不含 blank)ceil 後 − 訓練目標,ms。\n", "## 驗證集(估計用)\n", *head,
         *table(val, s_pp, args.rounding)]
    if args.test:
        test = pickle.load(open(Path(args.test).expanduser(), "rb"))["recs"]
        L += ["", f"## 測試集(只看,{len(test)} 句)\n", *head, *table(test, s_pp, args.rounding)]
    if s_pp in (GRID[0], GRID[-1]):
        L.append(f"\n★ s_pp 落在搜尋範圍邊界({GRID[0]}–{GRID[-1]}),結果可能不可靠")
    text = "\n".join(L)
    print(text)
    out = Path(args.out).expanduser() if args.out else Path(args.val).expanduser().parent / "calib_pp.json"
    json.dump({"pp": {"rounding": args.rounding, "s_pp": s_pp}}, open(out, "w"), indent=2)
    (out.parent / "fit_pp.md").write_text(text, encoding="utf-8")
    print(f"\n→ {out}(c_eval.py --calib {out} --variant pp,或直接 --s-pp {s_pp:.2f})")


if __name__ == "__main__":
    main()
