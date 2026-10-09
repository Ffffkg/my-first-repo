"""實驗 C1:推論時校正(手冊第 25.4 節)。不需要 GPU,讀 c0_diag.py 的 pickle。

在「驗證集」上估參數,在「測試集」上評估,絕不用測試集估參數:
    base       Matcha 原本的 ceil(w)                                  (基準線)
    cum        改用累積四捨五入                                        (只去掉取整偏差)
    global     ceil + 全域 length_scale,讓總長回到真人水準              (整體語速校正)
    class_ceil ceil + 入聲/舒聲母音核各自的縮放係數
    class_cum  cum  + 入聲/舒聲母音核各自的縮放係數

    python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian --split val
    python c1_calibrate.py --val ~/hakka_tts/exp_c/c0/hakka_joint_sixian_ladder_48h_val_all.pkl \
                           --test ~/hakka_tts/exp_c/c0/hakka_joint_sixian_ladder_48h_test_all.pkl

輸出 ~/hakka_tts/exp_c/c1/<cond>/:各變體的逐音節 CSV(給 c_compare.py)、calib.json(給 c_eval.py --calib 合成用)。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import eval_rows, pred_frames, summarize  # noqa: E402
from hakka_c.csvio import read_cols, write_rows  # noqa: E402
from hakka_c.durations import merge_blank  # noqa: E402
from hakka_c.stats import fmt  # noqa: E402

GRID = np.round(np.arange(0.30, 1.3001, 0.01), 2)


def class_bias(recs, entering, rounding, s):
    """只縮放某一類母音核時,該類的平均偏差(ms)。"""
    err, n = 0.0, 0
    kw = {"s_et": s} if entering else {"s_non": s}
    for rec in recs:
        merged = merge_blank(pred_frames(rec, rounding, **kw)) * C.HOP_MS
        for syl in rec["syllables"]:
            if syl["entering"] == entering:
                err += merged[syl["nuc"]] - syl["ref_ms"]
                n += 1
    return err / max(n, 1)


def fit_scale(recs, entering, rounding):
    biases = np.array([class_bias(recs, entering, rounding, s) for s in GRID])
    k = int(np.argmin(np.abs(biases)))
    return float(GRID[k]), float(biases[k])


def length_ratio(recs, **kw):
    return sum(pred_frames(r, **kw).sum() for r in recs) / sum(r["T"] for r in recs)


def fit_global(recs):
    grid = np.round(np.arange(0.60, 1.2001, 0.005), 3)
    r = np.array([length_ratio(recs, rounding="ceil", length_scale=g) for g in grid])
    k = int(np.argmin(np.abs(r - 1.0)))
    return float(grid[k]), float(r[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="c0_diag.py --split val 的 .pkl")
    ap.add_argument("--test", required=True, help="c0_diag.py --split test 的 .pkl")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    val, test = pickle.load(open(args.val, "rb")), pickle.load(open(args.test, "rb"))
    assert val["ckpt"] == test["ckpt"], "val 與 test 必須來自同一個 checkpoint"
    cond = test["cond"]
    print(f"{cond}:驗證集 {len(val['recs'])} 句用來估參數,測試集 {len(test['recs'])} 句用來評估")

    calib = {"cond": cond, "ckpt": test["ckpt"], "fitted_on": args.val}
    for rounding in ("ceil", "cum"):
        s_et, b_et = fit_scale(val["recs"], True, rounding)
        s_non, b_non = fit_scale(val["recs"], False, rounding)
        calib[f"class_{rounding}"] = {"rounding": rounding, "s_et": s_et, "s_non": s_non}
        print(f"  [{rounding}] s_et = {s_et:.2f}(驗證集殘餘偏差 {b_et:+.1f} ms),"
              f"s_non = {s_non:.2f}({b_non:+.1f} ms)")
    g, rg = fit_global(val["recs"])
    calib["global"] = {"rounding": "ceil", "length_scale": g}
    print(f"  global length_scale = {g:.3f}(驗證集總長比 {rg:.3f})")
    calib["base"] = {"rounding": "ceil"}
    calib["cum"] = {"rounding": "cum"}
    for k in ("class_ceil", "class_cum"):
        for sk in ("s_et", "s_non"):
            if np.isclose(calib[k][sk], GRID[0]) or np.isclose(calib[k][sk], GRID[-1]):
                print(f"  ★ {k} 的 {sk} 碰到搜尋範圍邊界({GRID[0]}–{GRID[-1]}),結果不可信,請擴大 GRID")

    out_dir = C.OUT_ROOT / "c1" / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# C1 推論時校正:{cond}\n", "| 變體 | 參數 | 總長比 | ET-VBias | ρ | NonET-VBias | ΔBias |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "cum", "global", "class_ceil", "class_cum"):
        p = calib[name]
        kw = {"rounding": p["rounding"], "length_scale": p.get("length_scale", 1.0),
              "s_et": p.get("s_et", 1.0), "s_non": p.get("s_non", 1.0)}
        rows = eval_rows(test["recs"], **kw)
        for r in rows:
            r["cond"], r["variant"] = cond, name
        write_rows(out_dir / f"{name}.csv", rows)
        s, txt = summarize(read_cols(out_dir / f"{name}.csv"), n_boot=args.n_boot)
        ratio = length_ratio(test["recs"], **kw)
        par = ", ".join(f"{k}={v}" for k, v in p.items())
        lines.append(f"| {name} | {par} | {ratio:.3f} | {fmt(s['et_vbias'])} | {s['rho']:.3f} | {fmt(s['nonet_vbias'])} | {fmt(s['delta_bias'])} |")
        print(f"\n[{name}] {par}  總長比 {ratio:.3f}\n{txt}")
    lines += ["", "注意:C1 直接改了時長,所以「從時長量到的偏差變小」幾乎是必然的;ρ 不應改變。",
              "效果必須再用實驗 D(波形量測)與 E(聽測)確認。"]
    (out_dir / "c1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump(calib, open(out_dir / "calib.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n→ {out_dir}/c1_summary.md、calib.json、各變體 CSV")
    print(f"   兩兩比較:python c_compare.py --base {out_dir}/base.csv --method {out_dir}/class_cum.csv")


if __name__ == "__main__":
    main()
