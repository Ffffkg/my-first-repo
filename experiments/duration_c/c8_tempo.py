"""C8:用「驗證集」的波形量測校正母音核的整體長度(入聲、舒聲各一個係數),測試集不參與。不需要 GPU。

C6 用 cum 取整後,入聲與舒聲母音核在波形上都偏短(四縣約 −11 / −7 ms)。這個偏差對照訓練目標看不到
(目標 → 波形之間的落差),所以改用驗證集合成語音的 MFA 量測來估:
    s_入聲 = Σ參考 / Σ預測(驗證集的入聲母音核),s_舒聲 同理;限制在 [0.85, 1.20]。
推論時把時長預測器輸出中「母音核」那一格乘上 s(c_eval.py --s-et / --s-non),子音與停頓不動。

    bash d1_pipeline.sh sixian c8val_c6_sixian c6_tr_dpbig_fair1h_joint_sixian_s1234 --which step40000 \
        --rounding cum --filelist <驗證集清單> --data-yaml ...
    python c8_tempo.py --val-dir ~/hakka_tts/exp_c/eval/c8val_c6_sixian
最後一行印「s_et s_non」給腳本讀;同時寫 <val-dir>/tempo.json。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402


def fit(c, lo=0.85, hi=1.20):
    et = boolcol(c["entering"])
    out = {}
    for key, m in (("s_et", et), ("s_non", ~et)):
        out[key] = float(np.clip(c["ref_ms"][m].sum() / c["pred_ms"][m].sum(), lo, hi))
    return out, et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, help="驗證集的 d1_pipeline 輸出(含 syllables_wave.csv)")
    ap.add_argument("--lo", type=float, default=0.85)
    ap.add_argument("--hi", type=float, default=1.20)
    args = ap.parse_args()
    d = Path(args.val_dir).expanduser()
    c = read_cols(d / "syllables_wave.csv")
    s, et = fit(c, args.lo, args.hi)
    err = c["pred_ms"] - c["ref_ms"]
    print(f"驗證集 {len(set(c['stem']))} 句:入聲 {et.sum()} 個、舒聲 {(~et).sum()} 個母音核")
    print(f"  校正前偏差:入聲 {err[et].mean():+.1f} ms、舒聲 {err[~et].mean():+.1f} ms")
    print(f"  係數:s_et = {s['s_et']:.3f}、s_non = {s['s_non']:.3f}")
    json.dump(s, open(d / "tempo.json", "w"), indent=2)
    print(f"{s['s_et']:.3f} {s['s_non']:.3f}")


if __name__ == "__main__":
    main()
