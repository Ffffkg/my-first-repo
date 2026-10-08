"""逐句紀錄 → 逐音節表 → ET 指標摘要。只用 numpy,可在沒有 GPU 的環境跑。

逐句紀錄(rec)是一個 dict,由 c0_diag.py / c_eval.py 以 pickle 存下:
    stem, spk, dialect, T(真實梅爾長度,可為 None), w(時長預測器的連續輸出,長度 2n+1),
    mas(teacher-forced MAS frame 數,可為 None), n(非 blank 符號數), syllables(Syllable 的 dict 列表)
"""
import numpy as np

from .config import HOP_MS, SPK_GENDER
from .durations import matcha_frames, merge_blank
from .stats import UttTable, boot_idx, delta_ci, fmt, mean_ci, pearson
from .structure import base_of


def scaled_w(rec, s_et=1.0, s_non=1.0):
    """只縮放母音核符號本身(不動相鄰 blank 與子音)。"""
    w = np.array(rec["w"], dtype=float)
    for s in rec["syllables"]:
        w[2 * s["nuc"] + 1] *= s_et if s["entering"] else s_non
    return w


def pred_frames(rec, rounding="ceil", length_scale=1.0, s_et=1.0, s_non=1.0):
    return matcha_frames(scaled_w(rec, s_et, s_non), length_scale, rounding)


def base_row(rec, s):
    return {
        "dialect": rec["dialect"], "stem": rec["stem"], "spk": rec["spk"],
        "gender": SPK_GENDER.get(rec["spk"], "?"), "syl_idx": s["idx"], "word": s["word"],
        "tone": s["tone"], "entering": int(s["entering"]), "coda": s["coda"],
        "prepausal": int(s["prepausal"]), "ref_ms": round(s["ref_ms"], 3),
        "nuc_phone": base_of(rec["names"][s["nuc"]]) if rec.get("names") else "",
    }


def eval_rows(recs, rule="half", **kw):
    """一個系統(一組推論設定)的逐音節表:ref_ms 與 pred_ms。"""
    rows = []
    for rec in recs:
        merged = merge_blank(pred_frames(rec, **kw), rule) * HOP_MS
        for s in rec["syllables"]:
            r = base_row(rec, s)
            r["pred_ms"] = round(float(merged[s["nuc"]]), 3)
            rows.append(r)
    return rows


def summarize(cols, pred="pred_ms", n_boot=10000, seed=42):
    """cols:read_cols() 的結果或同結構的 dict。回傳 (dict, 文字)。"""
    et = np.asarray(cols["entering"], float) > 0.5
    ref = np.asarray(cols["ref_ms"], float)
    p = np.asarray(cols[pred], float)
    err = p - ref
    t = UttTable(cols["stem"], {"err": err, "abs": np.abs(err), "ref": ref}, {"et": et, "non": ~et})
    idx = boot_idx(t.U, n_boot, seed)
    out = {
        "n_utt": t.U, "n_et": int(et.sum()), "n_non": int((~et).sum()),
        "et_vde": t.mean("abs", "et"),
        "et_vbias": mean_ci(t, "err", "et", idx),
        "ret_vbias_pct": 100 * err[et].sum() / ref[et].sum(),
        "rho": pearson(p[et], ref[et]),
        "nonet_vbias": mean_ci(t, "err", "non", idx),
        "delta_bias": delta_ci(t, "err", idx),
    }
    txt = (f"句數 {out['n_utt']}、入聲 {out['n_et']}、舒聲 {out['n_non']}\n"
           f"  ET-VDE      {out['et_vde']:.1f} ms\n"
           f"  ET-VBias    {fmt(out['et_vbias'])} ms   (rET-VBias {out['ret_vbias_pct']:+.1f}%)\n"
           f"  ρ           {out['rho']:.3f}\n"
           f"  NonET-VBias {fmt(out['nonet_vbias'])} ms\n"
           f"  ΔBias       {fmt(out['delta_bias'])} ms")
    return out, txt
