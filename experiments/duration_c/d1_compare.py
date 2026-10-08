"""實驗 D1 第 2 步:比較「從波形(MFA 對齊合成語音)量到的母音時長」與「attn 推得的母音時長」。

回答三件事:
  1. 合成語音的入聲母音,從波形量也偏長嗎?(ET-VBias、ΔBias 的波形版)
  2. attn 版的數字差多少?哪一種 blank 分配規則最接近波形?
  3. 母音前後的 blank,實際上有多少比例落在波形量到的母音裡?

    python d1_compare.py --eval-dir ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base --dialect sixian

輸出(都在 eval-dir):
  d1_rows.csv            逐音節 × 生成種子
  syllables_wave.csv     逐音節(5 次生成平均),pred_ms = 波形量到的母音時長 → 可直接給 c_compare.py
  syllables_attn_d1.csv  同一批音節的 attn 版(half 規則),pred_ms = attn 母音時長
  d1_summary.md
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import base_row, summarize  # noqa: E402
from hakka_c.csvio import read_cols, write_rows  # noqa: E402
from hakka_c.durations import RULES, merge_blank  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, fmt, mean_ci, pearson  # noqa: E402
from hakka_c.structure import StructError, build_utt  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402

SPK_ID = {"XF": 0, "XM": 1, "HF": 2, "HM": 3}


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--tg-dir", help="mfa align 的輸出(預設 <eval-dir>/mfa_tg)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    ev = Path(args.eval_dir).expanduser()
    tg_dir = Path(args.tg_dir).expanduser() if args.tg_dir else ev / "mfa_tg"
    tgs = {p.stem: p for p in tg_dir.rglob("*.TextGrid")}
    n_wav = len(list((ev / "wav").glob("*_g*.wav")))
    if not tgs:
        sys.exit(f"★ {tg_dir} 沒有 TextGrid:mfa align 跑完了嗎?")

    rows, why = [], Counter()
    for name, tgp in sorted(tgs.items()):
        stem, seed = name.rsplit("_g", 1)
        bj = ev / "bounds" / f"{stem}.json"
        if not bj.exists():
            why["沒有 bounds"] += 1
            continue
        b = json.load(open(bj, encoding="utf-8"))
        try:
            syn = build_utt(stem, b["names"], read_textgrid(tgp))
        except (StructError, KeyError) as e:
            why[str(e)[:50]] += 1
            continue
        wave_iv = {i: iv for i, iv in zip(syn.phone_sym, syn.phone_iv)}
        frames = np.array(b["frames"], float)
        st, en = np.array(b["start_s"]), np.array(b["end_s"])
        attn = {rule: merge_blank(frames, rule) * C.HOP_MS for rule in RULES}
        real = {s["idx"]: s for s in b["syllables"]}
        rec = {"dialect": args.dialect, "stem": stem, "spk": SPK_ID.get(stem.split("-")[0], -1), "names": b["names"]}
        for s in syn.syllables:
            if s.idx not in real:
                continue
            r = real[s.idx]
            w0, w1 = wave_iv[s.nuc]
            j = 2 * s.nuc + 1                       # 母音核在 intersperse 序列中的位置
            frac = {}
            for key, k in (("pb", j - 1), ("pv", j), ("pa", j + 1)):
                L = en[k] - st[k]
                frac[key] = overlap(st[k], en[k], w0, w1) / L if L > 0 else np.nan
            row = {**base_row(rec, r), "gen_seed": int(seed), "wave_ms": round(s.ref_ms, 3),
                   **{f"attn_{rule}_ms": round(float(attn[rule][s.nuc]), 3) for rule in RULES},
                   "blank_before_ms": round((en[j - 1] - st[j - 1]) * 1000, 2),
                   "blank_after_ms": round((en[j + 1] - st[j + 1]) * 1000, 2), **frac}
            rows.append(row)
    if not rows:
        sys.exit(f"★ 沒有任何句子對得上:{dict(why)}")
    write_rows(ev / "d1_rows.csv", rows)

    # 逐音節(跨生成種子平均)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["stem"], r["syl_idx"])].append(r)
    wave_rows, attn_rows = [], []
    for (stem, idx), rs in sorted(agg.items()):
        base = {k: rs[0][k] for k in ("dialect", "stem", "spk", "gender", "syl_idx", "word", "tone", "entering",
                                      "coda", "prepausal", "ref_ms", "nuc_phone")}
        wave_rows.append({**base, "pred_ms": round(float(np.mean([r["wave_ms"] for r in rs])), 3), "n_gen": len(rs)})
        attn_rows.append({**base, "pred_ms": rs[0]["attn_half_ms"]})
    write_rows(ev / "syllables_wave.csv", wave_rows)
    write_rows(ev / "syllables_attn_d1.csv", attn_rows)

    c = read_cols(ev / "d1_rows.csv")
    et = c["entering"] > 0.5
    L = [f"# D1 波形量測:{ev.name}\n",
         f"合成音檔 {n_wav} 個、MFA TextGrid {len(tgs)} 個、對上 {len(set(zip(c['stem'], c['gen_seed'])))} 個"
         f"({len(rows)} 個音節 × 種子)。失敗原因:{dict(why) or '無'}\n",
         "## 1. attn 推得的母音時長 vs 波形量到的母音時長(同一個合成音檔)\n",
         "| blank 規則 | 入聲:波形 − attn (ms) | 舒聲:波形 − attn (ms) |", "|---|---|---|"]
    cols = {rule: c["wave_ms"] - c[f"attn_{rule}_ms"] for rule in RULES}
    t = UttTable(c["stem"], cols, {"et": et, "non": ~et})
    idx = boot_idx(t.U, args.n_boot)
    best = {}
    for rule in RULES:
        e_, n_ = mean_ci(t, rule, "et", idx), mean_ci(t, rule, "non", idx)
        best[rule] = abs(e_[0]) + abs(n_[0])
        L.append(f"| {rule} | {fmt(e_)} | {fmt(n_)} |")
    rbest = min(best, key=best.get)
    d = cols["half"]
    L.append(f"\n最接近波形的 blank 規則:**{rbest}**。half 規則下 |波形 − attn| 平均 {np.abs(d).mean():.1f} ms,"
             f"r = {pearson(c['wave_ms'], c['attn_half_ms']):.3f}。\n")
    L += ["## 2. 母音前後的 blank 有多少落在波形量到的母音裡\n",
          "| | blank(前)平均長度 | 落在母音內比例 | 母音符號本身落在母音內比例 | blank(後)平均長度 | 落在母音內比例 |",
          "|---|---|---|---|---|---|"]
    for lab, m in (("入聲", et), ("舒聲", ~et)):
        L.append(f"| {lab} | {np.nanmean(c['blank_before_ms'][m]):.1f} ms | {np.nanmean(c['pb'][m]):.2f} | "
                 f"{np.nanmean(c['pv'][m]):.2f} | {np.nanmean(c['blank_after_ms'][m]):.1f} ms | {np.nanmean(c['pa'][m]):.2f} |")
    L.append("\n(half 規則假設前後 blank 各有一半屬於母音;比例 0.5 表示這個假設對。)\n")

    sw, txt_w = summarize(read_cols(ev / "syllables_wave.csv"), n_boot=args.n_boot)
    sa, txt_a = summarize(read_cols(ev / "syllables_attn_d1.csv"), n_boot=args.n_boot)
    L += ["## 3. 入聲時長指標:attn 版 vs 波形版(同一批音節;波形版為生成種子平均)\n",
          "| 指標 | attn(half) | 波形 |", "|---|---|---|",
          f"| ET-VBias (ms) | {fmt(sa['et_vbias'])} | {fmt(sw['et_vbias'])} |",
          f"| NonET-VBias (ms) | {fmt(sa['nonet_vbias'])} | {fmt(sw['nonet_vbias'])} |",
          f"| ΔBias (ms) | {fmt(sa['delta_bias'])} | {fmt(sw['delta_bias'])} |",
          f"| ρ | {sa['rho']:.3f} | {sw['rho']:.3f} |",
          f"| rET-VBias | {sa['ret_vbias_pct']:+.1f}% | {sw['ret_vbias_pct']:+.1f}% |", ""]
    dw = sw["delta_bias"]
    if dw[1] > 0:
        L.append("→ 從波形量,入聲母音仍然「額外」偏長(ΔBias CI 不含 0):ICASSP 的結論在波形層級成立。")
    elif dw[2] < 0:
        L.append("→ 從波形量,入聲母音反而相對較短:attn 版的 ΔBias 主要來自 blank 分配的約定,論文結論需要修正。")
    else:
        L.append("→ 從波形量,ΔBias 的 CI 含 0:attn 版看到的入聲額外偏長,在波形層級不顯著。")
    L.append("\n注意:MFA 是在真人錄音上訓練的,對合成語音可能較不準;關鍵比較請再用專家手標的合成片段(D3)複核。")
    text = "\n".join(L)
    (ev / "d1_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {ev}/d1_summary.md、syllables_wave.csv(可給 c_compare.py)")


if __name__ == "__main__":
    main()
