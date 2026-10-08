"""兩個系統的配對比較 + 事先登記的成功標準檢查(手冊第 25.8 節)。

輸入可以是 c_eval.py 的輸出資料夾(含 syllables.csv,可能有 mcd.csv),或 c1_calibrate.py 的變體 CSV。
兩邊只保留共同的音節(以 檔名 + 音節序號 對齊),以句子為單位配對重抽。

    python c_compare.py --base ~/hakka_tts/exp_c/eval/hakka_joint_sixian_ladder_1h__base \
                        --method ~/hakka_tts/exp_c/eval/c2_mfadur_joint_sixian_1h__base

門檻預設值是手冊列的「建議值」,啟動實驗前與老師確認後,用參數改成登記的數字。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, ci, delta_ci, fmt, mean_ci, pearson  # noqa: E402


def load(p):
    p = Path(p).expanduser()
    if p.is_dir():
        syl = read_cols(p / "syllables.csv")
        mcd = read_cols(p / "mcd.csv") if (p / "mcd.csv").exists() else None
        return syl, mcd, p.name
    return read_cols(p), None, p.stem


def join_syllables(a, b):
    ka = {(s, int(i)): k for k, (s, i) in enumerate(zip(a["stem"], a["syl_idx"]))}
    kb = {(s, int(i)): k for k, (s, i) in enumerate(zip(b["stem"], b["syl_idx"]))}
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka[k] for k in common])
    ib = np.array([kb[k] for k in common])
    assert np.allclose(a["ref_ms"][ia], b["ref_ms"][ib]), "兩邊的參考時長不同:是不是不同腔調或不同 TextGrid?"
    return ia, ib, len(ka), len(kb)


def per_utt(m, key):
    """mcd.csv → 每句 5 次生成平均。"""
    out = {}
    for s, g, v in zip(m["stem"], m["gender"], m[key]):
        out.setdefault(s, [g, []])[1].append(v)
    return {s: (g, float(np.mean(v))) for s, (g, v) in out.items()}


def strat_boot_diff(base, meth, n_boot, seed=42):
    """依性別分層、以句子為單位重抽的配對差(方法 − 基準)。"""
    stems = sorted(set(base) & set(meth))
    d = np.array([meth[s][1] - base[s][1] for s in stems])
    g = np.array([base[s][0] for s in stems])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(g == x) for x in sorted(set(g))]
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([grp[rng.integers(0, len(grp), len(grp))] for grp in groups])
        draws.append(d[pick].mean())
    return ci(d.mean(), np.array(draws)), len(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--max-nonet-change", type=float, default=5.0, help="守門:NonET-VBias 變化上限 (ms)")
    ap.add_argument("--max-mcd-increase", type=float, default=0.1, help="守門:MCD 增加上限 (dB)")
    ap.add_argument("--max-f0-increase", type=float, default=0.05, help="守門:F0-RMSE 增加上限 (st)")
    ap.add_argument("--out", help="Markdown 輸出路徑")
    args = ap.parse_args()

    a, ma, na = load(args.base)
    b, mb, nb = load(args.method)
    ia, ib, n_a, n_b = join_syllables(a, b)
    et = boolcol(a["entering"][ia])
    ref = a["ref_ms"][ia]
    ea, eb = a["pred_ms"][ia] - ref, b["pred_ms"][ib] - ref
    t = UttTable(a["stem"][ia], {"a": ea, "b": eb, "d": eb - ea, "aa": np.abs(ea), "ab": np.abs(eb)},
                 {"et": et, "non": ~et})
    idx = boot_idx(t.U, args.n_boot)

    L = [f"# 配對比較:{nb}(方法)vs {na}(基準)\n",
         f"共同音節 {len(ia)}(基準 {n_a}、方法 {n_b}),句數 {t.U}。CI = 以句子為單位配對重抽 {args.n_boot} 次。\n",
         "| 指標 | 基準 | 方法 | 方法 − 基準 |", "|---|---|---|---|"]
    r_et = (mean_ci(t, "a", "et", idx), mean_ci(t, "b", "et", idx), mean_ci(t, "d", "et", idx))
    r_non = (mean_ci(t, "a", "non", idx), mean_ci(t, "b", "non", idx), mean_ci(t, "d", "non", idx))
    r_dl = (delta_ci(t, "a", idx), delta_ci(t, "b", idx), delta_ci(t, "d", idx))
    rho_a, rho_b = pearson(a["pred_ms"][ia][et], ref[et]), pearson(b["pred_ms"][ib][et], ref[et])
    L.append(f"| ET-VDE (ms) | {t.mean('aa', 'et'):.1f} | {t.mean('ab', 'et'):.1f} | {t.mean('ab', 'et') - t.mean('aa', 'et'):+.1f} |")
    L.append(f"| ET-VBias (ms) | {fmt(r_et[0])} | {fmt(r_et[1])} | {fmt(r_et[2])} |")
    L.append(f"| NonET-VBias (ms) | {fmt(r_non[0])} | {fmt(r_non[1])} | {fmt(r_non[2])} |")
    L.append(f"| ΔBias (ms) | {fmt(r_dl[0])} | {fmt(r_dl[1])} | {fmt(r_dl[2])} |")
    L.append(f"| ρ | {rho_a:.3f} | {rho_b:.3f} | {rho_b - rho_a:+.3f} |")

    crit = [("主要", "ΔBias 差值 CI 上界 < 0", r_dl[2][2] < 0, fmt(r_dl[2])),
            ("次要", "|ET-VBias| 下降", abs(r_et[1][0]) < abs(r_et[0][0]), f"{r_et[0][0]:+.1f} → {r_et[1][0]:+.1f}"),
            ("次要", "ρ 不下降", rho_b >= rho_a, f"{rho_a:.3f} → {rho_b:.3f}"),
            ("守門", f"|NonET-VBias 變化| ≤ {args.max_nonet_change} ms", abs(r_non[2][0]) <= args.max_nonet_change,
             f"{r_non[2][0]:+.1f}")]
    if ma is not None and mb is not None:
        for key, lim, unit in (("mcd", args.max_mcd_increase, "dB"), ("f0_rmse", args.max_f0_increase, "st")):
            if key in ma and key in mb:
                c, n = strat_boot_diff(per_utt(ma, key), per_utt(mb, key), args.n_boot)
                L.append(f"| {key} 差值({n} 句,5 次生成平均) | | | {fmt(c, 3)} |")
                crit.append(("守門", f"{key} 增加 ≤ {lim} {unit}", c[0] <= lim, f"{c[0]:+.3f}"))
    else:
        crit.append(("守門", "MCD(未提供 mcd.csv,用 c_eval.py --synth 產生)", None, "—"))

    L += ["", "## 成功標準(手冊第 25.8 節,門檻為建議值)\n", "| 類別 | 標準 | 結果 | 數值 |", "|---|---|---|---|"]
    for cat, desc, ok, val in crit:
        L.append(f"| {cat} | {desc} | {'—' if ok is None else ('✓ 通過' if ok else '✗ 未通過')} | {val} |")
    L += ["", "提醒:單一訓練種子的結果只算初步;勝出的方法要補種子 123、2026,並用專家標註參考與實驗 D 的波形量測複核。"]
    text = "\n".join(L)
    print(text)
    meth = Path(args.method).expanduser()
    if args.out:
        out = Path(args.out)
    elif meth.is_dir():
        out = meth / f"compare_vs_{na}.md"
    else:
        out = meth.with_name(f"{meth.stem}.vs_{na}.md")
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
