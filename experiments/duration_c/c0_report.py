"""實驗 C0 的報表:讀 c0_diag.py 的逐音節 CSV,輸出 Markdown(不需要 GPU)。

    python c0_report.py ~/hakka_tts/exp_c/c0/hakka_joint_sixian_ladder_48h_test_all.csv

報表內容(對應手冊第 25.3 節):
  0. 回歸核對(--first 100 的 48h baseline 應重現主表)
  1. 偏差拆解:總偏差 = 目標偏差(MAS−MFA)+ 預測偏差(w−MAS)+ 取整偏差(ceil(w)−w)
  2. blank 分配規則敏感度
  3. 偏差對參考時長分箱(檢驗「短母音都被高估」)
  4. 迴歸:誤差 ~ 參考時長 + 是否入聲 + 語者,看 β(入聲)
  5. 分組預覽(實驗 F1):韻尾、調值、停頓前、性別
  6. 依結果建議下一步
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.durations import RULES  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, delta_ci, excludes_zero, fmt, mean_ci, ols_boot  # noqa: E402

COMPONENTS = [  # (名稱, 預測欄, 參考欄)
    ("總偏差 ceil−MFA(主表 ET-VBias 量的就是這個)", "ceil", "ref"),
    ("├ 目標偏差 MAS−MFA(訓練目標本身)", "mas", "ref"),
    ("├ 預測偏差 w−MAS(時長預測器)", "cont", "mas"),
    ("└ 取整偏差 ceil(w)−w(推論時 ceil)", "ceil", "cont"),
    ("若改用累積四捨五入:cum−MFA", "cum", "ref"),
]


def col(c, src, rule):
    return c["ref_ms"] if src == "ref" else c[f"{src}_{rule}_ms"]


def report(path, n_boot, md):
    c = read_cols(path)
    et = boolcol(c["entering"])
    stems = c["stem"]
    cond, protocol, dialect = c["cond"][0], c["protocol"][0], c["dialect"][0]
    md(f"# C0 診斷:{cond}({Path(path).stem})\n")
    md(f"句數 {len(set(stems))}、入聲 {et.sum()}、舒聲 {(~et).sum()}。CI = 以句子為單位的 cluster bootstrap {n_boot} 次。\n")
    if "utt_phone_ratio" in c:
        _, first = np.unique(stems, return_index=True)
        md(f"合成/真實總長比(只算音素,ceil):每句平均 {c['utt_phone_ratio'][first].mean():.3f}。\n")
    errs = {}
    for name, a, b in COMPONENTS:
        errs[name] = col(c, a, "half") - col(c, b, "half")
    t = UttTable(stems, {k: v for k, v in errs.items()}, {"et": et, "non": ~et})
    idx = boot_idx(t.U, n_boot)

    # 0. 回歸核對
    if protocol == "first100" and cond.endswith("_48h") and dialect in C.MAIN_TABLE_48H:
        exp = C.MAIN_TABLE_48H[dialect]
        got_et, got_non = t.mean(COMPONENTS[0][0], "et"), t.mean(COMPONENTS[0][0], "non")
        ok = abs(got_et - exp["et_vbias"]) <= 1.5 and abs(got_non - exp["nonet_vbias"]) <= 1.5
        md("## 0. 回歸核對(與手冊主表比較)\n")
        md(f"| | 本程式 | 主表 |\n|---|---|---|\n| ET-VBias | {got_et:+.1f} | {exp['et_vbias']:+.1f} |\n"
           f"| NonET-VBias | {got_non:+.1f} | {exp['nonet_vbias']:+.1f} |\n")
        md("→ " + ("✓ 重現主表(差距 ≤ 1.5 ms),後面的拆解可以信任。\n" if ok else
                   "★ 沒有重現主表。先不要往下解讀;請把這份報表貼回對話,一起找出第一個分歧點。\n"))

    # 1. 拆解
    md("## 1. 偏差拆解(blank 規則 = half)\n")
    md("| 成分 | 入聲 (ms) | 舒聲 (ms) | Δ = 入聲 − 舒聲 |\n|---|---|---|---|")
    res = {}
    for name, _, _ in COMPONENTS:
        e, n_, d = mean_ci(t, name, "et", idx), mean_ci(t, name, "non", idx), delta_ci(t, name, idx)
        res[name] = (e, n_, d)
        md(f"| {name} | {fmt(e)} | {fmt(n_)} | {fmt(d)} |")
    tot, tgt, prd, qnt = (res[COMPONENTS[i][0]] for i in range(4))
    gap = abs(tot[0][0] - (tgt[0][0] + prd[0][0] + qnt[0][0]))
    md(f"\n恆等式檢查:目標 + 預測 + 取整 − 總偏差 = {gap:.3f} ms(應為 0)。\n")

    # 2. blank 規則
    md("## 2. blank 分配規則敏感度\n")
    md("| 規則 | 總偏差 入聲 | 總偏差 ΔBias | 目標偏差 入聲 | 目標偏差 ΔBias |\n|---|---|---|---|---|")
    flip = False
    for rule in RULES:
        e1 = col(c, "ceil", rule) - c["ref_ms"]
        e2 = col(c, "mas", rule) - c["ref_ms"]
        tt = UttTable(stems, {"tot": e1, "tgt": e2}, {"et": et, "non": ~et})
        d1 = delta_ci(tt, "tot", idx)
        flip |= not (d1[1] > 0)
        md(f"| {rule} | {fmt(mean_ci(tt, 'tot', 'et', idx))} | {fmt(d1)} | {fmt(mean_ci(tt, 'tgt', 'et', idx))} | {fmt(delta_ci(tt, 'tgt', idx))} |")
    md("\n(none = 不把 blank 算給任何音素,是母音時長的下界。)\n")
    blank_l = col(c, "mas", "right") - col(c, "mas", "none")   # 母音前面那個 blank
    blank_r = col(c, "mas", "left") - col(c, "mas", "none")    # 母音後面那個 blank
    md(f"MAS 在母音前後的 blank 平均各分到:入聲 前 {blank_l[et].mean():.1f} / 後 {blank_r[et].mean():.1f} ms;"
       f"舒聲 前 {blank_l[~et].mean():.1f} / 後 {blank_r[~et].mean():.1f} ms。"
       "blank 分到越多,「母音時長」就越取決於 blank 怎麼分(約定),需要實驗 D 用波形確認。\n")
    d_rules = []
    for rule in RULES:
        e1 = col(c, "ceil", rule) - c["ref_ms"]
        tt = UttTable(stems, {"tot": e1}, {"et": et, "non": ~et})
        d_rules.append(delta_ci(tt, "tot", idx))

    # 3. 分箱
    md("## 3. 總偏差對參考母音時長分箱\n")
    edges = [0, 40, 60, 80, 100, 120, 140, 170, 200, 1e9]
    e = errs[COMPONENTS[0][0]]
    md("| 參考時長 (ms) | 入聲 平均誤差 (n) | 舒聲 平均誤差 (n) |\n|---|---|---|")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c["ref_ms"] >= lo) & (c["ref_ms"] < hi)
        cells = []
        for g in (et, ~et):
            k = m & g
            cells.append(f"{e[k].mean():+.1f} ({k.sum()})" if k.sum() >= 10 else f"— ({k.sum()})")
        md(f"| {lo:.0f}–{'' if hi > 1e8 else f'{hi:.0f}'} | {cells[0]} | {cells[1]} |")
    md("\n如果同樣長度的入聲與舒聲誤差差不多,「入聲額外偏長」主要是「短母音都被高估」(回歸平均)。\n")

    # 4. 入聲特有,還是短的音節類型都被高估?
    md("## 4. 入聲特有,還是「短的音節類型」都被高估?\n")
    md("注意:第 3 節依「每個音節自己的參考時長」分箱,含有回歸平均的假象。預測器只看文字,猜不到同一類音節"
       "每次說話的長短變化,所以即使預測器完全沒有偏差,誤差也必然隨參考時長下降(參考越長、誤差越負)。"
       "因此改用「音節類型的平均時長」當共變數,類型 = (母音, 韻尾):\n")
    md("誤差 = β0 + β1·類型平均時長 + β2·是否入聲 + 語者。β2 回答:入聲類型是否比「同樣短的其他類型」更被高估。\n")
    spk = c["spk"].astype(int)
    spks = sorted(set(spk))
    dummies = [(spk == s_).astype(float) for s_ in spks[1:]]
    beta2 = {}
    has_type = "nuc_phone" in c and any(str(v) for v in c["nuc_phone"])
    if has_type:
        types = np.array([f"{a}|{b}" for a, b in zip(c["nuc_phone"], c["coda"])], dtype=object)
        tmean = {t: c["ref_ms"][types == t].mean() for t in set(types)}
        x = np.array([tmean[t] for t in types])
        X = np.column_stack([np.ones_like(x), x - x.mean(), et.astype(float)] + dummies)
        md("| 依變數 | β1 每 ms 類型平均時長 | β2 入聲 (ms) |\n|---|---|---|")
        for label, y in (("總偏差", errs[COMPONENTS[0][0]]), ("目標偏差", errs[COMPONENTS[1][0]])):
            b, lo, hi = ols_boot(y, X, stems, n_boot=min(n_boot, 2000))
            beta2[label] = (b[2], lo[2], hi[2])
            md(f"| {label} | {b[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] | {fmt((b[2], lo[2], hi[2]))} |")
        big = [t for t in set(types) if (types == t).sum() >= 20]
        r_et = [tmean[t] for t in big if et[types == t][0]]
        r_non = [tmean[t] for t in big if not et[types == t][0]]
        if r_et and r_non:
            overlap = sum(min(r_et) <= v <= max(r_et) for v in r_non)
            md(f"\n類型平均時長範圍(n ≥ 20):入聲 {min(r_et):.0f}–{max(r_et):.0f} ms;舒聲 {min(r_non):.0f}–{max(r_non):.0f} ms;"
               f"落在入聲範圍內的舒聲類型 {overlap} 個。" + (" ★ 重疊太少,β2 是外推,解讀要保守。" if overlap < 3 else ""))
        md("\n各類型(n ≥ 20,依平均參考時長排序):\n")
        md("| 類型 (母音,韻尾) | 入聲 | n | 平均參考 (ms) | 平均總偏差 (ms) |\n|---|---|---|---|---|")
        e = errs[COMPONENTS[0][0]]
        rows_t = []
        for t in set(types):
            m = types == t
            if m.sum() >= 20:
                rows_t.append((tmean[t], t, bool(et[m][0]), int(m.sum()), e[m].mean()))
        for mref, t, ent, n_, me in sorted(rows_t)[:40]:
            v, cd = t.split("|")
            md(f"| {v}, {cd or '—'} | {'✓' if ent else ''} | {n_} | {mref:.0f} | {me:+.1f} |")
    else:
        md("(這份 CSV 沒有 nuc_phone 欄位,請用新版 c0_diag.py 重跑。)")
    ref_c = c["ref_ms"] - c["ref_ms"].mean()
    Xi = np.column_stack([np.ones_like(ref_c), ref_c, et.astype(float)] + dummies)
    b, lo, hi = ols_boot(errs[COMPONENTS[0][0]], Xi, stems, n_boot=min(n_boot, 2000))
    md(f"\n參考用(含上述假象,不要據此下結論):以個別參考時長為共變數時 β2 = {fmt((b[2], lo[2], hi[2]))} ms。\n")

    # 5. 分組
    md("## 5. 分組預覽(實驗 F1):總偏差的入聲 ET-VBias\n")
    md("| 分組 | 值 | n | ET-VBias |\n|---|---|---|---|")
    for key, label in (("coda", "韻尾"), ("tone", "調值"), ("prepausal", "停頓前"), ("gender", "性別")):
        for v in sorted(set(c[key][et])):
            m = et & (c[key] == v)
            if m.sum() < 15:
                continue
            tt = UttTable(stems[m], {"e": e[m]}, {"all": np.ones(m.sum(), bool)})
            ii = boot_idx(tt.U, min(n_boot, 2000))
            shown = int(v) if isinstance(v, float) and float(v).is_integer() else v
            md(f"| {label} | {shown} | {m.sum()} | {fmt(mean_ci(tt, 'e', 'all', ii))} |")
    md("\n(分組是事先登記的四個因素,全部列出;樣本少於 15 的組不列。)\n")

    # 6. 建議
    md("## 6. 依結果建議的下一步(手冊第 25.3 節決策表)\n")
    d_tot, d_tgt, d_prd = tot[2][0], tgt[2][0], prd[2][0]
    tips = []
    if flip:
        tips.append("ΔBias 在某些 blank 規則下 CI 碰到 0 或翻轉 → 結論對量測方法敏感,**先做實驗 D**(從波形量)。")
    else:
        lo_d, hi_d = min(d[0] for d in d_rules), max(d[0] for d in d_rules)
        if hi_d > 2 * lo_d:
            tips.append(f"ΔBias 在四種 blank 規則下都 > 0(方向穩健),但幅度從 {lo_d:+.1f} 到 {hi_d:+.1f} ms → "
                        "真正的幅度要靠**實驗 D**(波形量測)決定;C2 讓 blank 只佔 1 frame,可以消除這個約定問題。")
    if excludes_zero(tgt[2]) and d_tot > 0 and d_tgt / d_tot >= 0.5:
        tips.append(f"ΔBias 有 {100 * d_tgt / d_tot:.0f}% 已經存在於訓練目標(MAS)→ **優先做 C2**(MFA 時長監督)。")
    if excludes_zero(prd[2]) and d_tot > 0 and d_prd / d_tot >= 0.5:
        tips.append(f"ΔBias 有 {100 * d_prd / d_tot:.0f}% 來自時長預測器 → **做 C3**(預測器改良)。")
    if prd[0][2] < 0 and prd[1][2] < 0:
        tips.append(f"時長預測器整體偏短(入聲 {prd[0][0]:+.1f}、舒聲 {prd[1][0]:+.1f} ms,log 域均方誤差的典型現象),"
                    f"目前被 ceil 取整的 {qnt[1][0]:+.1f} ms 部分抵銷。")
    cum_non = res[COMPONENTS[4][0]][1][0]
    if abs(qnt[0][0]) >= 3 and abs(cum_non) > 5:
        tips.append(f"★ **不要單獨改用累積四捨五入**:ceil 正在抵銷預測器的偏短,改用 cum 後舒聲會變成 {cum_non:+.1f} ms。"
                    "C1 要看 class_cum(入聲、舒聲各自縮放)而不是 cum。")
    elif abs(qnt[0][0]) >= 3:
        tips.append(f"ceil 取整讓入聲母音平均多 {qnt[0][0]:+.1f} ms(舒聲 {qnt[1][0]:+.1f} ms)→ C1 改用累積四捨五入可去掉這部分。")
    if "總偏差" in beta2:
        b2 = beta2["總偏差"]
        if b2[1] > 0:
            tips.append(f"控制音節類型的平均時長後,入聲仍多被高估 {b2[0]:+.1f} ms → 有**入聲特有**的偏差。")
        elif b2[2] < 0:
            tips.append(f"控制音節類型的平均時長後,入聲反而比同樣短的其他類型少被高估({b2[0]:+.1f} ms)→ ΔBias 主要是"
                        "「短的音節類型都被高估」;論文對 ΔBias 的解讀要補上這一點。")
        else:
            tips.append("控制音節類型的平均時長後 β2 的 CI 含 0 → ΔBias 可由「短的音節類型都被高估」解釋;"
                        "論文對 ΔBias 的解讀要補上這一點。")
    if not tips:
        tips.append("沒有單一來源佔多數:C2、C3 都可能需要;先做 C1 當基準線,再做 C2(變因較單純)。")
    for s_ in tips:
        md(f"- {s_}")
    md("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    for p in args.csv:
        lines = []
        report(p, args.n_boot, lines.append)
        text = "\n".join(lines)
        print(text)
        out = Path(p).with_suffix(".md")
        out.write_text(text, encoding="utf-8")
        print(f"\n→ 已寫入 {out}\n")


if __name__ == "__main__":
    main()
