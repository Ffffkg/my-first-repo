"""S1:入聲母音的時長錯誤,來自「整個音節的長度」還是「音節內部的時間分配」?不需要訓練。

用來決定「音節→音素雙層時長預測器」值不值得做。對每個音節,從波形量(MFA 重新對齊合成語音,
與論文評估相同)四個數:合成與真人的音節總長 S、母音核長 V。再做兩種離線的反事實調整:

  (a) 音節總長換成真人的、母音比例維持合成的:V_a = V_syn × S_ref / S_syn
      → 「完美的音節長度預測器」最多能救回多少命中率(雙層模型的上限)
  (b) 母音比例換成真人的、音節總長維持合成的:V_b = S_syn × V_ref / S_ref
      → 錯誤有多少來自音節內部的分配(聲母／母音／韻尾之間)

兩者都用到真人時長,只是診斷,不是可以報告為模型成績的結果。
另外把母音的對數誤差拆成「音節總長誤差」與「母音比例誤差」兩部分,看哪一部分的變異較大:
  log(V_syn/V_ref) = log(S_syn/S_ref) + log(share_syn/share_ref)

音節的範圍 = 該音節第一個音素起點到最後一個音素終點(不含後面的停頓)。
命中率與論文相同:每個音節先對生成種子平均,再看 |合成 − 真人| ≤ 30 ms;多個訓練種子的評估資料夾合併計算,
信賴區間以句子為單位重抽(各訓練種子的同一句一起抽)。

    python s1_syllable_diag.py --dialect sixian --eval-dirs <dir1>,<dir2>,<dir3> --out s1_sixian.md
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, fmt, mean_ci, pearson  # noqa: E402
from hakka_c.structure import StructError, build_utt  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402

TOL = 30.0


def spans(utt):
    """{音節 idx: (音節起, 音節終, 母音起, 母音終, 聲母長, 韻尾長, 音節)},時間單位秒。
    聲母 = 母音核之前的音素、韻尾 = 母音核之後的音素(同一個音節內)。"""
    iv = dict(zip(utt.phone_sym, utt.phone_iv))
    out = {}
    for s in utt.syllables:
        a, b = iv[s.sym_idx[0]][0], iv[s.sym_idx[-1]][1]
        v0, v1 = iv[s.nuc]
        k = s.sym_idx.index(s.nuc)
        on = sum(iv[i][1] - iv[i][0] for i in s.sym_idx[:k])
        co = sum(iv[i][1] - iv[i][0] for i in s.sym_idx[k + 1:])
        out[s.idx] = (a, b, v0, v1, on, co, s)
    return out


def collect(ev, real_tg, why):
    """一個評估資料夾 → 逐音節(已對生成種子平均)的資料。"""
    rows = defaultdict(list)
    real_cache = {}
    for tgp in sorted((ev / "mfa_tg").rglob("*.TextGrid")):
        stem, seed = tgp.stem.rsplit("_g", 1)
        bj = ev / "bounds" / f"{stem}.json"
        if not bj.exists() or stem not in real_tg:
            why["缺 bounds 或真人 TextGrid"] += 1
            continue
        names = json.load(open(bj, encoding="utf-8"))["names"]
        try:
            if stem not in real_cache:
                real_cache[stem] = spans(build_utt(stem, names, read_textgrid(real_tg[stem])))
            syn = spans(build_utt(stem, names, read_textgrid(tgp)))
        except (StructError, KeyError) as e:
            why[str(e)[:40]] += 1
            continue
        real = real_cache[stem]
        for idx, (a, b, v0, v1, on, co, s) in syn.items():
            if idx not in real:
                continue
            ra, rb, rv0, rv1, ron, rco, _ = real[idx]
            S_syn, V_syn = (b - a) * 1000, (v1 - v0) * 1000
            S_ref, V_ref = (rb - ra) * 1000, (rv1 - rv0) * 1000
            if min(S_syn, V_syn, S_ref, V_ref) <= 0:
                why["長度為 0"] += 1
                continue
            rows[(stem, idx)].append({
                "S_syn": S_syn, "V_syn": V_syn, "S_ref": S_ref, "V_ref": V_ref,
                "V_a": V_syn * S_ref / S_syn, "V_b": S_syn * V_ref / S_ref,
                "O_syn": on * 1000, "C_syn": co * 1000, "O_ref": ron * 1000, "C_ref": rco * 1000,
                "entering": s.entering, "coda": s.coda, "prepausal": s.prepausal})
    out = []
    for (stem, idx), rs in rows.items():
        m = {k: float(np.mean([r[k] for r in rs]))
             for k in ("S_syn", "V_syn", "S_ref", "V_ref", "V_a", "V_b", "O_syn", "C_syn", "O_ref", "C_ref")}
        out.append({"stem": stem, "idx": idx, **m, "entering": rs[0]["entering"], "coda": rs[0]["coda"],
                    "prepausal": rs[0]["prepausal"], "n_gen": len(rs)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--eval-dirs", required=True, help="逗號分隔(通常是同一系統的三個訓練種子)")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    real_tg = {p.stem: p for p in C.textgrid_dir(args.dialect).rglob("*.TextGrid")}
    why, data = Counter(), []
    dirs = [Path(d).expanduser() for d in args.eval_dirs.split(",") if d]
    for k, ev in enumerate(dirs):
        if not (ev / "mfa_tg").exists():
            sys.exit(f"★ {ev}/mfa_tg 不存在:先跑 d1_pipeline.sh")
        rs = collect(ev, real_tg, why)
        for r in rs:
            r["sys"] = k
        data += rs
    if not data:
        sys.exit(f"★ 沒有可用的音節:{dict(why)}")

    g = lambda k: np.array([r[k] for r in data], float)  # noqa: E731
    S_syn, V_syn, S_ref, V_ref, V_a, V_b = (g(k) for k in ("S_syn", "V_syn", "S_ref", "V_ref", "V_a", "V_b"))
    O_syn, C_syn, O_ref, C_ref = (g(k) for k in ("O_syn", "C_syn", "O_ref", "C_ref"))
    et = np.array([r["entering"] for r in data], bool)
    stems = np.array([r["stem"] for r in data])
    hit = lambda v: (np.abs(v - V_ref) <= TOL).astype(float)  # noqa: E731
    h0, ha, hb = hit(V_syn), hit(V_a), hit(V_b)
    cols = {"h0": h0, "ha": ha, "hb": hb, "da": ha - h0, "db": hb - h0, "dba": hb - ha,
            "osh_s": O_syn / S_syn, "osh_r": O_ref / S_ref, "csh_s": C_syn / S_syn, "csh_r": C_ref / S_ref,
            "evsh": np.abs(V_syn / S_syn - V_ref / S_ref),
            "eS": np.abs(S_syn - S_ref), "eV": np.abs(V_syn - V_ref), "bS": S_syn - S_ref, "bV": V_syn - V_ref,
            "shs": V_syn / S_syn, "shr": V_ref / S_ref}
    t = UttTable(stems, cols, {"et": et, "non": ~et})
    idx = boot_idx(t.U, args.n_boot)
    pct = lambda c: tuple(100 * x for x in c)  # noqa: E731

    L = [f"# S1 音節總長 vs 音節內分配:{args.dialect}\n",
         f"評估資料夾:{', '.join(d.name for d in dirs)}",
         f"音節數:入聲 {int(et.sum())}、舒聲 {int((~et).sum())}(各訓練種子合計,每個已對生成種子平均);"
         f"句數 {t.U};略過:{dict(why) or '無'}\n",
         "## 1. 目前的誤差\n",
         "| | 入聲 | 舒聲 |", "|---|---|---|",
         f"| 音節總長平均絕對誤差 (ms) | {t.mean('eS', 'et'):.1f} | {t.mean('eS', 'non'):.1f} |",
         f"| 音節總長平均偏差 (ms) | {fmt(mean_ci(t, 'bS', 'et', idx))} | {fmt(mean_ci(t, 'bS', 'non', idx))} |",
         f"| 母音平均絕對誤差 (ms) | {t.mean('eV', 'et'):.1f} | {t.mean('eV', 'non'):.1f} |",
         f"| 母音平均偏差 (ms) | {fmt(mean_ci(t, 'bV', 'et', idx))} | {fmt(mean_ci(t, 'bV', 'non', idx))} |",
         f"| 母音占音節比例:合成 / 真人 | {t.mean('shs', 'et'):.3f} / {t.mean('shr', 'et'):.3f} | "
         f"{t.mean('shs', 'non'):.3f} / {t.mean('shr', 'non'):.3f} |",
         f"| 聲母占音節比例:合成 / 真人 | {t.mean('osh_s', 'et'):.3f} / {t.mean('osh_r', 'et'):.3f} | "
         f"{t.mean('osh_s', 'non'):.3f} / {t.mean('osh_r', 'non'):.3f} |",
         f"| 韻尾占音節比例:合成 / 真人 | {t.mean('csh_s', 'et'):.3f} / {t.mean('csh_r', 'et'):.3f} | "
         f"{t.mean('csh_s', 'non'):.3f} / {t.mean('csh_r', 'non'):.3f} |",
         f"| 母音比例平均絕對誤差(合成 vs 真人) | {t.mean('evsh', 'et'):.3f} | {t.mean('evsh', 'non'):.3f} |",
         f"| 音節誤差與母音誤差的相關 r | {pearson((S_syn - S_ref)[et], (V_syn - V_ref)[et]):.3f} | "
         f"{pearson((S_syn - S_ref)[~et], (V_syn - V_ref)[~et]):.3f} |", "",
         "## 2. 反事實調整後的母音 ±30 ms 命中率(百分比;差值括號為 95% 信賴區間)\n",
         "| | 入聲 | 舒聲 |", "|---|---|---|",
         f"| 目前 | {100 * t.mean('h0', 'et'):.1f} | {100 * t.mean('h0', 'non'):.1f} |",
         f"| (a) 音節總長換成真人的 | {100 * t.mean('ha', 'et'):.1f} | {100 * t.mean('ha', 'non'):.1f} |",
         f"| (b) 母音比例換成真人的 | {100 * t.mean('hb', 'et'):.1f} | {100 * t.mean('hb', 'non'):.1f} |",
         f"| (a) − 目前 | {fmt(pct(mean_ci(t, 'da', 'et', idx)))} | {fmt(pct(mean_ci(t, 'da', 'non', idx)))} |",
         f"| (b) − 目前 | {fmt(pct(mean_ci(t, 'db', 'et', idx)))} | {fmt(pct(mean_ci(t, 'db', 'non', idx)))} |",
         f"| (b) − (a)(配對) | {fmt(pct(mean_ci(t, 'dba', 'et', idx)))} | {fmt(pct(mean_ci(t, 'dba', 'non', idx)))} |", ""]

    # 原本沒命中的入聲母音,被哪一種調整救回
    miss = et & (h0 == 0)
    n_m = int(miss.sum())
    fa, fb = (ha[miss] == 1), (hb[miss] == 1)
    L += ["## 3. 原本沒命中的入聲母音,被哪一種調整救回\n",
          f"原本沒命中:{n_m} 個(占入聲 {100 * n_m / max(et.sum(), 1):.1f}%)\n",
          "| 只被 (a) 救回 | 只被 (b) 救回 | 兩者都救回 | 都救不回 |", "|---|---|---|---|",
          f"| {100 * np.mean(fa & ~fb):.1f}% | {100 * np.mean(~fa & fb):.1f}% | {100 * np.mean(fa & fb):.1f}% | "
          f"{100 * np.mean(~fa & ~fb):.1f}% |", ""]

    # 對數誤差分解
    eS = np.log(S_syn / S_ref)
    eR = np.log((V_syn / S_syn) / (V_ref / S_ref))
    L += ["## 4. 母音對數誤差的變異分解(入聲)\n",
          "log(V_syn/V_ref) = log(S_syn/S_ref) + log(母音比例_syn/母音比例_ref)\n",
          "| 音節總長部分的變異 | 母音比例部分的變異 | 2 × 共變異 | 合計 |", "|---|---|---|---|"]
    vS, vR, cv = np.var(eS[et]), np.var(eR[et]), 2 * np.cov(eS[et], eR[et])[0, 1]
    tot = vS + vR + cv
    L.append(f"| {vS:.4f}({100 * vS / tot:.0f}%) | {vR:.4f}({100 * vR / tot:.0f}%) | {cv:+.4f} | {tot:.4f} |\n")

    # 依韻尾與停頓前後分組
    L += ["## 5. 入聲依韻尾與位置分組(目前命中率 → (a) / (b))\n", "| 組別 | 音節數 | 目前 | (a) | (b) |", "|---|---|---|---|---|"]
    coda = np.array([r["coda"] for r in data])
    pp = np.array([r["prepausal"] for r in data], bool)
    groups = [(f"韻尾 {c}", et & (coda == c)) for c in sorted(set(coda[et]))] + [("停頓前", et & pp), ("非停頓前", et & ~pp)]
    for name, m in groups:
        if m.sum() >= 20:
            L.append(f"| {name} | {int(m.sum())} | {100 * h0[m].mean():.1f} | {100 * ha[m].mean():.1f} | {100 * hb[m].mean():.1f} |")

    da, db = mean_ci(t, "da", "et", idx), mean_ci(t, "db", "et", idx)
    L.append("\n## 判讀\n")
    if da[1] > 0 and da[0] >= 0.03:
        L.append(f"- 入聲:把音節總長換成真人的,命中率 {100 * da[0]:+.1f} 點,區間不含 0 → 音節總長解釋了一部分錯誤,"
                 "雙層時長預測器有改善空間(實際模型只能拿到這個上限的一部分)。")
    else:
        L.append(f"- 入聲:把音節總長換成真人的,命中率只 {100 * da[0]:+.1f} 點 → 音節總長不是主要瓶頸,雙層預測器的改善空間有限。")
    dba = mean_ci(t, "dba", "et", idx)
    if dba[1] > 0:
        verdict = "(b) 顯著大於 (a) → 入聲母音的錯誤較偏向「母音占音節的比例」(聲母、母音、韻尾之間的分配)。"
    elif dba[2] < 0:
        verdict = "(b) 顯著小於 (a) → 入聲母音的錯誤較偏向「整個音節的長度」。"
    else:
        verdict = "(b) 與 (a) 的差異不顯著 → 兩種來源的影響相近。"
    L.append(f"- 對照:把母音比例換成真人的,命中率 {100 * db[0]:+.1f} 點;(b) − (a) = {fmt(pct(dba))}。" + verdict)
    L.append("\n注意:(a)、(b) 都用到真人時長,只是上限診斷;MFA 是用真人錄音訓練的,量合成語音時可能較不準。")
    text = "\n".join(L)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
