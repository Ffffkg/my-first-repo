"""H3 探針 C:另一個腔調的資料,對「文字 → 母音核時長」有沒有幫助?「神經時長預測器 + 文字特徵模型」混合能不能更準?只用 CPU。

(1) 文字特徵梯度提升樹(特徵同 h1_ceiling.py,另加「腔別」,聲調標記加腔別前綴)。訓練集:
      目標腔 1h | 來源腔 48h(完全沒看過目標腔,零樣本)| 來源 48h + 目標 1h(目標加權 1 與 10)| 目標腔 10h(參考)
    在目標腔「測試句」的母音核上算 ρ、誤差、命中率。
    重點是「來源 48h + 目標 1h」是否明顯高於「目標 1h」:是 → 跨腔資料裡有可用的時長資訊,值得做神經的聯合訓練(C11)。
(2) 混合:log 域 α·GBM + (1−α)·神經預測(C6 的連續時長,可用多種子幾何平均)。
    GBM 版本與 α 都只在驗證集上選,測試集不參與。

    python h3_xdialect.py --dialect sixian --source hailu --valid-list val_clean_from_hailu \\
        --c6val $C3/c6val_c6_sixian/recs.pkl,... --c6test $E/c6_sixian_step40000_cum,...
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1_ceiling as H  # noqa: E402
import h3_probe as P  # noqa: E402
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import geo_mean_w  # noqa: E402
from hakka_c.durations import merge_blank  # noqa: E402

CAT2 = H.CAT + ["dia"]


class Enc:
    def __init__(self, rows):
        self.maps = {k: {v: i for i, v in enumerate(sorted({str(r[k]) for r in rows}))} for k in CAT2}

    def __call__(self, rows):
        X = np.array([[self.maps[k].get(str(r[k]), -1) for k in CAT2] + [float(r[k]) for k in H.NUM] for r in rows], float)
        X[X == -1] = np.nan
        return X


def tag(rows, dia):
    for r in rows:
        r["dia"] = dia
        for k in ("tone", "p_tone", "n_tone"):
            r[k] = f"{dia}:{r[k]}"
    return rows


def fit_gbm(rows, weights=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    enc = Enc(rows)
    mask = [len(enc.maps[k]) <= 250 for k in CAT2] + [False] * len(H.NUM)
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=1.0, categorical_features=mask, early_stopping=True,
                                      validation_fraction=0.1, random_state=seed)
    m.fit(enc(rows), np.log([r["ref_ms"] for r in rows]), sample_weight=weights)
    return lambda rs: np.exp(m.predict(enc(rs)))


def nn_from_recs(recs_list):
    """多個種子的 recs(各為一個列表)→ {(stem, idx): [各種子的連續母音核 ms]};只留共同的鍵。"""
    per = []
    for recs in recs_list:
        d = {}
        for rec in recs:
            merged = merge_blank(np.asarray(rec["w"], float), "halfcap") * C.HOP_MS      # 與 MFA 母音核同一個定義
            for s in rec["syllables"]:
                d[(rec["stem"], int(s["idx"]))] = float(merged[s["nuc"]])
        per.append(d)
    keys = set.intersection(*[set(d) for d in per])
    return {k: [d[k] for d in per] for k in keys}


def mae(et, ref, pred):
    a = np.abs(pred - ref)
    return 0.5 * (a[et].mean() + a[~et].mean())


def fit_alpha(et, ref, gbm, nn_list):
    """nn_list:(M, N) 的神經預測(M 個種子,或集成 M=1)。回傳 (α, 驗證 MAE)。"""
    best = None
    for a in np.round(np.linspace(0, 1, 21), 2):
        m = np.mean([mae(et, ref, np.exp(a * np.log(gbm) + (1 - a) * np.log(nn))) for nn in nn_list])
        key = (round(float(m), 6), a)
        if best is None or key < best[0]:
            best = (key, a, m)
    return float(best[1]), float(best[2])


def blend(a, gbm, nn):
    return np.exp(a * np.log(gbm) + (1 - a) * np.log(nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--source", required=True, choices=C.DIALECTS)
    ap.add_argument("--valid-list", required=True, help="目標腔驗證集的 filelist 名稱(不含 .txt,例如 val_clean_from_hailu)")
    ap.add_argument("--small", default="ladder_1h")
    ap.add_argument("--mid", default="ladder_10h")
    ap.add_argument("--source-list", default="ladder_48h")
    ap.add_argument("--source-max", type=int, default=8000)
    ap.add_argument("--target-weight", type=float, default=10.0)
    ap.add_argument("--c6val", help="逗號分隔:c3_target_check.py 在驗證集輸出的 recs.pkl(各種子)")
    ap.add_argument("--c6test", help="逗號分隔:C6 各種子的評估資料夾(含 recs.pkl)")
    ap.add_argument("--out")
    args = ap.parse_args()

    print("抽文字特徵 …")
    test = tag(H.extract(args.dialect, "test", 0), "tgt")
    valid = tag(H.extract(args.dialect, args.valid_list, 0), "tgt")
    tgt1 = tag(H.extract(args.dialect, args.small, 0), "tgt")
    tgt10 = tag(H.extract(args.dialect, args.mid, 0), "tgt")
    src = tag(H.extract(args.source, args.source_list, args.source_max), "src")
    held = {r["stem"] for r in test} | {r["stem"] for r in valid}
    tgt1 = [r for r in tgt1 if r["stem"] not in held]
    tgt10 = [r for r in tgt10 if r["stem"] not in held]

    nn_test = nn_val = None
    if args.c6test and args.c6val:
        nn_test = nn_from_recs([pickle.load(open(Path(d).expanduser() / "recs.pkl", "rb"))["recs"] for d in args.c6test.split(",")])
        nn_val = nn_from_recs([pickle.load(open(Path(p).expanduser(), "rb"))["recs"] for p in args.c6val.split(",")])
        test = [r for r in test if (r["stem"], int(r["syl_idx"])) in nn_test]
        valid = [r for r in valid if (r["stem"], int(r["syl_idx"])) in nn_val]
        print(f"  神經預測對得上的音節:測試 {len(test)}、驗證 {len(valid)}")

    def arrays(rows):
        return (np.array([bool(r["entering"]) for r in rows]), np.array([r["ref_ms"] for r in rows], float))
    et_t, ref_t = arrays(test)
    et_v, ref_v = arrays(valid)

    variants = [("文字特徵 GBM・目標腔 " + args.small, tgt1, None),
                (f"文字特徵 GBM・來源腔 {args.source_list}(零樣本跨腔)", src, None),
                (f"文字特徵 GBM・來源 + 目標 {args.small}(目標權重 1)", src + tgt1, [1.0] * len(src) + [1.0] * len(tgt1)),
                (f"文字特徵 GBM・來源 + 目標 {args.small}(目標權重 {args.target_weight:g})", src + tgt1,
                 [1.0] * len(src) + [args.target_weight] * len(tgt1)),
                ("文字特徵 GBM・目標腔 " + args.mid + "(參考)", tgt10, None)]
    L = [f"# H3 跨腔資料與混合:目標腔 {args.dialect}(來源腔 {args.source})\n",
         f"測試句 {len({r['stem'] for r in test})} 句、入聲母音核 {int(et_t.sum())} 個;所有預測皆為連續時長(未取整)。\n",
         "| 系統 | " + " | ".join(lab for _, lab, _ in P.COLS) + " |", "|---|" + "---|" * len(P.COLS)]
    preds_t, preds_v = {}, {}
    for name, rows, wts in variants:
        print(f"訓練 {name}({len(rows)} 個音節)…")
        f = fit_gbm(rows, wts)
        preds_t[name], preds_v[name] = f(test), f(valid)
        L.append(P.fmt_row(name, P.metrics(et_t, ref_t, preds_t[name])))

    if nn_test is not None:
        K = len(next(iter(nn_test.values())))
        nn_k_t = np.array([[nn_test[(r["stem"], int(r["syl_idx"]))][k] for r in test] for k in range(K)])
        nn_k_v = np.array([[nn_val[(r["stem"], int(r["syl_idx"]))][k] for r in valid] for k in range(K)])
        nn_e_t = np.exp(np.log(nn_k_t).mean(axis=0))
        nn_e_v = np.exp(np.log(nn_k_v).mean(axis=0))
        L.append(P.fmt_row(f"神經 C6・單一種子平均({K} 個)", P.mean_metrics(et_t, ref_t, nn_k_t)))
        if K > 1:
            L.append(P.fmt_row(f"神經 C6・{K} 種子集成", P.metrics(et_t, ref_t, nn_e_t)))
        # 驗證集選 GBM 版本(只考慮有用到目標腔資料的版本)
        cand = [v[0] for v in variants if "零樣本" not in v[0] and "參考" not in v[0]]
        best = min(cand, key=lambda n: mae(et_v, ref_v, preds_v[n]))
        L.append("")
        L.append(f"混合用的 GBM(驗證集選):**{best}**\n")
        L += ["| 混合 | α(GBM 的比例,驗證集選) | " + " | ".join(lab for _, lab, _ in P.COLS) + " |",
              "|---|---|" + "---|" * len(P.COLS)]
        a_s, _ = fit_alpha(et_v, ref_v, preds_v[best], nn_k_v)
        ms = [P.metrics(et_t, ref_t, blend(a_s, preds_t[best], nn)) for nn in nn_k_t]
        L.append(f"| 神經單一種子 + GBM | {a_s:.2f} | " + " | ".join(P.fmt_val(float(np.mean([m[k] for m in ms])), kind) for k, _, kind in P.COLS) + " |")
        if K > 1:
            a_e, _ = fit_alpha(et_v, ref_v, preds_v[best], nn_e_v[None])
            L.append(f"| **神經 {K} 種子集成 + GBM** | {a_e:.2f} | " + " | ".join(
                P.fmt_val(P.metrics(et_t, ref_t, blend(a_e, preds_t[best], nn_e_t))[k], kind) for k, _, kind in P.COLS) + " |")
    L += ["", "讀法:",
          "* 「來源 + 目標 1h」若明顯高於「目標 1h」→ 跨腔資料有可用的時長資訊,神經模型也該同時看到它(C11 聯合訓練)。",
          "* 混合若高於神經單獨 → 兩種預測器的錯誤不完全相同,可以互補;α 接近 0 → GBM 沒有補充資訊。"]
    text = "\n".join(L)
    print("\n" + text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "h3" / f"xdialect_{args.dialect}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
