"""H3 探針 A:時長預測器「本身」的準確度,以及多個訓練種子取平均(集成)能多多少。只用 CPU,幾秒到幾十秒。

為什麼要看「預測器本身」:
  波形評估 = 時長預測器 → 解碼器合成 → MFA 對合成音重新對齊 → 量母音核。最後一段 MFA 在合成音上也有 ±10–20 ms 的邊界誤差,
  會拉低命中率與 ρ。這裡直接用預測器輸出的連續時長 w(c_eval.py 存在每個評估資料夾的 recs.pkl 裡),
  取整後只看母音核本身(不含 blank,避免靜音被分給母音),跟真人錄音的 MFA 母音核時長比。
  好處:沒有合成端的量測誤差、用全部測試句(不是 100 句)、不用 GPU、改任何時長方法幾秒就有結果。

為什麼要看集成:
  1h 的時長預測器嚴重過擬合(訓練 loss 比驗證 loss 小 60–80 倍),過擬合模型的預測雜訊大,不同種子的雜訊彼此獨立,
  取平均(log 域)可以直接降低雜訊。這裡把同一系統的多個種子的 w 取幾何平均再取整,看 ρ、誤差、命中率多多少。

    python h3_probe.py --dialect sixian --sys c6=$E/c6_sixian_step40000_cum,$E/c6_sixian_s123_step40000_cum,... --wave
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import eval_rows, geo_mean_w  # noqa: E402
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402

COLS = (("et_rho", "入聲 ρ", "rho"), ("et_vde", "入聲誤差(ms)", "ms"), ("et_h20", "入聲 ±20 ms(%)", "pct"),
        ("et_h30", "入聲 ±30 ms(%)", "pct"), ("et_hp", "入聲 ±20%(%)", "pct"), ("non_rho", "舒聲 ρ", "rho"),
        ("non_h30", "舒聲 ±30 ms(%)", "pct"))


def metrics(et, ref, pred):
    out = {}
    for tag, m in (("et", et), ("non", ~et)):
        e = pred[m] - ref[m]
        a = np.abs(e)
        out[f"{tag}_rho"] = pearson(pred[m], ref[m])
        out[f"{tag}_vde"] = float(a.mean())
        out[f"{tag}_bias"] = float(e.mean())
        out[f"{tag}_h20"] = float((a <= 20).mean())
        out[f"{tag}_h30"] = float((a <= 30).mean())
        out[f"{tag}_hp"] = float((a <= 0.2 * ref[m]).mean())
    return out


def mean_metrics(et, ref, P):
    """P: (K, N) 多組預測 → 各組分別算再平均。"""
    ms = [metrics(et, ref, p) for p in P]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def load_recs(d):
    z = pickle.load(open(Path(d).expanduser() / "recs.pkl", "rb"))
    return z["recs"], z.get("params", {}).get("rounding", "ceil")


def to_arrays(rows):
    return (np.array([r["stem"] for r in rows]), np.array([int(r["syl_idx"]) for r in rows]),
            np.array([bool(r["entering"]) for r in rows]), np.array([r["ref_ms"] for r in rows], float),
            np.array([r["pred_ms"] for r in rows], float))


def system_tables(dirs):
    """回傳 (stem, idx, et, ref, P(K,N), ens(N) 或 None, 取整)。ens 用多數種子的取整。"""
    loaded = [load_recs(d) for d in dirs]
    by = [{r["stem"]: r for r in recs} for recs, _ in loaded]
    roundings = [rd for _, rd in loaded]
    stems = sorted(set.intersection(*[set(b) for b in by]))
    stems = [s for s in stems if len({len(b[s]["w"]) for b in by}) == 1 and len({len(b[s]["syllables"]) for b in by}) == 1]
    if not stems:
        raise SystemExit(f"★ {dirs} 沒有共同的測試句")
    P, base = [], None
    for b, rd in zip(by, roundings):
        stem, idx, et, ref, pred = to_arrays(eval_rows([b[s] for s in stems], rule="none", rounding=rd))
        if base is None:
            base = (stem, idx, et, ref)
        P.append(pred)
    P = np.array(P)
    ens = None
    if len(dirs) > 1:
        cnt = Counter(roundings).most_common()
        rd = cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else "cum"
        recs = [dict(by[0][s], w=geo_mean_w([b[s]["w"] for b in by])) for s in stems]
        ens = to_arrays(eval_rows(recs, rule="none", rounding=rd))[4]
    return (*base, P, ens, roundings)


def boot_ci(stem, et, ref, A, B, keys, n_boot=1000, seed=42):
    """B(N)− mean_k A(K,N) 的差,句子重抽。回傳 {key: (差, 下界, 上界)}。"""
    uniq, inv = np.unique(stem, return_inverse=True)
    groups = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = {k: mean_val(et, ref, B[None], k) - mean_val(et, ref, A, k) for k in keys}
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for k in keys:
            draws[k].append(mean_val(et[idx], ref[idx], B[None][:, idx], k) - mean_val(et[idx], ref[idx], A[:, idx], k))
    return {k: (point[k], *np.percentile(draws[k], [2.5, 97.5])) for k in keys}


def mean_val(et, ref, P, key):
    return float(np.mean([metrics(et, ref, p)[key] for p in P]))


def fmt_val(v, kind):
    return f"{v:.3f}" if kind == "rho" else (f"{v:.1f}" if kind == "ms" else f"{100 * v:.1f}")


def fmt_row(label, m):
    return f"| {label} | " + " | ".join(fmt_val(m[k], kind) for k, _, kind in COLS) + " |"


def wave_rows(dirs, stem, idx, P):
    """與波形評估同一批音節:預測器本身 vs 波形量測。回傳 (預測器本身 metrics, 波形 metrics),各種子平均。"""
    a_list, b_list = [], []
    for k, d in enumerate(dirs):
        f = Path(d).expanduser() / "syllables_wave.csv"
        if not f.exists():
            continue
        c = read_cols(f)
        wave = {(s, int(i)): (bool(e), r, p) for s, i, e, r, p in zip(c["stem"], c["syl_idx"], boolcol(c["entering"]), c["ref_ms"], c["pred_ms"])}
        sel = np.array([(s, int(i)) in wave for s, i in zip(stem, idx)])
        if sel.sum() < 50:
            continue
        et = np.array([wave[(s, int(i))][0] for s, i in zip(stem[sel], idx[sel])])
        ref = np.array([wave[(s, int(i))][1] for s, i in zip(stem[sel], idx[sel])])
        pw = np.array([wave[(s, int(i))][2] for s, i in zip(stem[sel], idx[sel])])
        a_list.append(metrics(et, ref, P[k][sel]))
        b_list.append(metrics(et, ref, pw))
    if not a_list:
        return None, None
    avg = lambda ms: {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}  # noqa: E731
    return avg(a_list), avg(b_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--sys", action="append", required=True, metavar="NAME=DIR1,DIR2,...",
                    help="評估資料夾(含 recs.pkl);多個種子用逗號分隔")
    ap.add_argument("--wave", action="store_true", help="另外列波形量測(同一批音節)作對照")
    ap.add_argument("--vs", help="系統名稱:其他系統對它的差(預測器本身,句子重抽 CI);有多個種子的系統用集成")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out")
    args = ap.parse_args()

    head = "| 系統 | " + " | ".join(lab for _, lab, _ in COLS) + " |"
    sep = "|---|" + "---|" * len(COLS)
    L = [f"# H3 預測器本身的時長準確度與種子集成:{args.dialect}\n",
         "母音核時長 = 預測器輸出取整後的母音核(不含 blank)vs 真人錄音 MFA 母音核;全部測試句。"
         "單一種子 = 各種子分別算再平均;集成 = 各種子的連續時長取幾何平均後再取整。\n", head, sep]
    gain_lines = ["", "## 集成相對單一種子的增益(句子重抽 95% CI)\n",
                  "| 系統 | 入聲 ρ | 入聲誤差(ms) | 入聲 ±20 ms(百分點) | 入聲 ±30 ms(百分點) | 入聲 ±20%(百分點) |", "|---|---|---|---|---|---|"]
    wave_lines = ["", "## 預測器本身 vs 波形量測(波形評估的同一批音節,各種子平均)\n", head, sep]
    store = {}
    for spec in args.sys:
        name, dirs = spec.split("=", 1)
        dirs = [x for x in dirs.split(",") if x]
        stem, idx, et, ref, P, ens, roundings = system_tables(dirs)
        store[name] = (stem, idx, et, ref, P, ens)
        L.append(fmt_row(f"{name}・單一種子平均({len(dirs)} 個,{'/'.join(roundings)})", mean_metrics(et, ref, P)))
        if ens is not None:
            L.append(fmt_row(f"**{name}・{len(dirs)} 種子集成**", metrics(et, ref, ens)))
            keys = ["et_rho", "et_vde", "et_h20", "et_h30", "et_hp"]
            ci = boot_ci(stem, et, ref, P, ens, keys, args.n_boot)

            def cell(k, kind):
                d, lo, hi = ci[k]
                sc = 100 if kind == "pct" else 1
                f = ".3f" if kind == "rho" else ".1f"
                return f"{d * sc:+{f}} [{lo * sc:+{f}}, {hi * sc:+{f}}]"
            gain_lines.append(f"| {name} | " + " | ".join(cell(k, kind) for k, kind in (
                ("et_rho", "rho"), ("et_vde", "ms"), ("et_h20", "pct"), ("et_h30", "pct"), ("et_hp", "pct"))) + " |")
        if args.wave:
            a, b = wave_rows(dirs, stem, idx, P)
            if a is not None:
                wave_lines.append(fmt_row(f"{name}・預測器本身", a))
                wave_lines.append(fmt_row(f"{name}・波形量測", b))
    vs_lines = []
    if args.vs:
        if args.vs not in store:
            raise SystemExit(f"★ --vs {args.vs} 不在 --sys 裡")
        sa, ia, ea, ra, Pa, _ = store[args.vs]
        vs_lines = ["", f"## 其他系統相對 {args.vs} 的差(預測器本身,連續母音核取整後;句子重抽 95% CI;多種子的系統用集成)\n",
                    "| 系統 | 入聲 ρ | 入聲誤差(ms) | 入聲 ±20 ms(百分點) | 入聲 ±30 ms(百分點) | 入聲 ±20%(百分點) |", "|---|---|---|---|---|---|"]
        pos_a = {(s_, int(i_)): n for n, (s_, i_) in enumerate(zip(sa, ia))}
        for name, (sb, ib, eb, rb, Pb, ensb) in store.items():
            if name == args.vs:
                continue
            pos_b = {(s_, int(i_)): n for n, (s_, i_) in enumerate(zip(sb, ib))}
            keys_ = sorted(set(pos_a) & set(pos_b))
            ia_ = np.array([pos_a[k] for k in keys_])
            ib_ = np.array([pos_b[k] for k in keys_])
            B = ensb if ensb is not None else Pb[0]
            ci = boot_ci(sa[ia_], ea[ia_], ra[ia_], Pa[:, ia_], B[ib_], ["et_rho", "et_vde", "et_h20", "et_h30", "et_hp"], args.n_boot)
            cells = []
            for k, kind in (("et_rho", "rho"), ("et_vde", "ms"), ("et_h20", "pct"), ("et_h30", "pct"), ("et_hp", "pct")):
                d_, lo_, hi_ = ci[k]
                sc = 100 if kind == "pct" else 1
                f_ = ".3f" if kind == "rho" else ".1f"
                cells.append(f"{d_ * sc:+{f_}} [{lo_ * sc:+{f_}}, {hi_ * sc:+{f_}}]")
            vs_lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(L + vs_lines + (gain_lines if len(gain_lines) > 4 else []) + (wave_lines if args.wave and len(wave_lines) > 4 else []))
    print(text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "h3" / f"probe_{args.dialect}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
