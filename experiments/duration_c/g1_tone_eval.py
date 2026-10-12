"""G1 第 2 步:客觀聲調辨識率(手冊 tone_error_rate.py 的實作版)。不需要 GPU。

做法:用「真人錄音」訓練一個聲調分類器(只看聲學:本音節母音核 5 點 F0、斜率、時長、濁音比例,
加上前後音節的 F0;不給任何文字標記),再拿它去辨識合成語音每個音節的聲調。
  * 真人測試集的辨識率 = 天花板(分類器本身的極限,含變調造成的「錯」)
  * 合成語音的辨識率越接近天花板,代表聲調唸得越像真人;聲調錯誤率 = 1 − 辨識率
  * 另報「入聲辨識率」:真的入聲被判成入聲調的比例(手冊 E3 入聲辨識的客觀版)

    G=~/hakka_tts/exp_c/g1; E=~/hakka_tts/exp_c/eval
    python g1_tone_eval.py --dialect sixian --train $G/real_sixian_train_full.csv --ceiling $G/real_sixian_test.csv \\
        --sys base=$E/base_sixian_step40000/syl_f0.csv --sys c2=$E/c2_sixian_step40000/syl_f0.csv --compare base c2

CI 以句子為單位重抽;同一句的多個生成種子一起抽。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import write_rows  # noqa: E402
from hakka_c.f0feat import speaker_medians, to_semitone  # noqa: E402

ENTERING_TONES = {"2", "5"}
FEATS = ["st1", "st2", "st3", "st4", "st5", "mean", "slope", "logdur", "vfrac",
         "prev_st5", "prev_mean", "next_st1", "next_mean"]


def tone_str(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def load(path):
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"★ {path} 是空的")
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    t = {"stem": np.array([r["stem"] for r in rows]), "spk": np.array([int(float(r["spk"])) for r in rows]),
         "gen_seed": np.array([int(float(r["gen_seed"])) for r in rows]),
         "syl_idx": np.array([int(float(r["syl_idx"])) for r in rows]),
         "tone": np.array([tone_str(r["tone"]) for r in rows]),
         "entering": np.array([float(r["entering"]) > 0.5 for r in rows]),
         "dur_ms": np.array([num(r["dur_ms"]) for r in rows]), "vfrac": np.array([num(r["vfrac"]) for r in rows]),
         "pts": np.array([[num(r[f"p{k}"]) for k in range(1, 6)] for r in rows])}
    return t


def featurize(t):
    st = to_semitone(t["spk"], t["pts"], speaker_medians(t["spk"], t["pts"]))
    with np.errstate(all="ignore"):
        mean = np.nanmean(st, axis=1)
    first = np.array([r[np.isfinite(r)][0] if np.isfinite(r).any() else np.nan for r in st])
    last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in st])
    F = {f"st{k + 1}": st[:, k] for k in range(5)}
    F.update(mean=mean, slope=last - first, logdur=np.log(np.maximum(t["dur_ms"], 1.0)), vfrac=t["vfrac"])
    # 前後音節(同一句、同一生成種子、索引相鄰)
    key = {(s, g, i): n for n, (s, g, i) in enumerate(zip(t["stem"], t["gen_seed"], t["syl_idx"]))}
    prev = np.array([key.get((s, g, i - 1), -1) for s, g, i in zip(t["stem"], t["gen_seed"], t["syl_idx"])])
    nxt = np.array([key.get((s, g, i + 1), -1) for s, g, i in zip(t["stem"], t["gen_seed"], t["syl_idx"])])
    pick = lambda a, ix: np.where(ix >= 0, a[np.maximum(ix, 0)], np.nan)  # noqa: E731
    F.update(prev_st5=pick(last, prev), prev_mean=pick(mean, prev), next_st1=pick(first, nxt), next_mean=pick(mean, nxt))
    return np.column_stack([F[k] for k in FEATS])


def per_stem(t, correct, mask, keys):
    pos = {k: n for n, k in enumerate(keys)}
    ok = np.array([s in pos for s in t["stem"]]) & mask
    idx = np.array([pos[s] for s in t["stem"][ok]])
    return (np.bincount(idx, weights=correct[ok].astype(float), minlength=len(keys)),
            np.bincount(idx, minlength=len(keys)).astype(float))


def rate(S, sel=None):
    s, n = S
    if sel is None:
        return s.sum() / max(n.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[sel].sum(axis=-1) / n[sel].sum(axis=-1)


def ci_str(p, draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--train", required=True, help="真人錄音特徵(訓練分類器)")
    ap.add_argument("--ceiling", help="真人測試集特徵(天花板)")
    ap.add_argument("--sys", action="append", default=[], metavar="NAME=CSV", help="合成系統的 syl_f0.csv")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"), help="配對比較 B − A")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", help="預設 ~/hakka_tts/exp_c/g1/<dialect>")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else C.OUT_ROOT / "g1" / args.dialect
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = load(args.train)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    clf.fit(featurize(tr), tr["tone"])
    print(f"分類器:{len(tr['tone'])} 個真人音節訓練,聲調 {sorted(set(tr['tone']))}")

    tables = {}
    if args.ceiling:
        tables["真人(天花板)"] = load(args.ceiling)
    for spec in args.sys:
        name, path = spec.split("=", 1)
        tables[name] = load(path)
    if not tables:
        sys.exit("★ 至少要給 --ceiling 或 --sys")

    res = {}
    for name, t in tables.items():
        pred = clf.predict(featurize(t))
        res[name] = (t, pred == t["tone"], np.isin(pred, list(ENTERING_TONES)))
        write_rows(out_dir / f"pred_{name.replace('(', '_').replace(')', '')}.csv",
                   [{"stem": s, "gen_seed": g, "syl_idx": i, "tone": y, "pred": p}
                    for s, g, i, y, p in zip(t["stem"], t["gen_seed"], t["syl_idx"], t["tone"], pred)])

    rng = np.random.default_rng(42)
    L = [f"# G1 客觀聲調辨識率:{args.dialect}\n",
         "分類器只用真人錄音訓練、只看聲學特徵。辨識率 % [95% CI,以句子重抽];聲調錯誤率 = 100 − 辨識率。\n",
         "| 系統 | 句數 | 全部音節 | 舒聲 | 入聲 | 入聲被判成入聲調 |", "|---|---|---|---|---|---|"]
    for name, (t, ok, pred_et) in res.items():
        keys = sorted(set(t["stem"]))
        sel = rng.integers(0, len(keys), size=(args.n_boot, len(keys)))
        cells = []
        for corr, mask in ((ok, np.ones_like(ok)), (ok, ~t["entering"]), (ok, t["entering"]), (pred_et, t["entering"])):
            S = per_stem(t, corr, mask, keys)
            cells.append(ci_str(rate(S), rate(S, sel)))
        L.append(f"| {name} | {len(keys)} | " + " | ".join(cells) + " |")

    tones = sorted(set(tr["tone"]), key=lambda x: (x in ENTERING_TONES, x))
    L += ["", "## 各聲調辨識率(%)\n", "| 系統 | " + " | ".join(tones) + " |", "|---|" + "---|" * len(tones)]
    for name, (t, ok, _) in res.items():
        L.append(f"| {name} | " + " | ".join(f"{100 * ok[t['tone'] == x].mean():.0f}" if (t["tone"] == x).any() else "—"
                                             for x in tones) + " |")

    if args.compare:
        L += ["", "## 配對比較(B − A,百分點)\n", "| 比較 | 全部音節 | 舒聲 | 入聲 | 入聲被判成入聲調 |", "|---|---|---|---|---|"]
        common = sorted(set.intersection(*[set(res[n][0]["stem"]) for pair in args.compare for n in pair]))
        sel = rng.integers(0, len(common), size=(args.n_boot, len(common)))
        pooled = {k: [] for k in range(4)}
        for a, b in args.compare:
            cells = []
            for k in range(4):
                diffs = []
                for name in (a, b):
                    t, ok, pred_et = res[name]
                    corr, mask = [(ok, np.ones_like(ok)), (ok, ~t["entering"]), (ok, t["entering"]), (pred_et, t["entering"])][k]
                    S = per_stem(t, corr, mask, common)
                    diffs.append((rate(S), rate(S, sel)))
                p, d = diffs[1][0] - diffs[0][0], diffs[1][1] - diffs[0][1]
                pooled[k].append((p, d))
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                cells.append(f"{100 * p:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            L.append(f"| {b} − {a} | " + " | ".join(cells) + " |")
        if len(args.compare) > 1:
            cells = []
            for k in range(4):
                p = np.mean([x[0] for x in pooled[k]])
                d = np.mean([x[1] for x in pooled[k]], axis=0)
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                cells.append(f"{100 * p:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            L.append(f"| **合併 {len(args.compare)} 組** | " + " | ".join(cells) + " |")

    L += ["", "讀法:天花板不是 100%(變調、語者差異);合成系統越接近天花板越好。配對差的 CI 不含 0 才算有差。"]
    text = "\n".join(L)
    print(text)
    (out_dir / "g1_summary.md").write_text(text, encoding="utf-8")
    print(f"\n→ {out_dir}/g1_summary.md、pred_*.csv")


if __name__ == "__main__":
    main()
