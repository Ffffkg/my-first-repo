"""多個訓練種子合併:方法 − 基準 的配對差,平均跨種子,CI 以句子為單位重抽(論文主表用)。不需要 GPU。

每一對 (基準, 方法) 是同一個訓練種子的兩個 eval 資料夾(d1_pipeline.sh 的輸出,含 syllables_wave.csv、mcd.csv)。
三個種子用同一批測試句,所以每次重抽句子後,各種子分別算差值再平均:CI 反映「換一批句子」的變異,
各種子的值另外列出,讓讀者看到「換一個訓練種子」的變異。

    E=~/hakka_tts/exp_c/eval
    python c_seeds.py --name 四縣 \\
        --pair $E/base_sixian_step40000 $E/c2_sixian_step40000 \\
        --pair $E/base_sixian_s123_step40000 $E/c2_sixian_s123_step40000 \\
        --pair $E/base_sixian_s2026_step40000 $E/c2_sixian_s2026_step40000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import ci, pearson  # noqa: E402

METRICS = (("et_vde", "入聲時長誤差 ET-VDE (ms)"), ("et_vbias", "入聲偏差 ET-VBias (ms)"),
           ("non_vde", "舒聲時長誤差 NonET-VDE (ms)"), ("non_vbias", "舒聲偏差 NonET-VBias (ms)"))


def load_pair(base_dir, meth_dir):
    """對齊同一批音節,回傳每音節的 (句子, 入聲?, 兩邊各自的參考, 基準預測, 方法預測)。"""
    b = read_cols(Path(base_dir).expanduser() / "syllables_wave.csv")
    m = read_cols(Path(meth_dir).expanduser() / "syllables_wave.csv")
    kb = {(s, int(i)): j for j, (s, i) in enumerate(zip(b["stem"], b["syl_idx"]))}
    pairs = [(kb[(s, int(i))], j) for j, (s, i) in enumerate(zip(m["stem"], m["syl_idx"])) if (s, int(i)) in kb]
    ib, im = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    return {"stem": np.asarray(b["stem"][ib]).astype(str), "et": boolcol(b["entering"][ib]), "ref_base": b["ref_ms"][ib].astype(float),
            "ref_meth": m["ref_ms"][im].astype(float), "base": b["pred_ms"][ib].astype(float), "meth": m["pred_ms"][im].astype(float)}


def per_stem(d, keys):
    """每句的加總與個數,依 keys 的順序排好(重抽時直接用索引)。"""
    pos = {k: n for n, k in enumerate(keys)}
    idx = np.array([pos[s] for s in d["stem"]])
    out = {}
    for sysname in ("base", "meth"):
        e = d[sysname] - d["ref_" + sysname]
        for mname, mask in (("et", d["et"]), ("non", ~d["et"])):
            out[(sysname, mname, "e")] = np.bincount(idx, weights=np.where(mask, e, 0), minlength=len(keys))
            out[(sysname, mname, "ae")] = np.bincount(idx, weights=np.where(mask, np.abs(e), 0), minlength=len(keys))
            out[("n", mname)] = np.bincount(idx, weights=mask.astype(float), minlength=len(keys))
    return out


def stat(S, sysname, metric, sel=None):
    mname = "et" if metric.startswith("et") else "non"
    kind = "ae" if metric.endswith("vde") else "e"
    s, n = S[(sysname, mname, kind)], S[("n", mname)]
    if sel is None:
        return s.sum() / n.sum()
    return s[sel].sum(axis=-1) / n[sel].sum(axis=-1)


def mcd_per_stem(base_dir, meth_dir, keys):
    """每句 MCD 差(方法 − 基準,生成種子平均);沒有 mcd.csv 或沒有共同句子時回傳 None。"""
    try:
        b = read_cols(Path(base_dir).expanduser() / "mcd.csv")
        m = read_cols(Path(meth_dir).expanduser() / "mcd.csv")
    except FileNotFoundError:
        return None

    def avg(c):
        out = {}
        for s, v in zip(np.asarray(c["stem"]).astype(str), c["mcd"].astype(float)):
            out.setdefault(s, []).append(v)
        return {s: float(np.mean(v)) for s, v in out.items()}
    mb, mm = avg(b), avg(m)
    common = sorted(set(mb) & set(mm))
    return {s: mm[s] - mb[s] for s in common} if common else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("BASE", "METHOD"),
                    help="同一訓練種子的 基準、方法 eval 資料夾;每個種子給一次")
    ap.add_argument("--name", default="", help="報告標題用(例如 四縣)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", help="Markdown 輸出(預設第一個方法資料夾旁的 seeds_<name>.md)")
    args = ap.parse_args()

    data = [load_pair(b, m) for b, m in args.pair]
    keys = sorted(set.intersection(*[set(d["stem"]) for d in data]))
    data = [{k: v[np.isin(d["stem"], keys)] for k, v in d.items()} for d in data]
    S = [per_stem(d, keys) for d in data]
    rng = np.random.default_rng(42)
    sel = rng.integers(0, len(keys), size=(args.n_boot, len(keys)))
    labels = [Path(m).expanduser().name for _, m in args.pair]

    L = [f"# 多種子合併{(':' + args.name) if args.name else ''}\n",
         f"{len(args.pair)} 個訓練種子、共同測試句 {len(keys)} 句。數值為各種子平均;"
         f"CI = 以句子為單位重抽 {args.n_boot} 次(各種子用同一組重抽句子,算完再平均)。\n",
         "| 指標 | 基準 | 方法 | 方法 − 基準 [95% CI] | 各種子的差 |", "|---|---|---|---|---|"]
    for key, label in METRICS:
        pb = np.mean([stat(s, "base", key) for s in S])
        pm = np.mean([stat(s, "meth", key) for s in S])
        draws = np.mean([stat(s, "meth", key, sel) - stat(s, "base", key, sel) for s in S], axis=0)
        per_seed = [stat(s, "meth", key) - stat(s, "base", key) for s in S]
        p, lo, hi = ci(pm - pb, draws)
        L.append(f"| {label} | {pb:.1f} | {pm:.1f} | {p:+.1f} [{lo:+.1f}, {hi:+.1f}] | "
                 + " / ".join(f"{v:+.1f}" for v in per_seed) + " |")

    rb = [pearson(d["base"][d["et"]], d["ref_base"][d["et"]]) for d in data]
    rm = [pearson(d["meth"][d["et"]], d["ref_meth"][d["et"]]) for d in data]
    L.append(f"| 入聲長短相關 ρ | {np.mean(rb):.3f} | {np.mean(rm):.3f} | {np.mean(rm) - np.mean(rb):+.3f} | "
             + " / ".join(f"{m - b:+.3f}" for b, m in zip(rb, rm)) + " |")

    mcds = [mcd_per_stem(b, m, keys) for b, m in args.pair]
    if all(x is not None for x in mcds):
        mk = sorted(set.intersection(*[set(x) for x in mcds]))
        arr = np.array([[x[s] for s in mk] for x in mcds])          # 種子 × 句子
        msel = rng.integers(0, len(mk), size=(args.n_boot, len(mk)))
        p, lo, hi = ci(arr.mean(), arr[:, msel].mean(axis=-1).mean(axis=0))
        L.append(f"| MCD (dB) | | | {p:+.3f} [{lo:+.3f}, {hi:+.3f}] | " + " / ".join(f"{v:+.3f}" for v in arr.mean(axis=1)) + " |")

    L += ["", "各種子順序:" + "、".join(labels),
          "", "讀法:CI 不含 0 表示換一批句子結果仍成立;「各種子的差」方向一致表示換訓練種子也成立。"]
    text = "\n".join(L)
    print(text)
    out = Path(args.out).expanduser() if args.out else \
        Path(args.pair[0][1]).expanduser().parent / f"seeds_{args.name or 'result'}.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
