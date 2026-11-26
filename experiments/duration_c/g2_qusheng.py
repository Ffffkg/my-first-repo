"""G2:去聲方向不對稱的「音節層級」檢驗(補手冊第 7.6、13.2、20 章 analyze_qusheng.py 的初步分析)。不需要 GPU。

假設(手冊第 2.2、4.5 節):四縣去聲合併成 55,海陸分成陰去 11 / 陽去 33。
  → 四縣→海陸的遷移模型,在「海陸去聲」上應該額外吃虧(來源端沒有 11/33 的區別);
    海陸→四縣則不應該(兩個去聲併成一個比較容易)。
原本的初步分析用「整句去聲比例」近似;這裡直接在每個音節上量:
  F0 誤差 = 合成母音核 5 點 F0 與同一句真人錄音同一音節的 5 點 F0 的 RMSE(半音,各自語者正規化)
  去聲額外誤差 = 去聲音節平均誤差 − 其他舒聲音節平均誤差(同一個系統內相減,排除「某個系統整體較差」)
  方向效應 = 遷移模型的去聲額外誤差 − 從零模型的去聲額外誤差(配對、以句子重抽)
海陸另外報:真的陰去/陽去被 G1 分類器判成「另一個去聲」的比例(11↔33 混淆),需要 g1_tone_eval.py 的 pred_*.csv。

    G=~/hakka_tts/exp_c/g1; E=~/hakka_tts/exp_c/eval
    python g2_qusheng.py --dialect hailu --real $G/real_hailu_test.csv \\
        --pair joint $E/taai_hailu_joint_scratch/syl_f0.csv $E/taai_hailu_joint_transfer/syl_f0.csv \\
        --pair separate $E/taai_hailu_sep_scratch/syl_f0.csv $E/taai_hailu_sep_transfer/syl_f0.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.f0feat import speaker_medians, to_semitone  # noqa: E402
from g1_tone_eval import load, tone_str  # noqa: E402

QUSHENG = {"sixian": {"55"}, "hailu": {"11", "33"}}
ENTERING = {"2", "5"}


def contour_errors(real, syn):
    """每個合成音節(每個生成種子)的 F0 RMSE(半音);對不上真人音節的回傳 nan。"""
    rst = to_semitone(real["spk"], real["pts"], speaker_medians(real["spk"], real["pts"]))
    sst = to_semitone(syn["spk"], syn["pts"], speaker_medians(syn["spk"], syn["pts"]))
    ref = {(s, i): rst[n] for n, (s, i) in enumerate(zip(real["stem"], real["syl_idx"]))}
    err = np.full(len(syn["stem"]), np.nan)
    for n, (s, i) in enumerate(zip(syn["stem"], syn["syl_idx"])):
        r = ref.get((s, i))
        if r is None:
            continue
        d = sst[n] - r
        d = d[np.isfinite(d)]
        if d.size >= 3:
            err[n] = float(np.sqrt(np.mean(d ** 2)))
    return err


def per_stem(stems, vals, mask, keys):
    pos = {k: n for n, k in enumerate(keys)}
    ok = mask & np.isfinite(vals) & np.array([s in pos for s in stems])
    idx = np.array([pos[s] for s in stems[ok]], dtype=int)
    return (np.bincount(idx, weights=vals[ok], minlength=len(keys)), np.bincount(idx, minlength=len(keys)).astype(float))


def mean_of(S, sel=None):
    s, n = S
    if sel is None:
        return s.sum() / max(n.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[sel].sum(axis=-1) / n[sel].sum(axis=-1)


def extra(t, err, keys, sel, qs):
    """去聲額外誤差(點估計, 重抽分布)與兩組各自的平均。"""
    tone = t["tone"]
    q = np.isin(tone, list(qs))
    other = ~q & ~np.isin(tone, list(ENTERING))
    Sq, So = per_stem(t["stem"], err, q, keys), per_stem(t["stem"], err, other, keys)
    return mean_of(Sq) - mean_of(So), mean_of(Sq, sel) - mean_of(So, sel), mean_of(Sq), mean_of(So)


def confusion_1133(pred_path, keys, sel):
    """海陸:真的 11/33 被判成另一個去聲的比例。"""
    with open(Path(pred_path).expanduser(), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stems = np.array([r["stem"] for r in rows])
    y = np.array([tone_str(r["tone"]) for r in rows])
    p = np.array([tone_str(r["pred"]) for r in rows])
    mask = np.isin(y, ["11", "33"])
    conf = ((y == "11") & (p == "33")) | ((y == "33") & (p == "11"))
    S = per_stem(stems, conf.astype(float), mask, keys)
    return mean_of(S), mean_of(S, sel)


def fmt(p, d, scale=1.0, digits=2):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return f"{scale * p:+.{digits}f} [{scale * lo:+.{digits}f}, {scale * hi:+.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS, help="目標腔(合成的腔調)")
    ap.add_argument("--real", required=True, help="同一批測試句的真人特徵(g1_tone_feats.py real --split test)")
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("LABEL", "SCRATCH", "TRANSFER"),
                    help="一種表徵(joint / separate)的 從零、遷移 syl_f0.csv")
    ap.add_argument("--pred", nargs=3, action="append", default=[], metavar=("LABEL", "SCRATCH_PRED", "TRANSFER_PRED"),
                    help="(海陸)g1_tone_eval.py 輸出的 pred_*.csv,算 11↔33 混淆")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", help="Markdown 輸出(預設 ~/hakka_tts/exp_c/g2/qusheng_<dialect>.md)")
    args = ap.parse_args()

    qs = QUSHENG[args.dialect]
    src = "海陸" if args.dialect == "sixian" else "四縣"
    real = load(args.real)
    rng = np.random.default_rng(42)
    tabs = {lab: (load(a), load(b)) for lab, a, b in args.pair}
    keys = sorted(set(real["stem"]).intersection(*[set(t["stem"]) for pair in tabs.values() for t in pair]))
    if not keys:
        sys.exit("★ 真人與合成沒有共同句子")
    sel = rng.integers(0, len(keys), size=(args.n_boot, len(keys)))
    pred_hint = "海陸→四縣:預測方向效應 ≈ 0" if args.dialect == "sixian" else "四縣→海陸:預測方向效應 > 0(遷移在去聲額外吃虧)"
    L = [f"# G2 去聲方向不對稱(音節層級):目標 {'四縣' if args.dialect == 'sixian' else '海陸'}(遷移來源 {src})\n",
         f"共同測試句 {len(keys)} 句;去聲 = {'/'.join(sorted(qs))};「其他舒聲」不含入聲。F0 誤差為半音 RMSE。"
         f"CI 以句子重抽 {args.n_boot} 次。{pred_hint}。\n",
         "| 表徵 | 系統 | 去聲誤差 | 其他舒聲誤差 | 去聲額外誤差 [95% CI] |", "|---|---|---|---|---|"]
    effects = []
    for lab, (sc, tr) in tabs.items():
        res = {}
        for name, t in (("從零", sc), ("遷移", tr)):
            res[name] = extra(t, contour_errors(real, t), keys, sel, qs)
            p, d, mq, mo = res[name]
            L.append(f"| {lab} | {name} | {mq:.2f} | {mo:.2f} | {fmt(p, d)} |")
        p = res["遷移"][0] - res["從零"][0]
        d = res["遷移"][1] - res["從零"][1]
        effects.append((lab, p, d))
    L += ["", "## 方向效應(遷移的去聲額外誤差 − 從零的去聲額外誤差,半音)\n", "| 表徵 | 方向效應 [95% CI] |", "|---|---|"]
    for lab, p, d in effects:
        L.append(f"| {lab} | {fmt(p, d)} |")

    if args.pred:
        if args.dialect != "hailu":
            print("(--pred 只對海陸有意義,略過)")
        else:
            L += ["", "## 海陸陰去 11 ↔ 陽去 33 混淆率(%,G1 分類器)\n", "| 表徵 | 從零 | 遷移 | 遷移 − 從零 [95% CI] |", "|---|---|---|---|"]
            for lab, ps, pt in args.pred:
                a, da = confusion_1133(ps, keys, sel)
                b, db = confusion_1133(pt, keys, sel)
                L.append(f"| {lab} | {100 * a:.1f} | {100 * b:.1f} | {fmt(b - a, db - da, 100, 1)} |")

    L += ["", "讀法:方向效應的 CI 不含 0 且為正 → 遷移在去聲上額外吃虧;兩個目標腔都要跑,比較方向是否不對稱。"]
    text = "\n".join(L)
    print(text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "g2" / f"qusheng_{args.dialect}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
