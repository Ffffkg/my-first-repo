"""實驗 E2 分析:成對比較聽測(e_build_ab.py 產生的測驗)。

    python e_analyze.py --key ~/hakka_tts/exp_e/sixian_base_vs_c2/key.csv --responses ~/hakka_tts/exp_e/sixian_base_vs_c2/responses

分數方向:轉成「B − A」,正值代表系統 B(通常是新方法)比較好。
  q1 = 整體自然度,q2 = 入聲字自然度,範圍 −3 … +3。

事先登記的規則:
  * 注意力檢查:「兩邊相同」的題目,|分數| 平均 ≥ 2 的聽者整份排除(報告會列出)。
  * CI:聽者與句子交叉重抽(crossed bootstrap)5000 次,同時反映「換一批聽者」與「換一批句子」的變異。
  * 另報:偏好比例(B 好 / 一樣 / A 好)、每位聽者平均、聽者層級的符號檢定。
"""
import argparse
import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def crossed_boot(M, n_boot=5000, seed=42):
    """M:聽者 × 題目 的分數矩陣(缺值為 nan)。回傳平均與 95% CI。"""
    rng = np.random.default_rng(seed)
    L, T = M.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sub = M[rng.integers(0, L, L)][:, rng.integers(0, T, T)]
        draws[b] = np.nanmean(sub)
    return float(np.nanmean(M)), *np.percentile(draws, [2.5, 97.5])


def sign_test(x):
    """聽者平均 > 0 的人數之雙尾符號檢定(排除 0)。"""
    x = [v for v in x if v != 0]
    n, k = len(x), sum(v > 0 for v in x)
    if n == 0:
        return float("nan"), 0, 0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail), k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--responses", required=True, help="放聽者 CSV 的資料夾(listening_*.csv)")
    ap.add_argument("--same-threshold", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    key = {r["trial"]: r for r in read_csv(args.key)}
    files = sorted(Path(args.responses).expanduser().glob("*.csv"))
    if not files:
        sys.exit("★ 沒有找到回傳的 CSV")
    resp = defaultdict(dict)
    meta = {}
    for f in files:
        for r in read_csv(f):
            lid = r["listener"].strip() or f.stem
            meta[lid] = (r.get("dialect", ""), r.get("frequency", ""))
            k = key[r["trial"]]
            q1, q2 = int(r["q1"]), int(r["q2"])
            if k["kind"] == "same":
                resp[lid][r["trial"]] = ("same", abs(q1), abs(q2))
            else:
                sgn = 1 if k["pos2"] == "B" else -1        # 位置 2 是 B → 分數本身就是 B − A
                resp[lid][r["trial"]] = ("ab", sgn * q1, sgn * q2)

    L = ["# E2 成對比較聽測結果\n", f"回收 {len(resp)} 位聽者。分數 = B − A(正值:B 較好),範圍 −3…+3。\n",
         "## 注意力檢查(兩邊相同的題目)\n", "| 聽者 | 腔調 | 使用頻率 | 相同題平均 |q1| | 相同題平均 |q2| | 結果 |", "|---|---|---|---|---|---|"]
    keep = []
    for lid in sorted(resp):
        same = [v for v in resp[lid].values() if v[0] == "same"]
        m1 = np.mean([s[1] for s in same]) if same else float("nan")
        m2 = np.mean([s[2] for s in same]) if same else float("nan")
        ok = not same or (m1 < args.same_threshold and m2 < args.same_threshold)
        keep += [lid] if ok else []
        L.append(f"| {lid} | {meta[lid][0]} | {meta[lid][1]} | {m1:.2f} | {m2:.2f} | {'保留' if ok else '★排除'} |")
    if not keep:
        sys.exit("\n".join(L) + "\n\n★ 所有聽者都沒通過注意力檢查")

    trials = sorted(t for t, k in key.items() if k["kind"] != "same")
    L += ["", f"## 結果(保留 {len(keep)} 位聽者 × {len(trials)} 題)\n",
          "| 問題 | CMOS(B − A)[95% CI] | B 較好 | 一樣 | A 較好 | 聽者符號檢定 |", "|---|---|---|---|---|---|"]
    for qi, label in ((1, "整體自然度"), (2, "入聲字自然度")):
        M = np.full((len(keep), len(trials)), np.nan)
        for a, lid in enumerate(keep):
            for b, t in enumerate(trials):
                if t in resp[lid]:
                    M[a, b] = resp[lid][t][qi]
        mean, lo, hi = crossed_boot(M, args.n_boot)
        v = M[~np.isnan(M)]
        p, k, n = sign_test(np.nanmean(M, axis=1))
        L.append(f"| {label} | {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] | {np.mean(v > 0):.0%} | {np.mean(v == 0):.0%} | "
                 f"{np.mean(v < 0):.0%} | {k}/{n} 位偏好 B,p = {p:.3f} |")
    L += ["", "## 每位聽者的平均(q1 整體 / q2 入聲)\n", "| 聽者 | q1 | q2 |", "|---|---|---|"]
    for lid in keep:
        ab = [v for v in resp[lid].values() if v[0] == "ab"]
        L.append(f"| {lid} | {np.mean([x[1] for x in ab]):+.2f} | {np.mean([x[2] for x in ab]):+.2f} |")
    L += ["", "解讀:CI 不含 0 才表示聽者整體上分得出差別;CMOS 絕對值 < 0.3 通常代表差異很小。"]
    text = "\n".join(L)
    print(text)
    out = Path(args.responses).expanduser() / "e2_summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
