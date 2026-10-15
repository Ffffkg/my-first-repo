"""H2:兩種「看得懂的」指標,給「成功率 0.9、loss 大幅降低」這個目標定基準。不需要 GPU。

hit   時長命中率:入聲(與舒聲)母音核的時長誤差落在容許範圍內的比例,多種子合併、句子重抽 CI。
      容許範圍事先固定三種,不事後挑:±20 ms、±30 ms、±20%(相對)。
      人對語音片段時長差異的分辨閾約 10–20%(或 20–30 ms),所以 ±20% 最接近「聽不出差別」。
loss  訓練 / 驗證 loss:讀每個實驗最新一次 run 的 TensorBoard 紀錄(或 metrics.csv),
      列出每個 loss 的最小值(與步數)和最後一步的值,用來看「還能降多少」以及有沒有過擬合。

    python h2_metrics.py hit --sys base=$E/base_sixian_step40000,... --sys c6=... --out $E/h2_hit_sixian.md
    python h2_metrics.py loss --exp fair1h_joint_sixian_scratch_s1234 --exp c6_tr_dpbig_fair1h_joint_sixian_s1234
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import boolcol, read_cols  # noqa: E402

TOLS = (("±20 ms", lambda e, r: np.abs(e) <= 20), ("±30 ms", lambda e, r: np.abs(e) <= 30),
        ("±20%", lambda e, r: np.abs(e) <= 0.2 * r))


def hit_table(systems, n_boot=2000, seed=42):
    """systems: [(名稱, [每個種子的 syllables_wave 欄位])]。回傳 Markdown 表格列。"""
    rng = np.random.default_rng(seed)
    stems = sorted(set(systems[0][1][0]["stem"]))
    sel = rng.integers(0, len(stems), size=(n_boot, len(stems)))
    pos = {s: i for i, s in enumerate(stems)}
    L = ["| 系統 | 類別 | " + " | ".join(t for t, _ in TOLS) + " |", "|---|---|" + "---|" * len(TOLS)]
    for name, seeds in systems:
        for cls, want in (("入聲", True), ("舒聲", False)):
            cells = []
            for _, f in TOLS:
                hits, ns = np.zeros(len(stems)), np.zeros(len(stems))
                for c in seeds:
                    m = (boolcol(c["entering"]) == want) & np.array([s in pos for s in c["stem"]])
                    idx = np.array([pos[s] for s in c["stem"][m]])
                    ok = f(c["pred_ms"][m] - c["ref_ms"][m], c["ref_ms"][m]).astype(float)
                    hits += np.bincount(idx, weights=ok, minlength=len(stems))
                    ns += np.bincount(idx, minlength=len(stems))
                p = hits.sum() / ns.sum()
                draws = hits[sel].sum(axis=1) / ns[sel].sum(axis=1)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                cells.append(f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
            L.append(f"| {name}({len(seeds)} 種子) | {cls} | " + " | ".join(cells) + " |")
    return L


def read_scalars(run_dir):
    """{tag: [(step, value)]}:先找 TensorBoard 事件檔,沒有就找 metrics.csv。"""
    out = {}
    events = sorted(run_dir.rglob("events.out.tfevents.*"))
    if events:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            events = []
            print("  (沒有 tensorboard 套件,改找 metrics.csv)")
    for ev in events:
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    if not out:
        for f in sorted(run_dir.rglob("metrics.csv")):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                for k, v in r.items():
                    if k not in ("step", "epoch") and v not in ("", None):
                        out.setdefault(k, []).append((int(float(r.get("step") or 0)), float(v)))
    return {k: sorted(v) for k, v in out.items()}


def loss_table(exps):
    L = []
    for exp in exps:
        runs = sorted((C.exp_dir(exp) / "runs").glob("*"))
        if not runs:
            L += [f"### {exp}\n", "★ 找不到 runs/\n"]
            continue
        sc = read_scalars(runs[-1])
        tags = sorted(t for t in sc if "loss" in t.lower())
        L += [f"### {exp}\n", f"run:`{runs[-1].name}`\n", "| loss | 最小值(步數) | 最後一步的值(步數) | 最後 ÷ 最小 |", "|---|---|---|---|"]
        for t in tags:
            v = sc[t]
            s_min, v_min = min(v, key=lambda x: x[1])
            s_last, v_last = v[-1]
            ratio = v_last / v_min if v_min > 0 else float("nan")
            L.append(f"| {t} | {v_min:.4f}({s_min}) | {v_last:.4f}({s_last}) | {ratio:.2f} |")
        if not tags:
            L.append(f"(沒有 loss 紀錄;現有標籤:{sorted(sc)[:10]})")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hit")
    h.add_argument("--sys", action="append", required=True, metavar="NAME=DIR1,DIR2,...")
    h.add_argument("--n-boot", type=int, default=2000)
    h.add_argument("--out")
    lo = sub.add_parser("loss")
    lo.add_argument("--exp", action="append", required=True)
    lo.add_argument("--out")
    args = ap.parse_args()

    if args.cmd == "hit":
        systems = []
        for spec in args.sys:
            name, dirs = spec.split("=", 1)
            systems.append((name, [read_cols(Path(x).expanduser() / "syllables_wave.csv") for x in dirs.split(",") if x]))
        L = ["# H2 時長命中率(誤差在容許範圍內的母音核比例,%)\n",
             "多種子合併;CI 以句子重抽。容許範圍事先固定,±20% 約等於人耳的時長分辨閾。\n"] + hit_table(systems, args.n_boot)
    else:
        L = ["# H2 訓練 / 驗證 loss\n",
             "注意:base(MAS 對齊)和 C2/C6(MFA 時長)的 dur_loss 目標不同,不能直接比大小;"
             "flow-matching 的 diff_loss 有很高的下限(雜訊本身),降不了多少是正常的。\n"] + loss_table(args.exp)
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
