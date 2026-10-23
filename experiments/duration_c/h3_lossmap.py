"""H3 探針 B:時長 loss 的質量在哪些符號上?只用 CPU,幾秒。

Matcha 的時長 loss = 所有符號(含 blank、停頓)的 (log 預測 − log(目標 frame + 1e-8))² 的平均。
問題:這個數字「大幅降低」會不會真的反映入聲母音核變準?這裡把驗證集的 loss 依符號類別拆開,看各類佔多少。

輸入是 c3_target_check.py 在驗證集輸出的 recs.pkl(每句有目標 d 與預測器連續輸出 w):
    python h3_lossmap.py --recs 1h=$C3/c6val_c6_sixian/recs.pkl --recs 10h=$C3/c10val_c6_10_sixian/recs.pkl

類別:blank(目標 = 1 frame)、blank(目標 > 1 frame = 裡面有靜音)、停頓符號、聲母、入聲母音核、舒聲母音核、韻尾、其他。
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.structure import is_pause  # noqa: E402

CLASSES = ("blank_1", "blank_gap", "pause", "onset", "nuc_et", "nuc_non", "coda", "other")
LABELS = {"blank_1": "blank(目標 1 frame)", "blank_gap": "blank(目標 >1 frame,含靜音)", "pause": "停頓符號",
          "onset": "聲母(母音前)", "nuc_et": "入聲母音核", "nuc_non": "舒聲母音核", "coda": "韻尾(母音後)", "other": "其他"}


def classify(rec):
    d = np.asarray(rec["d"], float)
    L = len(d)
    cls = np.array(["other"] * L, dtype=object)
    for p in range(0, L, 2):
        cls[p] = "blank_1" if d[p] <= 1 else "blank_gap"
    for i, nm in enumerate(rec["names"]):
        if is_pause(nm):
            cls[2 * i + 1] = "pause"
    for s in rec["syllables"]:
        for j in s["sym_idx"]:
            if j < s["nuc"]:
                cls[2 * j + 1] = "onset"
            elif j > s["nuc"]:
                cls[2 * j + 1] = "coda"
        cls[2 * s["nuc"] + 1] = "nuc_et" if s["entering"] else "nuc_non"
    return cls


def loss_by_class(recs):
    tot = {c: 0.0 for c in CLASSES}
    cnt = {c: 0 for c in CLASSES}
    for rec in recs:
        e = np.log(np.maximum(np.asarray(rec["w"], float), 1e-8)) - np.log(1e-8 + np.asarray(rec["d"], float))
        cls = classify(rec)
        for c in CLASSES:
            m = cls == c
            tot[c] += float((e[m] ** 2).sum())
            cnt[c] += int(m.sum())
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", action="append", required=True, metavar="LABEL=recs.pkl")
    ap.add_argument("--out")
    args = ap.parse_args()

    res = []
    for spec in args.recs:
        label, path = spec.split("=", 1)
        recs = pickle.load(open(Path(path).expanduser(), "rb"))["recs"]
        tot, cnt = loss_by_class(recs)
        res.append((label, tot, cnt, len(recs)))
    L = ["# H3 時長 loss 的質量分布(驗證集)\n",
         "每格:該類符號的均方 log 誤差 /「佔總 loss 的比例」。總 loss = 全部符號的平均(與訓練紀錄的 val_dur_loss 同定義,可核對)。\n",
         "| 類別 | 佔符號數 | " + " | ".join(lab for lab, *_ in res) + " |", "|---|---|" + "---|" * len(res)]
    n0 = sum(res[0][2].values())
    for c in CLASSES:
        if all(r[2][c] == 0 for r in res):
            continue
        cells = []
        for _, tot, cnt, _ in res:
            total = sum(tot.values())
            cells.append(f"{tot[c] / max(cnt[c], 1):.3f} / {100 * tot[c] / total:.0f}%")
        L.append(f"| {LABELS[c]} | {100 * res[0][2][c] / n0:.0f}% | " + " | ".join(cells) + " |")
    L.append("| **總 loss(平均)** | 100% | " + " | ".join(f"**{sum(t.values()) / sum(c.values()):.4f}**" for _, t, c, _ in res) + " |")
    L += ["", "讀法:", "* 總 loss 要大幅降低,必須降低「佔比大」的類別;母音核的佔比若很小,降低總 loss 不代表入聲變準。",
          "* 「blank(含靜音)」與「停頓」的時長本來就變異極大(說話者何時換氣、停多久),文字預測不到,是 loss 的主要下限來源。"]
    text = "\n".join(L)
    print(text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "h3" / "lossmap.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
