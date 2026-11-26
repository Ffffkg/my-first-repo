"""實驗 F1:入聲(檢查音)依類型分組的時長準確度。

輸入任何逐音節 CSV(需有 ref_ms、pred_ms):c_eval.py 的 syllables.csv(attn)、
d1_compare.py 的 syllables_wave.csv(波形,建議),或 c1 的變體 CSV。可以一次給多個系統並排比較。

    python f1_subgroups.py ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base/syllables_wave.csv \
                           ~/hakka_tts/exp_c/eval/c2_sixian_step40000/syllables_wave.csv

每個分組報:n、平均偏差 VBias(95% CI)、平均絕對誤差 VDE、ρ。分組因素事先固定(韻尾、調值、停頓前、性別),
並與舒聲同一因素並列,用來看「是入聲特有,還是所有音節都這樣」。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, fmt, mean_ci, pearson  # noqa: E402

FACTORS = (("coda", "韻尾"), ("tone", "調值"), ("prepausal", "停頓前"), ("gender", "性別"))


def show(v):
    return int(v) if isinstance(v, float) and float(v).is_integer() else v


def group_stats(c, m, n_boot):
    err = c["pred_ms"][m] - c["ref_ms"][m]
    t = UttTable(c["stem"][m], {"e": err}, {"all": np.ones(m.sum(), bool)})
    return (int(m.sum()), mean_ci(t, "e", "all", boot_idx(t.U, n_boot)), float(np.abs(err).mean()),
            pearson(c["pred_ms"][m], c["ref_ms"][m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", help="Markdown 輸出(預設第一個 CSV 旁的 f1_subgroups.md)")
    args = ap.parse_args()

    systems = [(Path(p).expanduser().parent.name or Path(p).stem, read_cols(Path(p).expanduser())) for p in args.csv]
    L = ["# F1 分組:時長準確度(VBias = 預測 − 參考,ms)\n",
         "系統:" + "、".join(f"`{n}`" for n, _ in systems) + "\n"]
    for key, label in (("__class__", "類別"),) + FACTORS:
        L += [f"## {label}\n", "| 類別 | 值 | " + " | ".join(f"{n} VBias / VDE / ρ (n)" for n, _ in systems) + " |",
              "|---|---|" + "---|" * len(systems)]
        for cls_name, cls_val in (("入聲", True), ("舒聲", False)):
            values = [None] if key == "__class__" else sorted(set(systems[0][1][key][boolcol(systems[0][1]["entering"]) == cls_val]))
            for v in values:
                cells = []
                for _, c in systems:
                    m = boolcol(c["entering"]) == cls_val
                    if v is not None:
                        m &= c[key] == v
                    if m.sum() < args.min_n:
                        cells.append(f"— ({m.sum()})")
                        continue
                    n, vb, vde, rho = group_stats(c, m, args.n_boot)
                    cells.append(f"{fmt(vb)} / {vde:.1f} / {rho:.2f} ({n})")
                if all(x.startswith("—") for x in cells):
                    continue
                L.append(f"| {cls_name} | {'全部' if v is None else show(v)} | " + " | ".join(cells) + " |")
        L.append("")
    text = "\n".join(L)
    print(text)
    out = Path(args.out) if args.out else Path(args.csv[0]).expanduser().with_name("f1_subgroups.md")
    out.write_text(text, encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
