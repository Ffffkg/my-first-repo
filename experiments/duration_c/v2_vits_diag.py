"""V2:為什麼 VITS +AD 在海陸變差、在四縣沒有?只讀已有的評估與訓練紀錄,不需要 GPU。

把問題拆成三段,各看一組數字(三個種子平均):
  1. 時長預測器本身準不準(模型輸出的時長 vs 真人 MFA 時長,測試集全部句子):ET-VDE、ρ(預測)
  2. 合成的聲音有沒有照著預測的時長發音:同一批音節「預測時長的 ρ」vs「從波形量的 ρ」
  3. 訓練時聲學部分學得好不好:訓練結束時的 mel、KL、時長損失
另外列出訓練句數(+AD 只用 MFA 對齊成功的句子)。

判讀:
  - 海陸 +AD 在第 1 段就比較差 → 時長預測器沒學好(標籤或學習的問題)
  - 第 1 段不差、第 2 段才差 → VITS 沒有照著時長發音(聲學部分的問題)
  - 海陸 +AD 的 mel / KL 損失明顯比 Base 高,而四縣沒有 → 強制套用 MFA 對齊和 VITS 的聲學表示衝突
注意:MAS 系統(Base、X)和 MFA 系統(+AD、XAD)的 blank 分配方式不同,第 1、2 段「預測」那一欄
在兩類系統之間不能直接比;+AD 與 XAD 之間、同一系統的預測 vs 波形之間可以比。

    python v2_vits_diag.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402

E = C.OUT_ROOT / "eval"
KINDS = {"vbase": ("Base", "vits_base"), "vad": ("+AD", "vits_ad"), "vx": ("X", "vits_x"), "vxad": ("XAD", "vits_xad")}
SEEDS = (1234, 123, 2026)


def sfx(s):
    return "" if s == 1234 else f"_s{s}"


def eval_dir(k, d, s):
    for name in (f"{k}_{d}{sfx(s)}_step200000_cum", f"{k}_{d}{sfx(s)}_step200000"):
        if (E / name / "summary.json").exists():
            return E / name
    return None


def d1_rho(path):
    """d1_summary.md 第 3 節:| ρ | attn | 波形 |"""
    for line in open(path, encoding="utf-8"):
        if line.startswith("| ρ |"):
            parts = [x.strip() for x in line.strip().strip("|").split("|")]
            return float(parts[1]), float(parts[2])
    return np.nan, np.nan


def last_losses(log):
    last = None
    for line in open(log, encoding="utf-8"):
        if line.startswith("step "):
            last = line
    if not last:
        return {}
    return {k: float(v) for k, v in re.findall(r"\b(mel|kl|dur) ([-0-9.]+)", last)}


def n_lines(p):
    try:
        return sum(1 for l in open(p, encoding="utf-8") if l.strip())
    except FileNotFoundError:
        return None


def main():
    import yaml
    L = ["# V2:VITS 各系統的時長預測、發音忠實度與訓練損失(三個種子平均)\n",
         "| 腔調 | 系統 | ① 預測 ET-VDE (ms) | ① 預測 ρ | ② 同批音節:預測 ρ → 波形 ρ | ③ mel 損失 | ③ KL 損失 | ③ 時長損失 | 種子數 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d in C.DIALECTS:
        for k, (lab, run) in KINDS.items():
            vde, rho, ra, rw, mel, kl, dur, n = [], [], [], [], [], [], [], 0
            for s in SEEDS:
                ev = eval_dir(k, d, s)
                if ev is None:
                    continue
                n += 1
                sm = json.load(open(ev / "summary.json", encoding="utf-8"))
                vde.append(sm["et_vde"])
                rho.append(sm["rho"])
                a, w = d1_rho(ev / "d1_summary.md")
                ra.append(a)
                rw.append(w)
                lg = C.VITS_RUNS / f"{run}_fair1h_{d}_s{s}" / "train.log"
                if lg.exists():
                    ls = last_losses(lg)
                    mel.append(ls.get("mel", np.nan))
                    kl.append(ls.get("kl", np.nan))
                    dur.append(ls.get("dur", np.nan))
            if not n:
                continue
            m = lambda v: np.nanmean(v) if v else np.nan  # noqa: E731
            L.append(f"| {d} | {lab} | {m(vde):.1f} | {m(rho):.3f} | {m(ra):.3f} → {m(rw):.3f} | {m(mel):.2f} | "
                     f"{m(kl):.3f} | {m(dur):.4f} | {n} |")
    L += ["", "## 訓練句數(1 小時清單)\n", "| 腔調 | Base、X 的清單 | +AD、XAD 的清單(MFA 對齊成功) |", "|---|---|---|"]
    for d in C.DIALECTS:
        def tl(y):
            try:
                v = Path(yaml.safe_load(open(C.MATCHA / "configs" / "data" / y, encoding="utf-8"))["train_filelist_path"])
                return n_lines(v if v.is_absolute() else C.MATCHA / v)
            except FileNotFoundError:
                return None
        L.append(f"| {d} | {tl(f'fair1h_joint_{d}.yaml')} | {tl(f'fair1h_joint_{d}_mfadur.yaml')} |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
