"""V3a:Matcha-TTS 與 VITS 並列比較「時長預測準不準」與「聲音有沒有照著預測的時長發出來」。不需要 GPU。

入聲命中率 = 「時長預測器預測的長度」經過「聲學模型實際發出來」兩步。這裡把兩步拆開(測試集,三個種子平均):
  ① 預測:時長預測器輸出(換成母音核長度)與真人 MFA 母音的平均絕對誤差、ρ
  ② 發音忠實度:同一批音節,「波形量到的母音長度」與「模型自己預測的長度」相差多少(|波形 − 預測|、r)
     差越小、r 越高,代表聲學部分越照著預測的時長發音
  ③ 最終:波形上的入聲命中率、ρ
只有用 MFA 時長訓練的系統(+AD、XAD)預測與波形的 blank 分配方式一致,② 在它們之間才能公平比較;
MAS 系統(Base、X)的 ② 會混入 blank 分配方式的差異,只列出供參考。

    python v3_arch_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import read_cols  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402

E = C.OUT_ROOT / "eval"
SEEDS = (1234, 123, 2026)
SYSTEMS = [("Matcha-TTS", "Base", "base", 40000, False), ("Matcha-TTS", "+AD", "c2", 40000, True),
           ("Matcha-TTS", "XAD", "c6", 40000, True),
           ("VITS", "Base", "vbase", 200000, False), ("VITS", "+AD", "vad", 200000, True),
           ("VITS", "X", "vx", 200000, False), ("VITS", "XAD", "vxad", 200000, True)]


def eval_dir(k, d, s, step):
    sf = "" if s == 1234 else f"_s{s}"
    for name in (f"{k}_{d}{sf}_step{step}_cum", f"{k}_{d}{sf}_step{step}"):
        if (E / name / "d1_rows.csv").exists():
            return E / name
    return None


def main():
    L = ["# V3a:時長預測 vs 發音忠實度(入聲,測試集,三個種子平均)\n",
         "| 腔調 | 架構 | 系統 | ① 預測誤差 (ms) | ① 預測 ρ | ② 波形與預測相差 (ms) | ② 波形與預測 r | ③ 命中率 (%) | ③ 波形 ρ | 種子 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in C.DIALECTS:
        for arch, lab, k, step, mfa in SYSTEMS:
            v = {x: [] for x in ("vde", "rho", "gap", "r", "hit", "wrho")}
            n = 0
            for s in SEEDS:
                ev = eval_dir(k, d, s, step)
                if ev is None:
                    continue
                n += 1
                sm = json.load(open(ev / "summary.json", encoding="utf-8"))
                v["vde"].append(sm["et_vde"])
                v["rho"].append(sm["rho"])
                r = read_cols(ev / "d1_rows.csv")
                et = r["entering"] > 0.5
                w, a = r["wave_ms"][et].astype(float), r["attn_half_ms"][et].astype(float)
                v["gap"].append(np.abs(w - a).mean())
                v["r"].append(pearson(w, a))
                sw = read_cols(ev / "syllables_wave.csv")
                e2 = sw["entering"] > 0.5
                p, ref = sw["pred_ms"][e2].astype(float), sw["ref_ms"][e2].astype(float)
                v["hit"].append(100 * (np.abs(p - ref) <= 30).mean())
                v["wrho"].append(pearson(p, ref))
            if not n:
                continue
            m = {x: np.mean(y) for x, y in v.items()}
            mark = "" if mfa else "(參考)"
            L.append(f"| {d} | {arch} | {lab} | {m['vde']:.1f} | {m['rho']:.3f} | {m['gap']:.1f}{mark} | {m['r']:.3f}{mark} | "
                     f"{m['hit']:.1f} | {m['wrho']:.3f} | {n} |")
    L += ["", "判讀:",
          "- 同樣用 MFA 時長訓練(+AD、XAD),若 VITS 的 ② 相差明顯較大、r 較低 → VITS 沒有照著預測的時長發音,是它比 Matcha-TTS 差的主因;",
          "- 若 VITS 的 ① 預測誤差就比較大 → 主因在時長預測器(例如編碼器表示或訓練方式不同)。"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
