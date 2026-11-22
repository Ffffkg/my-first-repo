"""ICEIC 投稿後研究進度報告(Word)用的圖。數字全部來自實驗 log(見報告附錄)。
    python make_figs_202611.py <輸出資料夾>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if Path(f).exists():
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "font.size": 11, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "font.style": "normal"})
SIX, HAI, MAT, VIT, GRID, TXT = "#4E6A9E", "#3A7D44", "#4E6A9E", "#C98A1B", "#E3E6EB", "#1B2333"


def bars(ax, cats, series, colors, ylim, fmt="{:.1f}", fs=9):
    n = len(series)
    w = 0.8 / n
    for k, ((name, vals), c) in enumerate(zip(series, colors)):
        xs = [i + (k - (n - 1) / 2) * w for i in range(len(cats))]
        b = ax.bar(xs, vals, w, label=name, color=c, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.4 if v >= 0 else -0.012) if ylim[0] >= 0 else v - 0.012, fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=fs, color=TXT)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, zorder=0)


# 圖 1:實驗一 入聲命中率
fig, ax = plt.subplots(figsize=(8, 3.6))
cats = ["Base", "+AD", "+AD＊", "XAD", "XAD＊", "48 小時\nBase"]
bars(ax, cats, [("四縣", [71.0, 77.0, 75.4, 81.3, 79.1, 81.1]), ("海陸", [69.4, 77.7, 77.4, 82.2, 82.4, 83.0])],
     [SIX, HAI], (50, 90))
ax.set_ylabel("入聲命中率（％）")
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig1_x4.png", dpi=200)

# 圖 2:兩個架構、兩個腔調的入聲命中率
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
cfg = ["Base", "+AD", "X", "XAD"]
data = {"四縣": ([71.0, 77.0, 76.1, 81.3], [68.3, 70.6, 70.7, 74.8]),
        "海陸": ([69.4, 77.7, 78.1, 82.2], [75.3, 67.6, 72.9, 73.8])}
for ax, (d, (m, v)) in zip(axs, data.items()):
    bars(ax, cfg, [("Matcha-TTS", m), ("VITS", v)], [MAT, VIT], (50, 90))
    ax.set_title(d)
axs[0].set_ylabel("入聲命中率（％）")
axs[0].legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig2_arch_hit.png", dpi=200)

# 圖 3:與各自 Base 的差(95％ 信賴區間)
rows = {"四縣": [("Matcha +AD", 6.0, 2.8, 9.1, MAT), ("Matcha XAD", 10.3, 6.9, 13.9, MAT), ("VITS +AD", 2.3, -0.4, 5.2, VIT),
                 ("VITS X", 2.4, -0.9, 5.5, VIT), ("VITS XAD", 6.5, 3.2, 9.6, VIT)],
        "海陸": [("Matcha +AD", 8.3, 5.3, 11.3, MAT), ("Matcha XAD", 12.8, 9.1, 16.5, MAT), ("VITS +AD", -7.7, -11.3, -4.0, VIT),
                 ("VITS X", -2.4, -6.1, 1.3, VIT), ("VITS XAD", -1.5, -5.9, 2.6, VIT)]}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (d, rs) in zip(axs, rows.items()):
    for i, (lab, v, lo, hi, c) in enumerate(rs):
        y = len(rs) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=2.5)
        ax.plot(v, y, "o", color=c, ms=7)
        ax.text(23.5, y, f"{v:+.1f}", va="center", ha="right", fontsize=9, color=TXT)
    ax.axvline(0, color="#5A6372", ls="--", lw=1)
    ax.set_yticks(range(len(rs)), [r[0] for r in rs][::-1])
    ax.set_xlim(-13, 24)
    ax.set_title(d)
    ax.set_xlabel("入聲命中率差（百分點）")
    ax.grid(axis="x", color=GRID)
fig.tight_layout()
fig.savefig(OUT / "fig3_forest.png", dpi=200)

# 圖 4:MCD 與各自 Base 的差
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
mcd = {"四縣": ([-0.119, -0.313, -0.332], [-0.089, -0.095, -0.076]),
       "海陸": ([-0.096, -0.252, -0.235], [-0.020, -0.222, -0.217])}
for ax, (d, (m, v)) in zip(axs, mcd.items()):
    n = 2
    w = 0.38
    for k, (vals, c, name) in enumerate(((m, MAT, "Matcha-TTS"), (v, VIT, "VITS"))):
        xs = [i + (k - 0.5) * w for i in range(3)]
        ax.bar(xs, vals, w, color=c, label=name, zorder=2)
        for x, val in zip(xs, vals):
            ax.text(x, val - 0.01, f"{val:.2f}", ha="center", va="top", fontsize=9, color=TXT)
    ax.set_xticks(range(3), ["+AD", "X", "XAD"])
    ax.set_ylim(-0.4, 0.02)
    ax.axhline(0, color="#5A6372", lw=1)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_title(d)
axs[0].set_ylabel("MCD 差（dB，越低越好）")
axs[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig4_mcd.png", dpi=200)

# 圖 5:VITS 系統的平均偏差與命中率
pts = [("四縣 Base", -12.2, 68.3, SIX, "o"), ("四縣 +AD", -12.6, 70.6, SIX, "s"), ("四縣 X", -7.8, 70.7, SIX, "^"), ("四縣 XAD", -15.2, 74.8, SIX, "D"),
       ("海陸 Base", -10.1, 75.3, HAI, "o"), ("海陸 +AD", 1.0, 67.6, HAI, "s"), ("海陸 X", -2.1, 72.9, HAI, "^"), ("海陸 XAD", 2.3, 73.8, HAI, "D")]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for lab, x, y, c, mk in pts:
    ax.plot(x, y, mk, color=c, ms=8)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color=TXT)
ax.axvline(0, color="#5A6372", ls="--", lw=1)
ax.set_xlabel("入聲母音平均偏差（毫秒，負值＝太短）")
ax.set_ylabel("入聲命中率（％）")
ax.set_xlim(-19, 8)
ax.grid(color=GRID)
fig.tight_layout()
fig.savefig(OUT / "fig5_bias.png", dpi=200)
print("ok")
