"""音節母音核的 F0 特徵(給聲調辨識與去聲分析用)。

與手冊第 4.1 節相同的原則:整句算一次 F0,再取母音核區間的值(不單獨切母音算,避免分析窗失真);
F0 以 pyworld dio + stonemask 抽(與 evaluate_v2.py 同一套),時間步長 5 ms。
母音核取 5 個點(10/30/50/70/90%),每點取 ±10 ms 內濁音幀的中位數,存 log2(Hz);
語者正規化(半音 = 12 × (log2 F0 − 語者中位數))留到分析時做,同一份特徵表可以重複使用。
"""
import numpy as np

FRAME_MS = 5.0
POINTS = (0.1, 0.3, 0.5, 0.7, 0.9)
HALF_WIN_S = 0.010


def f0_track(wav, sr, f0_floor=60.0, f0_ceil=500.0):
    """回傳 (f0 Hz,無聲為 0;每幀時間 s)。"""
    import pyworld as pw
    x = np.ascontiguousarray(np.asarray(wav, dtype=np.float64))
    f0, t = pw.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    return f0, t


def nucleus_points(f0, t, t0, t1):
    """母音核 [t0, t1] 的 5 點 log2 F0(沒有濁音則 nan)與濁音比例。"""
    pts = []
    for q in POINTS:
        c = t0 + q * (t1 - t0)
        m = (t >= c - HALF_WIN_S) & (t <= c + HALF_WIN_S) & (f0 > 0)
        pts.append(float(np.log2(np.median(f0[m]))) if m.any() else np.nan)
    inside = (t >= t0) & (t <= t1)
    vfrac = float((f0[inside] > 0).mean()) if inside.any() else 0.0
    return pts, vfrac


def speaker_medians(spk, pts):
    """每位語者所有點的 log2 F0 中位數。spk: (N,);pts: (N, 5)。"""
    med = {}
    for s in np.unique(spk):
        v = pts[spk == s].ravel()
        v = v[np.isfinite(v)]
        med[s] = float(np.median(v)) if v.size else np.nan
    return med


def to_semitone(spk, pts, med):
    return 12.0 * (pts - np.array([med[s] for s in spk])[:, None])
