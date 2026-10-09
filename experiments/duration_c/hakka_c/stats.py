"""以句子為單位的 cluster bootstrap(與 etvdb_ci.py、delta_bias_ci.py 同一套想法),只用 numpy。

所有統計量都寫成「每句的加總 / 每句的個數」,重抽句子後再相除,所以一次重抽可以同時算
ET-VBias、NonET-VBias、ΔBias,以及兩個系統之間的配對差。
"""
import numpy as np


class UttTable:
    """把逐音節資料彙整成每句的 (加總, 個數)。

    utt   : 每個音節所屬句子的 key
    cols  : {名稱: 逐音節數值}
    masks : {名稱: 逐音節布林遮罩}(例如 'et'、'non')
    """

    def __init__(self, utt, cols, masks):
        keys, inv = np.unique(np.asarray(utt), return_inverse=True)
        self.keys, self.U = keys, len(keys)
        self.S, self.C = {}, {}
        for mname, m in masks.items():
            m = np.asarray(m, bool)
            self.C[mname] = np.bincount(inv, weights=m.astype(float), minlength=self.U)
            for cname, v in cols.items():
                v = np.asarray(v, float)
                self.S[(cname, mname)] = np.bincount(inv, weights=np.where(m, v, 0.0), minlength=self.U)

    def mean(self, col, mask, idx=None):
        s, c = self.S[(col, mask)], self.C[mask]
        if idx is None:
            return s.sum() / c.sum()
        return s[idx].sum(axis=-1) / c[idx].sum(axis=-1)


def boot_idx(U, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, U, size=(n_boot, U))


def ci(point, draws, alpha=0.05):
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mean_ci(t, col, mask, idx):
    return ci(t.mean(col, mask), t.mean(col, mask, idx))


def delta_ci(t, col, idx, a="et", b="non"):
    """ΔBias = mean(col | a) − mean(col | b),配對(同一批句子)。"""
    p = t.mean(col, a) - t.mean(col, b)
    d = t.mean(col, a, idx) - t.mean(col, b, idx)
    return ci(p, d)


def fmt(c, digits=1):
    p, lo, hi = c
    return f"{p:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def excludes_zero(c):
    return c[1] > 0 or c[2] < 0


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ols_boot(y, X, utt, n_boot=2000, seed=42):
    """y = Xβ + ε,以句子為單位重抽估 β 的 CI。回傳 (β, lo, hi) 陣列。"""
    y, X = np.asarray(y, float), np.asarray(X, float)
    keys, inv = np.unique(np.asarray(utt), return_inverse=True)
    rows = [np.flatnonzero(inv == u) for u in range(len(keys))]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[u] for u in rng.integers(0, len(keys), len(keys))])
        draws.append(np.linalg.lstsq(X[pick], y[pick], rcond=None)[0])
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return beta, lo, hi
