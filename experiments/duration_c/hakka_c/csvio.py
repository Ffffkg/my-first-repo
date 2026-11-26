"""逐音節 CSV 的讀寫(不依賴 pandas)。"""
import csv
from pathlib import Path

import numpy as np


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"沒有資料可寫入 {path}")
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_cols(path):
    """回傳 {欄名: np.array};能轉成數字的欄位會轉成 float。"""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} 是空的")
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def boolcol(v):
    if v.dtype == object:
        return np.array([str(x).lower() in ("1", "true", "1.0") for x in v])
    return v.astype(float) > 0.5
