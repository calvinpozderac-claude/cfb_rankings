"""Evaluation metrics for binary (home-win) probabilistic predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def accuracy(y, p):
    return float(np.mean((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)))


def log_loss(y, p):
    p = _clip(p)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def evaluate(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"n": 0, "acc": np.nan, "logloss": np.nan, "brier": np.nan}
    return {
        "n": int(len(y)),
        "acc": accuracy(y, p),
        "logloss": log_loss(y, p),
        "brier": brier(y, p),
    }


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    """Reliability table: predicted vs observed home-win rate by probability bin."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(p)
    y, p = y[mask], p[mask]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()),
            "obs_rate": float(y[m].mean()),
        })
    return pd.DataFrame(rows)
