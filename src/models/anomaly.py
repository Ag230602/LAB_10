"""
Anomaly Detection Module
─────────────────────────
Three complementary methods, combined via a consensus score:

  1. Z-score          – fast statistical baseline
  2. IsolationForest  – detects structural outliers
  3. Rolling IQR      – detects temporal spikes in time-series

The combined anomaly score is 0 (normal) to 1 (strong anomaly).
Falls back gracefully if scikit-learn is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest as _IForest
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# ── Z-score ───────────────────────────────────────────────────────────────────

def zscore_anomaly(
    series: pd.Series,
    threshold: float = 2.5,
) -> pd.Series:
    """
    Returns a boolean Series where True = anomaly (|z| > threshold).
    """
    vals = pd.to_numeric(series, errors="coerce")
    mu   = vals.mean()
    sigma = vals.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series([False] * len(series), index=series.index)
    z = (vals - mu) / sigma
    return z.abs() > threshold


def zscore_scores(series: pd.Series) -> pd.Series:
    """Returns absolute z-scores normalised to [0, 1] (clamped at z=4)."""
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mu    = vals.mean()
    sigma = vals.std()
    if sigma == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    z = (vals - mu) / sigma
    return (z.abs() / 4.0).clip(0, 1)


# ── IsolationForest ───────────────────────────────────────────────────────────

def isolation_forest_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.Series:
    """
    Returns an anomaly score in [0, 1] for each row of *df*.
    Higher = more anomalous.
    """
    if not _SKLEARN_AVAILABLE:
        log.debug("sklearn not available; returning zero anomaly scores.")
        return pd.Series([0.0] * len(df), index=df.index)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    model = _IForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    raw_scores = model.decision_function(X)   # higher = more normal
    # Invert and normalise to [0, 1]
    normalised = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return pd.Series(normalised.clip(0, 1), index=df.index)


# ── Rolling IQR spike detection ───────────────────────────────────────────────

def rolling_iqr_anomaly(
    series: pd.Series,
    window: int = 6,
    multiplier: float = 1.5,
) -> pd.Series:
    """
    Flags values outside rolling [Q1 - m*IQR, Q3 + m*IQR] as anomalies.
    Returns a float Series in [0, 1] where 1 = far outside rolling bounds.
    """
    vals = pd.to_numeric(series, errors="coerce").ffill().fillna(0.0)
    q1   = vals.rolling(window, min_periods=2).quantile(0.25)
    q3   = vals.rolling(window, min_periods=2).quantile(0.75)
    iqr  = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    excess = pd.concat([lower - vals, vals - upper], axis=1).max(axis=1).clip(lower=0)
    denom  = (iqr + 1e-9)
    score  = (excess / (multiplier * denom)).clip(0, 1)
    return score.fillna(0.0)


# ── Consensus detector ────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    value_col: str,
    feature_cols: Optional[list[str]] = None,
    zscore_threshold: float = 2.5,
    rolling_window: int = 6,
    contamination: float = 0.05,
    w_zscore: float = 0.35,
    w_iforest: float = 0.40,
    w_rolling: float = 0.25,
) -> pd.DataFrame:
    """
    Runs all three detectors and returns *df* with appended columns:

      anomaly_zscore     – bool flag from z-score
      anomaly_rolling    – 0–1 score from rolling IQR
      anomaly_iforest    – 0–1 score from IsolationForest
      anomaly_score      – 0–1 consensus score
      is_anomaly         – bool (consensus_score > 0.5)

    Parameters
    ----------
    df            : input DataFrame
    value_col     : column to analyse
    feature_cols  : additional feature columns for IsolationForest
                    (defaults to [value_col])
    """
    out = df.copy()
    feats = feature_cols or [value_col]

    z_flag   = zscore_anomaly(out[value_col], threshold=zscore_threshold)
    z_score  = zscore_scores(out[value_col])
    rl_score = rolling_iqr_anomaly(out[value_col], window=rolling_window)

    valid_feats = [c for c in feats if c in out.columns]
    if valid_feats and _SKLEARN_AVAILABLE:
        if_score = isolation_forest_scores(out, valid_feats, contamination=contamination)
    else:
        if_score = pd.Series([0.0] * len(out), index=out.index)

    out["anomaly_zscore"]  = z_flag
    out["anomaly_rolling"] = rl_score.round(4)
    out["anomaly_iforest"] = if_score.round(4)
    out["anomaly_score"]   = (
        w_zscore  * z_score.fillna(0)
        + w_rolling * rl_score.fillna(0)
        + w_iforest * if_score.fillna(0)
    ).clip(0, 1).round(4)
    out["is_anomaly"] = out["anomaly_score"] > 0.5

    return out
