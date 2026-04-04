"""Benchmark Metrics
────────────────────
Defines evaluation metrics for forecasting and early-warning performance.

Forecast metrics:
  - MAE, RMSE, MAPE

EWS alert metrics:
  - precision, recall, F1 for spike detection using a binary ground truth

This module deliberately avoids heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-9) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return 0.0
    denom = np.maximum(np.abs(yt[mask]), eps)
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / denom)))


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    f1: float


def classification_report(y_true: pd.Series, y_pred: pd.Series) -> ClassificationReport:
    yt = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    yp = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int)

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return ClassificationReport(
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
    )


def evaluate_forecast_df(
    actual: pd.Series,
    forecast_df: pd.DataFrame,
    pred_col: str = "forecast",
) -> Dict[str, float]:
    """Convenience evaluation for forecast DataFrames."""
    y_pred = forecast_df[pred_col] if pred_col in forecast_df.columns else pd.Series(dtype=float)
    return {
        "MAE": round(mae(actual, y_pred), 4),
        "RMSE": round(rmse(actual, y_pred), 4),
        "MAPE": round(mape(actual, y_pred), 6),
    }
