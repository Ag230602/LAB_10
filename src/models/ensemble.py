"""
Ensemble Forecasting Module
─────────────────────────────
Combines three forecasting methods with learned weights to produce a
final ensemble prediction with confidence intervals.

Components
──────────
  1. LinearRegression  (sklearn)
  2. Holt's Double Exponential Smoothing (via statsmodels if available)
  3. ARIMA             (via src.models.arima_forecast)

Weights are set by inverse-MAE on a held-out validation split
(last 20% of the training series).  If a component fails, its weight
is redistributed to the remaining components.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.arima_forecast import arima_forecast

log = logging.getLogger(__name__)


# ── Individual component forecasters ─────────────────────────────────────────

def _linear_forecast(train: np.ndarray, horizon: int) -> np.ndarray:
    n = len(train)
    X = np.arange(n).reshape(-1, 1)
    model = LinearRegression().fit(X, train)
    X_fut = np.arange(n, n + horizon).reshape(-1, 1)
    return model.predict(X_fut)


def _holt_forecast(train: np.ndarray, horizon: int) -> np.ndarray:
    """Holt double-exponential smoothing (no external dependency)."""
    alpha, beta = 0.3, 0.1
    level = float(train[0])
    trend = float(train[1] - train[0]) if len(train) > 1 else 0.0
    for v in train[1:]:
        prev = level
        level = alpha * float(v) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return np.array([level + h * trend for h in range(1, horizon + 1)])


def _arima_component(train: pd.Series, horizon: int) -> np.ndarray:
    df = arima_forecast(train, horizon=horizon, max_p=2, max_q=2)
    return df["forecast"].to_numpy()


# ── Validation-based weight learning ─────────────────────────────────────────

def _learn_weights(
    series: np.ndarray,
    horizon: int,
    val_frac: float = 0.20,
) -> Dict[str, float]:
    """
    Splits *series* into train / val, forecasts horizon=len(val) steps,
    and assigns inverse-MAE weights.
    """
    n_val = max(1, int(len(series) * val_frac))
    n_train = len(series) - n_val
    if n_train < 3:
        return {"linear": 1 / 3, "holt": 1 / 3, "arima": 1 / 3}

    train = series[:n_train]
    val   = series[n_train:]
    h     = len(val)

    maes: Dict[str, float] = {}
    for name, fn in [
        ("linear", lambda: _linear_forecast(train, h)),
        ("holt",   lambda: _holt_forecast(train, h)),
        ("arima",  lambda: _arima_component(pd.Series(train), h)),
    ]:
        try:
            pred = fn()
            maes[name] = float(np.mean(np.abs(pred - val)))
        except Exception as exc:
            log.debug("Component %s failed during validation: %s", name, exc)
            maes[name] = np.inf

    # Inverse-MAE weights; if all fail assign equal
    finite = {k: v for k, v in maes.items() if np.isfinite(v) and v > 0}
    if not finite:
        return {"linear": 1 / 3, "holt": 1 / 3, "arima": 1 / 3}

    inv = {k: 1.0 / v for k, v in finite.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    # Zero-weight failed components
    for k in maes:
        if k not in weights:
            weights[k] = 0.0
    return weights


# ── Public API ────────────────────────────────────────────────────────────────

def ensemble_forecast(
    series: pd.Series,
    horizon: int = 12,
    confidence: float = 0.90,
    val_frac: float = 0.20,
) -> pd.DataFrame:
    """
    Fit an ensemble of three forecasters and return a combined forecast.

    Parameters
    ----------
    series     : historical time-series values
    horizon    : number of future steps to forecast
    confidence : confidence level for empirical uncertainty bands
    val_frac   : fraction of data held out for weight learning

    Returns
    -------
    DataFrame with columns:
      step, forecast, lower_ci, upper_ci,
      weight_linear, weight_holt, weight_arima
    """
    values = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    arr    = values.to_numpy(dtype=float)

    if len(arr) < 3:
        # Not enough data – return flat forecast
        last = float(arr[-1]) if len(arr) else 0.0
        return pd.DataFrame({
            "step":          list(range(1, horizon + 1)),
            "forecast":      [round(last, 2)] * horizon,
            "lower_ci":      [round(last * 0.85, 2)] * horizon,
            "upper_ci":      [round(last * 1.15, 2)] * horizon,
            "weight_linear": [1.0] * horizon,
            "weight_holt":   [0.0] * horizon,
            "weight_arima":  [0.0] * horizon,
        })

    weights = _learn_weights(arr, horizon, val_frac)

    component_preds: Dict[str, np.ndarray] = {}
    for name, fn in [
        ("linear", lambda: _linear_forecast(arr, horizon)),
        ("holt",   lambda: _holt_forecast(arr, horizon)),
        ("arima",  lambda: _arima_component(values, horizon)),
    ]:
        try:
            component_preds[name] = fn()
        except Exception as exc:
            log.debug("Component %s failed at forecast time: %s", name, exc)
            component_preds[name] = np.full(horizon, float(arr[-1]))

    # Weighted combination
    ensemble = sum(
        weights.get(name, 0.0) * preds
        for name, preds in component_preds.items()
    )

    # Empirical uncertainty from component spread
    all_preds = np.stack(list(component_preds.values()))     # (3, horizon)
    z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96}.get(confidence, 1.645)
    spread = all_preds.std(axis=0)
    lower  = ensemble - z * spread
    upper  = ensemble + z * spread

    return pd.DataFrame({
        "step":          list(range(1, horizon + 1)),
        "forecast":      np.round(ensemble, 2).tolist(),
        "lower_ci":      np.round(lower, 2).tolist(),
        "upper_ci":      np.round(upper, 2).tolist(),
        "weight_linear": [round(weights.get("linear", 0), 3)] * horizon,
        "weight_holt":   [round(weights.get("holt",   0), 3)] * horizon,
        "weight_arima":  [round(weights.get("arima",  0), 3)] * horizon,
    })
