"""
ARIMA Forecasting Module
─────────────────────────
Provides auto-order ARIMA forecasting with confidence intervals
for overdose time-series data.

Strategy
────────
1. Try statsmodels ARIMA with grid-searched (p, d, q)
2. Fall back to simple exponential smoothing if ARIMA fails
3. Always return a standardised DataFrame with forecast + CI columns
"""

from __future__ import annotations

import itertools
import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    _STATSMODELS_AVAILABLE = True
except Exception:
    _STATSMODELS_AVAILABLE = False


# ── Public helpers ─────────────────────────────────────────────────────────────

def _is_stationary(series: pd.Series) -> bool:
    """ADF test at 5 % significance level."""
    try:
        p_val = adfuller(series.dropna())[1]
        return p_val < 0.05
    except Exception:
        return True   # assume stationary to be safe


def _auto_order(series: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
    """
    Minimal AIC-based grid search over (p, d, q).
    d ∈ {0, 1} based on ADF test; tries all (p, q) combos up to max_p × max_q.
    """
    d = 0 if _is_stationary(series) else 1
    best_aic = np.inf
    best_order = (1, d, 1)

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order


def arima_forecast(
    series: pd.Series,
    horizon: int = 12,
    confidence: float = 0.95,
    max_p: int = 3,
    max_q: int = 3,
) -> pd.DataFrame:
    """
    Fits ARIMA to *series* and returns a *horizon*-step forecast.

    Parameters
    ----------
    series     : univariate time-series (float values)
    horizon    : number of periods to forecast
    confidence : confidence level for interval (default 0.95)
    max_p      : maximum AR order to search
    max_q      : maximum MA order to search

    Returns
    -------
    DataFrame with columns: step, forecast, lower_ci, upper_ci, method
    """
    values = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)

    if len(values) < 4:
        return _fallback_forecast(values, horizon)

    if _STATSMODELS_AVAILABLE:
        try:
            order = _auto_order(values, max_p=max_p, max_q=max_q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(values, order=order).fit()
            pred = model.get_forecast(steps=horizon)
            mean = pred.predicted_mean.reset_index(drop=True)
            ci   = pred.conf_int(alpha=1 - confidence).reset_index(drop=True)
            return pd.DataFrame({
                "step":     list(range(1, horizon + 1)),
                "forecast": mean.round(2).tolist(),
                "lower_ci": ci.iloc[:, 0].round(2).tolist(),
                "upper_ci": ci.iloc[:, 1].round(2).tolist(),
                "method":   f"ARIMA{order}",
                "arima_order": str(order),
            })
        except Exception as exc:
            log.debug("ARIMA failed (%s), falling back to ETS.", exc)

    return _fallback_forecast(values, horizon)


def _fallback_forecast(values: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Simple exponential smoothing fallback (Holt's double exponential).
    No external library required.
    """
    alpha = 0.3
    beta  = 0.1
    n = len(values)
    if n == 0:
        return pd.DataFrame({
            "step": list(range(1, horizon + 1)),
            "forecast": [0.0] * horizon,
            "lower_ci": [0.0] * horizon,
            "upper_ci": [0.0] * horizon,
            "method": "fallback_zero",
            "arima_order": "N/A",
        })

    if n == 1:
        last = float(values.iloc[0])
        return pd.DataFrame({
            "step": list(range(1, horizon + 1)),
            "forecast": [round(last, 2)] * horizon,
            "lower_ci": [round(last * 0.8, 2)] * horizon,
            "upper_ci": [round(last * 1.2, 2)] * horizon,
            "method": "constant",
            "arima_order": "N/A",
        })

    # Holt double exponential smoothing
    level = float(values.iloc[0])
    trend = float(values.iloc[1] - values.iloc[0])
    for v in values.iloc[1:]:
        prev_level = level
        level = alpha * float(v) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    residuals = float(values.std()) if len(values) > 1 else abs(trend)
    z = 1.96 if False else {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(0.95, 1.96)

    forecasts, lowers, uppers = [], [], []
    for h in range(1, horizon + 1):
        f = level + h * trend
        margin = z * residuals * np.sqrt(h)
        forecasts.append(round(f, 2))
        lowers.append(round(f - margin, 2))
        uppers.append(round(f + margin, 2))

    return pd.DataFrame({
        "step":        list(range(1, horizon + 1)),
        "forecast":    forecasts,
        "lower_ci":    lowers,
        "upper_ci":    uppers,
        "method":      "Holt_ETS",
        "arima_order": "N/A",
    })


def forecast_with_dates(
    series: pd.Series,
    start_date: Optional[pd.Timestamp] = None,
    freq: str = "MS",
    horizon: int = 12,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper that attaches calendar dates to the forecast output.

    Parameters
    ----------
    series     : float time-series
    start_date : date of the last observation (default: today)
    freq       : pandas offset string for forecast periods (default: "MS" = month start)
    horizon    : periods to forecast
    """
    df = arima_forecast(series, horizon=horizon, **kwargs)
    last = start_date or pd.Timestamp.today()
    future_dates = pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]
    df.insert(0, "date", future_dates.tolist())
    return df
