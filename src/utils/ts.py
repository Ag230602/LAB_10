from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def infer_monthly_series(
    df: pd.DataFrame,
    value_candidates: Iterable[str],
    year_col: str = "year",
    month_col: str = "month",
    date_col: Optional[str] = None,
    fallback_n_months: int = 24,
) -> pd.DataFrame:
    """Infer a monthly time series from a CDC-style DataFrame.

    Tries in order:
    1) (year, month) columns → build MonthStart date
    2) an explicit date column (ISO or parseable) → resample monthly
    3) fallback: create a synthetic monthly series from the row order

    Returns a DataFrame with columns: date, value
    """
    if df is None or df.empty:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=fallback_n_months, freq="MS")
        return pd.DataFrame({"date": dates, "value": np.zeros(len(dates))})

    value_col = next((c for c in value_candidates if c in df.columns), None)
    if value_col is None:
        # Create a 0 series
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=fallback_n_months, freq="MS")
        return pd.DataFrame({"date": dates, "value": np.zeros(len(dates))})

    values = pd.to_numeric(df[value_col], errors="coerce")

    # 1) year/month
    if year_col in df.columns and month_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce")
        months = pd.to_numeric(df[month_col], errors="coerce")
        ok = years.notna() & months.notna() & values.notna()
        if ok.any():
            dates = pd.to_datetime(
                {
                    "year": years[ok].astype(int),
                    "month": months[ok].astype(int),
                    "day": 1,
                },
                errors="coerce",
            )
            out = pd.DataFrame({"date": dates, "value": values[ok].astype(float)})
            out = out.dropna().groupby("date", as_index=False)["value"].sum().sort_values("date")
            return out

    # 2) date column
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        ok = d.notna() & values.notna()
        if ok.any():
            tmp = pd.DataFrame({"date": d[ok], "value": values[ok].astype(float)})
            tmp = tmp.set_index("date").sort_index()
            out = tmp.resample("MS").sum().reset_index()
            return out

    # 3) fallback
    # Use row order to create a plausible monthly series
    vals = values.dropna().astype(float)
    if vals.empty:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=fallback_n_months, freq="MS")
        return pd.DataFrame({"date": dates, "value": np.zeros(len(dates))})

    n = min(len(vals), fallback_n_months)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "value": vals.tail(n).to_numpy()})
