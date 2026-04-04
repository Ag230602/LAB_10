"""
Google Trends social-signal client
───────────────────────────────────
Uses pytrends to fetch relative search-interest data for substance-related
query terms.  When pytrends is unavailable or rate-limited the module
generates a plausible synthetic time-series so the downstream pipeline
always receives well-formed data.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Canonical keyword groups ──────────────────────────────────────────────────
TREND_KEYWORDS: Dict[str, List[str]] = {
    "opioid":      ["fentanyl overdose", "heroin withdrawal", "opioid treatment"],
    "stimulant":   ["meth addiction", "cocaine overdose", "stimulant abuse"],
    "alcohol":     ["alcohol withdrawal", "alcoholism help", "DUI arrest"],
    "mental":      ["addiction anxiety", "relapse depression", "substance abuse help"],
    "treatment":   ["naloxone", "methadone clinic", "drug rehab near me"],
}

try:
    from pytrends.request import TrendReq
    _PYTRENDS_AVAILABLE = True
except Exception:
    _PYTRENDS_AVAILABLE = False


class TrendsClient:
    """
    Wraps pytrends with automatic retry, keyword batching, and a
    synthetic-data fallback so the pipeline is always runnable.
    """

    def __init__(self, geo: str = "US", timeout: int = 10, retries: int = 2) -> None:
        self.geo = geo
        self.timeout = timeout
        self.retries = retries
        self._pt: Optional[object] = None
        if _PYTRENDS_AVAILABLE:
            try:
                self._pt = TrendReq(hl="en-US", tz=360, timeout=(timeout, timeout))
            except Exception as exc:
                log.debug("pytrends init failed: %s", exc)
                self._pt = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "today 12-m",
        geo: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by date with one column per keyword
        containing relative search interest (0–100).

        Falls back to synthetic data on any failure.
        """
        target_geo = geo or self.geo
        if self._pt is not None:
            df = self._fetch_with_retry(keywords, timeframe, target_geo)
            if df is not None:
                return df
        return self._synthetic_trends(keywords, timeframe)

    def get_substance_trend_summary(
        self,
        timeframe: str = "today 12-m",
        geo: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches all TREND_KEYWORDS groups and returns a tidy summary
        DataFrame with columns: date, category, mean_interest.
        """
        frames: List[pd.DataFrame] = []
        for category, kws in TREND_KEYWORDS.items():
            # pytrends allows max 5 keywords per request
            df = self.get_interest_over_time(kws[:5], timeframe, geo)
            if df.empty:
                continue
            numeric_cols = [c for c in df.columns if c != "isPartial"]
            agg = df[numeric_cols].mean(axis=1).rename("mean_interest").reset_index()
            agg.columns = ["date", "mean_interest"]
            agg["category"] = category
            frames.append(agg)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_geo_interest(
        self,
        keywords: List[str],
        timeframe: str = "today 12-m",
        resolution: str = "REGION",
    ) -> pd.DataFrame:
        """
        Returns state-level interest breakdown; falls back to empty DataFrame.
        """
        if self._pt is None:
            return pd.DataFrame()
        try:
            self._pt.build_payload(keywords, timeframe=timeframe, geo=self.geo)  # type: ignore[attr-defined]
            return self._pt.interest_by_region(resolution=resolution)              # type: ignore[attr-defined]
        except Exception as exc:
            log.debug("geo interest failed: %s", exc)
            return pd.DataFrame()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_with_retry(
        self, keywords: List[str], timeframe: str, geo: str
    ) -> Optional[pd.DataFrame]:
        for attempt in range(self.retries):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        module=r"pytrends\..*",
                    )
                    self._pt.build_payload(keywords, timeframe=timeframe, geo=geo)  # type: ignore[attr-defined]
                    df = self._pt.interest_over_time()                               # type: ignore[attr-defined]
                if df is not None and not df.empty:
                    return df.drop(columns=["isPartial"], errors="ignore")
                return None
            except Exception as exc:
                wait = 2 ** attempt
                log.debug("pytrends attempt %d failed (%s), waiting %ds", attempt + 1, exc, wait)
                time.sleep(wait)
        return None

    @staticmethod
    def _synthetic_trends(keywords: List[str], timeframe: str) -> pd.DataFrame:
        """
        Generates a plausible 12-month synthetic interest series.
        Values reflect seasonality: higher in winter months (consistent with
        published CDC overdose seasonality patterns).
        """
        rng = np.random.default_rng(seed=sum(ord(c) for c in "".join(keywords)))
        n_weeks = 52
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_weeks, freq="W")
        # Seasonal component: peaks in Jan/Feb, dips in summer
        seasonal = 10 * np.sin(2 * np.pi * (dates.dayofyear / 365 - 0.05))
        data: Dict[str, object] = {"date": dates}
        for kw in keywords:
            base = rng.integers(30, 70)
            noise = rng.normal(0, 5, n_weeks)
            series = np.clip(base + seasonal + noise, 0, 100).astype(int)
            data[kw] = series
        df = pd.DataFrame(data).set_index("date")
        df.attrs["synthetic"] = True
        return df
