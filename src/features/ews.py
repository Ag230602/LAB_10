"""
Enhanced Early Warning Score (EWS)
────────────────────────────────────
Multi-dimensional scoring across five risk domains:
  1. Mortality      – CDC overdose death trajectory
  2. Substance use  – NIDA/SAMHSA prevalence signals
  3. Social         – Reddit / text composite risk
  4. Search trends  – Google Trends interest surge
  5. Socioeconomic  – Census poverty / income context

Each domain returns a 0–1 score; the aggregate EWS is a weighted sum.
Alert levels: LOW < 0.30 | MODERATE 0.30-0.50 | HIGH 0.50-0.70 | CRITICAL ≥ 0.70
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from src.config import settings


@dataclass
class EWSResult:
    ews: float                                          # 0–1 aggregate score
    alert_level: str                                    # LOW/MODERATE/HIGH/CRITICAL
    domain_scores: Dict[str, float] = field(default_factory=dict)
    trend_direction: str = "STABLE"                     # RISING / FALLING / STABLE
    delta_ews: float = 0.0                              # change vs prior period


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _alert_level(score: float) -> str:
    if score < settings.ews_warning_threshold:
        return "LOW"
    if score < settings.ews_critical_threshold:
        return "MODERATE" if score < 0.50 else "HIGH"
    return "CRITICAL"


def compute_ews(
    df: pd.DataFrame,
    # Legacy column names kept for backwards compatibility
    emotion_col: str = "distress_mentions",
    substance_col: str = "substance_mentions",
    trend_col: str = "trend_velocity",
    # New domain columns (all optional)
    overdose_rate_col: Optional[str] = None,
    opioid_misuse_col: Optional[str] = None,
    social_composite_col: Optional[str] = None,
    trends_interest_col: Optional[str] = None,
    poverty_rate_col: Optional[str] = None,
    # Domain weights
    w_mortality: float = 0.30,
    w_substance: float = 0.25,
    w_social: float = 0.20,
    w_trends: float = 0.10,
    w_socioeconomic: float = 0.15,
) -> pd.DataFrame:
    """
    Computes EWS for each row of *df* and appends result columns.

    Returned extra columns
    ----------------------
    ews                 – aggregate 0–1 score
    ews_alert_level     – categorical label
    ews_mortality       – mortality domain score
    ews_substance       – substance-use domain score
    ews_social          – social domain score
    ews_trends          – trends domain score
    ews_socioeconomic   – socioeconomic domain score
    """
    out = df.copy()

    def _num(col: Optional[str], default: float = 0.0) -> pd.Series:
        if col and col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(default)
        return pd.Series([default] * len(out), index=out.index)

    # ── Domain 1: Mortality (overdose trend) ─────────────────────────────────
    trend_v = _num(trend_col)
    # Normalise trend velocity: 0 → no change, 1 → very rapid increase
    trend_norm = (trend_v.clip(lower=0) / 15.0).clip(upper=1.0)
    rate_norm = (_num(overdose_rate_col) / 90.0).clip(0, 1)
    d_mortality = (0.6 * trend_norm + 0.4 * rate_norm).clip(0, 1)

    # ── Domain 2: Substance use (NIDA + legacy mentions) ─────────────────────
    sub_mentions = (_num(substance_col) / 5.0).clip(0, 1)
    opioid_pct   = (_num(opioid_misuse_col) / 9.0).clip(0, 1)
    d_substance  = (0.5 * sub_mentions + 0.5 * opioid_pct).clip(0, 1)

    # ── Domain 3: Social / text signals ──────────────────────────────────────
    distress_m   = (_num(emotion_col) / 5.0).clip(0, 1)
    social_comp  = _num(social_composite_col)
    d_social     = (0.4 * distress_m + 0.6 * social_comp).clip(0, 1)

    # ── Domain 4: Search trends ───────────────────────────────────────────────
    d_trends = (_num(trends_interest_col) / 100.0).clip(0, 1)

    # ── Domain 5: Socioeconomic ───────────────────────────────────────────────
    d_socioeconomic = (_num(poverty_rate_col) / 22.0).clip(0, 1)

    # ── Aggregate ────────────────────────────────────────────────────────────
    out["ews_mortality"]      = d_mortality.round(4)
    out["ews_substance"]      = d_substance.round(4)
    out["ews_social"]         = d_social.round(4)
    out["ews_trends"]         = d_trends.round(4)
    out["ews_socioeconomic"]  = d_socioeconomic.round(4)
    out["ews"] = (
        w_mortality    * d_mortality
        + w_substance  * d_substance
        + w_social     * d_social
        + w_trends     * d_trends
        + w_socioeconomic * d_socioeconomic
    ).clip(0, 1).round(4)
    out["ews_alert_level"] = out["ews"].apply(_alert_level)

    return out


def compute_ews_timeseries(
    ts_df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute rolling EWS from a simple time-series DataFrame.
    Adds columns: ews_score, ews_alert_level, ews_delta.
    """
    out = ts_df.copy().sort_values(date_col)
    vals = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    # Normalise to national reference range
    v_max = vals.max()
    norm = (vals / v_max).clip(0, 1) if v_max > 0 else vals
    # Rolling trend: 3-period slope
    rolling_slope = norm.diff(3).fillna(0.0) / 3.0
    trend_contrib = rolling_slope.clip(0).clip(upper=1.0)
    out["ews_score"] = ((0.7 * norm + 0.3 * trend_contrib)).clip(0, 1).round(4)
    out["ews_alert_level"] = out["ews_score"].apply(_alert_level)
    out["ews_delta"] = out["ews_score"].diff().fillna(0.0).round(4)
    return out
