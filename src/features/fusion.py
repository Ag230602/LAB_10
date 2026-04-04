"""
Multimodal Data Fusion Engine
──────────────────────────────
Normalises signals from heterogeneous sources (CDC, Census, Reddit,
Google Trends, NIDA/SAMHSA) onto a common [0, 1] scale and produces:
  • A fused risk vector
  • Per-source confidence scores (proxy for data completeness)
  • A single aggregate fusion score

Design
──────
Signal normalization uses percentile-based min–max scaling anchored to
national reference ranges so a state's score is interpretable relative to
all US states rather than to its own historical variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Reference ranges (approximate national min/max for each raw signal) ──────
# Source: CDC WISQARS, SAMHSA NSDUH 2022, Census ACS 2022
REFERENCE_RANGES: Dict[str, Tuple[float, float]] = {
    # CDC signals
    "overdose_rate_per_100k":      (2.0,   90.0),
    "trend_velocity":              (-5.0,  15.0),
    "drug_overdose_deaths_mean":   (0.0,   500.0),
    # NIDA/SAMHSA signals
    "illicit_drug_use_pct":        (10.0,  25.0),
    "opioid_misuse_pct":           (2.0,   9.0),
    "meth_use_pct":                (0.3,   3.5),
    "alcohol_disorder_pct":        (4.0,   10.0),
    "treatment_need_pct":          (6.0,   13.0),
    # Census / demographic signals
    "poverty_rate_pct":            (5.0,   22.0),
    "unemployment_rate_pct":       (2.0,   12.0),
    "median_income_inv":           (0.0,   1.0),   # already inverted & normalised
    # Social / Reddit signals
    "substance_score":             (0.0,   0.5),
    "distress_score":              (0.0,   0.5),
    "urgency_score":               (0.0,   0.3),
    "composite_risk":              (0.0,   1.0),
    # Semantic / embedding-style text signals
    "semantic_substance_score":     (0.0,   1.0),
    "semantic_distress_score":      (0.0,   1.0),
    "semantic_help_seeking_score":  (0.0,   1.0),
    "semantic_composite_risk":      (0.0,   1.0),
    # Video behavioral signals (optional)
    "video_activity_mean":          (0.0,   1.0),
    "video_anomaly_score":          (0.0,   1.0),
    "video_low_light_frac":         (0.0,   1.0),
    "video_scene_change_rate":      (0.0,   2.0),
    # Google Trends signals
    "trends_opioid_mean":          (0.0,  100.0),
    "trends_stimulant_mean":       (0.0,  100.0),
    "trends_treatment_mean":       (0.0,  100.0),
}

# ── Domain weights ────────────────────────────────────────────────────────────
DOMAIN_WEIGHTS: Dict[str, float] = {
    "cdc":     0.35,
    "nida":    0.25,
    "social":  0.20,
    "video":   0.10,
    "trends":  0.10,
    "census":  0.10,
}

# ── Per-signal domain assignment ──────────────────────────────────────────────
SIGNAL_DOMAIN: Dict[str, str] = {
    "overdose_rate_per_100k":    "cdc",
    "trend_velocity":            "cdc",
    "drug_overdose_deaths_mean": "cdc",
    "illicit_drug_use_pct":      "nida",
    "opioid_misuse_pct":         "nida",
    "meth_use_pct":              "nida",
    "alcohol_disorder_pct":      "nida",
    "treatment_need_pct":        "nida",
    "poverty_rate_pct":          "census",
    "unemployment_rate_pct":     "census",
    "median_income_inv":         "census",
    "substance_score":           "social",
    "distress_score":            "social",
    "urgency_score":             "social",
    "composite_risk":            "social",
    "semantic_substance_score":   "social",
    "semantic_distress_score":    "social",
    "semantic_help_seeking_score": "social",
    "semantic_composite_risk":    "social",
    "video_activity_mean":        "video",
    "video_anomaly_score":        "video",
    "video_low_light_frac":       "video",
    "video_scene_change_rate":    "video",
    "trends_opioid_mean":        "trends",
    "trends_stimulant_mean":     "trends",
    "trends_treatment_mean":     "trends",
}


@dataclass
class FusionResult:
    """Container for fusion output."""
    fusion_score: float                          # 0–1 aggregate risk
    domain_scores: Dict[str, float] = field(default_factory=dict)   # per-domain
    normalized_signals: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0                      # proportion of signals present
    n_signals: int = 0
    alert_level: str = "LOW"                     # LOW / MODERATE / HIGH / CRITICAL


def _normalize_signal(name: str, value: float) -> float:
    """Clip then min-max scale a raw signal value using reference ranges."""
    lo, hi = REFERENCE_RANGES.get(name, (0.0, 1.0))
    if hi == lo:
        return 0.0
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo)


def _alert_level(score: float) -> str:
    if score < 0.30:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.70:
        return "HIGH"
    return "CRITICAL"


def fuse(signals: Dict[str, Optional[float]]) -> FusionResult:
    """
    Fuse a dictionary of raw signal values into a single risk score.

    Parameters
    ----------
    signals : dict of {signal_name: raw_value | None}
        None values are treated as missing and excluded from the fusion
        with a confidence penalty.

    Returns
    -------
    FusionResult
    """
    normed: Dict[str, float] = {}
    domain_buckets: Dict[str, List[float]] = {d: [] for d in DOMAIN_WEIGHTS}

    total_expected = len(SIGNAL_DOMAIN)
    present = 0

    for sig_name, domain in SIGNAL_DOMAIN.items():
        raw = signals.get(sig_name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        n = _normalize_signal(sig_name, float(raw))
        normed[sig_name] = n
        domain_buckets[domain].append(n)
        present += 1

    # Domain averages
    domain_scores: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_used = 0.0
    for domain, vals in domain_buckets.items():
        if vals:
            ds = float(np.mean(vals))
            domain_scores[domain] = ds
            weighted_sum += DOMAIN_WEIGHTS[domain] * ds
            weight_used += DOMAIN_WEIGHTS[domain]

    fusion_score = (weighted_sum / weight_used) if weight_used > 0 else 0.0
    confidence = present / max(total_expected, 1)

    return FusionResult(
        fusion_score=round(fusion_score, 4),
        domain_scores={k: round(v, 4) for k, v in domain_scores.items()},
        normalized_signals={k: round(v, 4) for k, v in normed.items()},
        confidence=round(confidence, 3),
        n_signals=present,
        alert_level=_alert_level(fusion_score),
    )


def build_signal_dict(
    *,
    cdc_df: Optional[pd.DataFrame] = None,
    nida_df: Optional[pd.DataFrame] = None,
    census_df: Optional[pd.DataFrame] = None,
    reddit_df: Optional[pd.DataFrame] = None,
    trends_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[float]]:
    """
    Convenience helper that extracts scalar signal values from the
    DataFrames produced by each data-source client and returns a signals
    dict ready for *fuse()*.
    """
    sigs: Dict[str, Optional[float]] = {}

    # CDC signals
    if cdc_df is not None and not cdc_df.empty:
        death_col = next((c for c in cdc_df.columns if "death" in c.lower()), None)
        if death_col is None and "provisional_drug_overdose" in cdc_df.columns:
            death_col = "provisional_drug_overdose"
        if death_col:
            vals = pd.to_numeric(cdc_df[death_col], errors="coerce").dropna()
            if not vals.empty:
                sigs["drug_overdose_deaths_mean"] = float(vals.mean())
                diffs = vals.diff().dropna()
                sigs["trend_velocity"] = float(diffs.mean()) if not diffs.empty else 0.0

    # NIDA / SAMHSA signals
    if nida_df is not None and not nida_df.empty:
        for col in ["illicit_drug_use_pct", "opioid_misuse_pct", "meth_use_pct",
                    "alcohol_disorder_pct", "treatment_need_pct"]:
            if col in nida_df.columns:
                sigs[col] = float(
                    pd.to_numeric(nida_df[col], errors="coerce").iloc[0]
                )

    # Census signals
    if census_df is not None and not census_df.empty:
        if "B19013_001E" in census_df.columns:
            inc = pd.to_numeric(census_df["B19013_001E"], errors="coerce").iloc[0]
            if not np.isnan(inc) and inc > 0:
                # Invert: lower income → higher risk; normalise against US range
                sigs["median_income_inv"] = float(1.0 - np.clip((inc - 30_000) / 90_000, 0, 1))

    # Social (Reddit) signals
    if reddit_df is not None and not reddit_df.empty:
        for col in ["substance_score", "distress_score", "urgency_score", "composite_risk"]:
            if col in reddit_df.columns:
                sigs[col] = float(
                    pd.to_numeric(reddit_df[col], errors="coerce").mean()
                )
        for col in [
            "semantic_substance_score",
            "semantic_distress_score",
            "semantic_help_seeking_score",
            "semantic_composite_risk",
        ]:
            if col in reddit_df.columns:
                sigs[col] = float(pd.to_numeric(reddit_df[col], errors="coerce").mean())

    # Google Trends signals
    if trends_df is not None and not trends_df.empty:
        if "category" in trends_df.columns and "mean_interest" in trends_df.columns:
            for cat in ["opioid", "stimulant", "treatment"]:
                sub = trends_df[trends_df["category"] == cat]["mean_interest"]
                if not sub.empty:
                    sigs[f"trends_{cat}_mean"] = float(sub.mean())

    return sigs


def multi_state_fusion_table(
    state_signals: Dict[str, Dict[str, Optional[float]]],
) -> pd.DataFrame:
    """
    Accepts a mapping of {state_abbr: signals_dict} and returns a
    ranked DataFrame with fusion scores for all states.
    """
    rows = []
    for state, sigs in state_signals.items():
        result = fuse(sigs)
        rows.append({
            "state":        state,
            "fusion_score": result.fusion_score,
            "alert_level":  result.alert_level,
            "confidence":   result.confidence,
            **{f"domain_{k}": v for k, v in result.domain_scores.items()},
        })
    df = pd.DataFrame(rows).sort_values("fusion_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)
