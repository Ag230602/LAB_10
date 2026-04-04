"""Risk Narrator
────────────────
Generates an interpretable narrative report for a state's current risk.

Two modes:
  • Template mode (default): always available, deterministic.
  • LLM mode (optional): if OPENAI_API_KEY is set and openai package is
    installed (NOT required for this starter kit).

This file is intentionally safe: it does not call any remote service unless
explicitly configured by the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from src.config import settings


@dataclass
class Narrative:
    title: str
    summary: str
    bullets: list[str]
    caveats: list[str]


def generate_template_narrative(
    *,
    state: str,
    ews_row: Dict,
    fusion_row: Optional[Dict] = None,
    forecast_tail: Optional[pd.DataFrame] = None,
    anomalies: Optional[pd.DataFrame] = None,
) -> Narrative:
    ews = float(ews_row.get("ews", ews_row.get("fusion_score", 0.0)) or 0.0)
    alert = str(ews_row.get("ews_alert_level", ews_row.get("alert_level", "LOW")))

    bullets = []
    bullets.append(f"Overall alert level: {alert} (score={ews:.3f}).")

    for key in ["ews_mortality", "ews_substance", "ews_social", "ews_trends", "ews_socioeconomic"]:
        if key in ews_row:
            bullets.append(f"{key.replace('ews_', '').title()} domain: {float(ews_row[key]):.3f}.")

    if fusion_row:
        conf = float(fusion_row.get("confidence", 0.0) or 0.0)
        bullets.append(f"Data coverage confidence: {conf:.2f} (higher is better).")

    if forecast_tail is not None and not forecast_tail.empty and "forecast" in forecast_tail.columns:
        last_pred = float(pd.to_numeric(forecast_tail["forecast"], errors="coerce").dropna().iloc[-1])
        bullets.append(f"Near-term forecast indicates ~{last_pred:.1f} units in the last horizon step.")

    if anomalies is not None and not anomalies.empty and "is_anomaly" in anomalies.columns:
        n_spikes = int(anomalies["is_anomaly"].sum())
        if n_spikes > 0:
            bullets.append(f"Detected {n_spikes} anomaly spike(s) in the recent trend window.")

    # Optional semantic (embedding-style) social score
    if "semantic_social_composite" in ews_row:
        try:
            sem = float(ews_row.get("semantic_social_composite", 0.0) or 0.0)
            if sem > 0:
                bullets.append(f"Semantic social risk signal (embedding-style): {sem:.3f} (0–1).")
        except Exception:
            pass

    # Optional video-derived behavioral signals (non-identifying)
    vid_keys = [
        "video_activity_mean",
        "video_anomaly_score",
        "video_low_light_frac",
        "video_scene_change_rate",
    ]
    if any(k in ews_row for k in vid_keys):
        try:
            v_activity = float(ews_row.get("video_activity_mean", 0.0) or 0.0)
            v_anom = float(ews_row.get("video_anomaly_score", 0.0) or 0.0)
            v_dark = float(ews_row.get("video_low_light_frac", 0.0) or 0.0)
            v_scene = float(ews_row.get("video_scene_change_rate", 0.0) or 0.0)
            if max(v_activity, v_anom, v_dark, v_scene) > 0:
                bullets.append(
                    "Video-derived behavioral signals added "
                    f"(activity={v_activity:.3f}, anomaly={v_anom:.3f}, low_light={v_dark:.3f}, scene_change={v_scene:.3f})."
                )
        except Exception:
            pass

    summary = (
        f"{state} shows {alert.lower()} addiction/overdose risk based on multimodal signals "
        f"(public health, social, search trends, and socioeconomic context)."
    )

    caveats = [
        "EWS is a screening score, not a clinical diagnosis.",
        "Social and search signals may be noisy and rate-limited.",
        "API datasets are provisional; interpret trends with caution.",
    ]

    return Narrative(
        title=f"DMARG Risk Brief – {state}",
        summary=summary,
        bullets=bullets,
        caveats=caveats,
    )


def generate_llm_narrative(prompt: str) -> str:
    """Optional OpenAI call (only if user config enables it)."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set; use template narrative.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("openai package not installed; pip install openai") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""
