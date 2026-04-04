from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── API credentials ────────────────────────────────────────────────────
    cdc_app_token: str = os.getenv("CDC_APP_TOKEN", "")
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "dmarg-research/0.1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ── CDC Socrata endpoints ──────────────────────────────────────────────
    cdc_overdose_url: str = "https://data.cdc.gov/resource/xkb8-kh2a.json"
    cdc_county_overdose_url: str = "https://data.cdc.gov/resource/gb4e-yj24.json"
    cdc_specific_drugs_url: str = "https://data.cdc.gov/resource/8hzs-zshh.json"
    # Drug poisoning mortality – state-level, multiple years
    cdc_drug_poisoning_url: str = "https://data.cdc.gov/resource/jx5p-9qp5.json"

    # ── Census ACS endpoint ────────────────────────────────────────────────
    census_acs_url: str = "https://api.census.gov/data/2023/acs/acs5"

    # ── SAMHSA / NIDA – Treatment Locator & NSDUH indicators ──────────────
    samhsa_treatment_url: str = "https://findtreatment.samhsa.gov/locator/listing"
    # NSDUH state-level estimates via CDC Socrata mirror
    nsduh_url: str = "https://data.cdc.gov/resource/iuq5-y9ct.json"

    # ── Forecasting hyper-parameters ──────────────────────────────────────
    forecast_horizon: int = 12          # months ahead
    arima_max_p: int = 3
    arima_max_q: int = 3

    # ── Graph construction ────────────────────────────────────────────────
    graph_temporal_bins: int = 6        # number of time snapshots to keep

    # ── EWS thresholds ────────────────────────────────────────────────────
    ews_warning_threshold: float = 0.40
    ews_critical_threshold: float = 0.65


settings = Settings()
