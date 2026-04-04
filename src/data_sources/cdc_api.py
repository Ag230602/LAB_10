from __future__ import annotations

from typing import Any, Dict, Optional
import requests
import pandas as pd

from src.config import settings


class CDCAPIClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        if settings.cdc_app_token:
            self.session.headers.update({"X-App-Token": settings.cdc_app_token})

    def fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> list[dict]:
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_national_overdose(self, limit: int = 5000) -> pd.DataFrame:
        params = {
            "$limit": limit,
            "$order": "year desc, month desc",
        }
        data = self.fetch_json(settings.cdc_overdose_url, params=params)
        return pd.DataFrame(data)

    def get_county_overdose(self, state_abbr: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "$limit": limit,
        }
        if state_abbr:
            # County-level dataset uses 'st_abbrev' (not 'state')
            params["st_abbrev"] = state_abbr.upper()
        data = self.fetch_json(settings.cdc_county_overdose_url, params=params)
        return pd.DataFrame(data)

    def get_specific_drug_counts(self, limit: int = 5000) -> pd.DataFrame:
        params = {
            "$limit": limit,
            "$order": "year desc, month desc",
        }
        data = self.fetch_json(settings.cdc_specific_drugs_url, params=params)
        return pd.DataFrame(data)
