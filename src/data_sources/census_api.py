from __future__ import annotations

from typing import Optional
import requests
import pandas as pd

from src.config import settings


class CensusAPIClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def get_state_context(self, state_fips: str) -> pd.DataFrame:
        """Richer ACS context for a state (single row).

        Includes:
        - population (B01003_001E)
        - median household income (B19013_001E)
        - poverty count/total + computed poverty_rate_pct (B17001_002E, B17001_001E)
        - unemployed + in labor force + computed unemployment_rate_pct (B23025_005E, B23025_002E)
        """
        params = {
            "get": "NAME,B01003_001E,B19013_001E,B17001_002E,B17001_001E,B23025_005E,B23025_002E",
            "for": f"state:{state_fips}",
        }
        if settings.census_api_key:
            params["key"] = settings.census_api_key
        response = self.session.get(settings.census_acs_url, params=params, timeout=60)
        response.raise_for_status()
        rows = response.json()
        df = pd.DataFrame(rows[1:], columns=rows[0])

        # Compute rates
        if not df.empty:
            pop = pd.to_numeric(df.get("B01003_001E"), errors="coerce")
            pov_num = pd.to_numeric(df.get("B17001_002E"), errors="coerce")
            pov_den = pd.to_numeric(df.get("B17001_001E"), errors="coerce")
            lf = pd.to_numeric(df.get("B23025_002E"), errors="coerce")
            unemp = pd.to_numeric(df.get("B23025_005E"), errors="coerce")

            df["poverty_rate_pct"] = ((pov_num / pov_den) * 100.0).where(pov_den > 0)
            df["unemployment_rate_pct"] = ((unemp / lf) * 100.0).where(lf > 0)
            df["population"] = pop
        return df

    def get_state_population(self, state_fips: str) -> pd.DataFrame:
        params = {
            "get": "NAME,B01003_001E,B19013_001E",
            "for": f"state:{state_fips}",
        }
        if settings.census_api_key:
            params["key"] = settings.census_api_key
        response = self.session.get(settings.census_acs_url, params=params, timeout=60)
        response.raise_for_status()
        rows = response.json()
        return pd.DataFrame(rows[1:], columns=rows[0])

    def get_all_states_population(self) -> pd.DataFrame:
        params = {
            "get": "NAME,B01003_001E,B19013_001E",
            "for": "state:*",
        }
        if settings.census_api_key:
            params["key"] = settings.census_api_key
        response = self.session.get(settings.census_acs_url, params=params, timeout=60)
        response.raise_for_status()
        rows = response.json()
        return pd.DataFrame(rows[1:], columns=rows[0])
