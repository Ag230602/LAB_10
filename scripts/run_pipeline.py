from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.data_sources.cdc_api import CDCAPIClient
from src.data_sources.census_api import CensusAPIClient
from src.data_sources.reddit_api import RedditAPIClient
from src.data_sources.nida_api import NIDASAMHSAClient
from src.data_sources.trends_api import TrendsClient
from src.features.sentiment import analyze_dataframe, aggregate_signal
from src.features.fusion import build_signal_dict, fuse, multi_state_fusion_table
from src.features.ews import compute_ews
from src.models.ensemble import ensemble_forecast
from src.models.anomaly import detect_anomalies
from src.models.policy_sim import compare_interventions, run_simulation
from src.graph.build_graph import build_risk_graph, graph_to_edge_frame, graph_to_node_frame, multi_state_graph
from src.graph.temporal_graph import TemporalGraph
from src.llm.risk_narrator import generate_template_narrative
from src.utils.ts import infer_monthly_series

STATE_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="KS", help="Single state abbreviation (legacy)")
    parser.add_argument("--states", type=str, default="", help="Comma-separated list of state abbreviations")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--use-reddit", type=str, default="false")
    parser.add_argument("--use-trends", type=str, default="false")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--snapshots", type=int, default=6, help="Number of temporal graph snapshots")
    parser.add_argument("--simulate-policy", type=str, default="false")
    parser.add_argument("--intervention", type=str, default="naloxone_distribution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_abbr = args.state.upper()
    use_reddit = args.use_reddit.lower() == "true"
    use_trends = args.use_trends.lower() == "true"
    simulate_policy = args.simulate_policy.lower() == "true"

    states: List[str]
    if args.states.strip():
        states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    else:
        states = [state_abbr]

    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    cdc = CDCAPIClient()
    census = CensusAPIClient()
    reddit = RedditAPIClient()
    nida = NIDASAMHSAClient()
    trends = TrendsClient(geo="US")

    state_graphs: Dict[str, object] = {}
    fusion_inputs: Dict[str, Dict[str, Optional[float]]] = {}

    for state in states:
        print(f"Processing {state}...")

        # ── CDC ───────────────────────────────────────────────────────────
        county_df = cdc.get_county_overdose(state_abbr=state, limit=5000)
        county_df.to_csv(out_dir / f"cdc_county_{state}.csv", index=False)

        # Derive a monthly series if possible
        overdose_ts = infer_monthly_series(
            county_df,
            value_candidates=[
                "provisional_drug_overdose",
                "drug_overdose_deaths",
                "deaths",
                "value",
            ],
            year_col="year",
            month_col="month",
            date_col="date",
            fallback_n_months=max(args.horizon * 2, 24),
        )
        overdose_ts.to_csv(out_dir / f"overdose_ts_{state}.csv", index=False)

        latest_overdose = float(pd.to_numeric(overdose_ts["value"], errors="coerce").dropna().tail(1).iloc[0]) if not overdose_ts.empty else 0.0
        trend_velocity = float(pd.to_numeric(overdose_ts["value"], errors="coerce").diff().dropna().tail(6).mean()) if len(overdose_ts) > 1 else 0.0

        # ── Census ────────────────────────────────────────────────────────
        state_fips = STATE_TO_FIPS.get(state, "20")
        census_ctx = census.get_state_context(state_fips=state_fips)
        census_ctx.to_csv(out_dir / f"census_context_{state}.csv", index=False)

        # Keep legacy file name for the dashboard
        census_df = census.get_state_population(state_fips=state_fips)
        census_df.to_csv(out_dir / f"census_{state}.csv", index=False)

        median_income = None
        if not census_df.empty and "B19013_001E" in census_df.columns:
            median_income = pd.to_numeric(census_df["B19013_001E"], errors="coerce").fillna(0.0).iloc[0]

        population = None
        if not census_ctx.empty and "population" in census_ctx.columns:
            pop_val = pd.to_numeric(census_ctx["population"], errors="coerce").fillna(0.0).iloc[0]
            population = float(pop_val) if pop_val > 0 else None

        poverty_rate = None
        if not census_ctx.empty and "poverty_rate_pct" in census_ctx.columns:
            pr = pd.to_numeric(census_ctx["poverty_rate_pct"], errors="coerce").fillna(0.0).iloc[0]
            poverty_rate = float(pr)

        # ── NIDA/SAMHSA ───────────────────────────────────────────────────
        nida_df = nida.get_state_drug_stats(state)
        nida_df.to_csv(out_dir / f"nida_{state}.csv", index=False)
        nida_row = nida_df.iloc[0].to_dict() if not nida_df.empty else {}

        # ── Reddit / social signals ───────────────────────────────────────
        reddit_df = pd.DataFrame(columns=["title", "selftext"])  # default empty
        social_summary: Dict[str, float] = {}
        if use_reddit:
            reddit_df = reddit.search_posts(
                query="opioid OR fentanyl OR relapse OR overdose OR withdrawal OR naloxone",
                subreddit="all",
                limit=75,
            )
            if not reddit_df.empty:
                reddit_df["text"] = reddit_df["title"].fillna("") + " " + reddit_df["selftext"].fillna("")
                reddit_df = analyze_dataframe(reddit_df, text_col="text")
                reddit_df.to_csv(out_dir / f"reddit_{state}.csv", index=False)
                social_summary = aggregate_signal(reddit_df)

        # ── Google Trends ────────────────────────────────────────────────
        trends_df = pd.DataFrame()
        trend_summary: Dict[str, float] = {}
        if use_trends:
            trends_df = trends.get_substance_trend_summary(timeframe="today 12-m", geo="US")
            trends_df.to_csv(out_dir / f"trends_{state}.csv", index=False)
            if not trends_df.empty:
                for cat in ["opioid", "stimulant", "treatment"]:
                    sub = trends_df[trends_df["category"] == cat]
                    trend_summary[f"trends_{cat}_mean"] = float(pd.to_numeric(sub["mean_interest"], errors="coerce").mean()) if not sub.empty else 0.0

        # ── Fusion & EWS ─────────────────────────────────────────────────
        sig_dict = build_signal_dict(
            cdc_df=county_df,
            nida_df=nida_df,
            census_df=census_ctx,
            reddit_df=reddit_df,
            trends_df=trends_df,
        )
        # add explicitly computed signals
        sig_dict["trend_velocity"] = trend_velocity
        fusion_inputs[state] = sig_dict

        fusion_result = fuse(sig_dict)
        fusion_df = pd.DataFrame([
            {
                "state": state,
                "fusion_score": fusion_result.fusion_score,
                "alert_level": fusion_result.alert_level,
                "confidence": fusion_result.confidence,
                **{f"domain_{k}": v for k, v in fusion_result.domain_scores.items()},
            }
        ])
        fusion_df.to_csv(out_dir / f"fusion_{state}.csv", index=False)

        risk_row = {
            "state": state,
            "trend_velocity": trend_velocity,
            "distress_mentions": float(social_summary.get("distress_score", 0.0) * 5.0),
            "substance_mentions": float(social_summary.get("substance_score", 0.0) * 5.0),
            "opioid_misuse_pct": float(nida_row.get("opioid_misuse_pct", 0.0)),
            "social_composite": float(social_summary.get("composite_risk", 0.0)),
            "trends_interest": float(trend_summary.get("trends_opioid_mean", 0.0)),
            "poverty_rate_pct": poverty_rate if poverty_rate is not None else 0.0,
            "overdose_rate_per_100k": float(latest_overdose),
        }
        risk_df = pd.DataFrame([risk_row])
        risk_df = compute_ews(
            risk_df,
            overdose_rate_col="overdose_rate_per_100k",
            opioid_misuse_col="opioid_misuse_pct",
            social_composite_col="social_composite",
            trends_interest_col="trends_interest",
            poverty_rate_col="poverty_rate_pct",
        )
        risk_df.to_csv(out_dir / f"risk_{state}.csv", index=False)

        # ── Forecast + anomaly detection ──────────────────────────────────
        series = pd.to_numeric(overdose_ts["value"], errors="coerce").fillna(0.0)
        forecast_df = ensemble_forecast(series.tail(60), horizon=args.horizon)
        forecast_df.to_csv(out_dir / f"forecast_{state}.csv", index=False)

        # Anomalies on recent history
        recent = overdose_ts.tail(max(24, args.horizon * 2)).copy()
        recent["value"] = pd.to_numeric(recent["value"], errors="coerce").fillna(0.0)
        anomalies_df = detect_anomalies(recent, value_col="value")
        anomalies_df.to_csv(out_dir / f"anomalies_{state}.csv", index=False)

        # ── Graph (latest snapshot) ───────────────────────────────────────
        g = build_risk_graph(
            state_name=state,
            overdose_value=latest_overdose,
            ews_value=float(risk_df["ews"].iloc[0]),
            median_income=float(median_income) if median_income is not None else None,
            nida_row=nida_row,
            social_signals=social_summary if social_summary else None,
            trend_signals=trend_summary if trend_summary else None,
            timestamp=datetime.now(timezone.utc),
        )
        node_df = graph_to_node_frame(g)
        edge_df = graph_to_edge_frame(g)
        node_df.to_csv(out_dir / f"graph_nodes_{state}.csv", index=False)
        edge_df.to_csv(out_dir / f"graph_edges_{state}.csv", index=False)

        # ── Temporal snapshots ────────────────────────────────────────────
        tg = TemporalGraph()
        snap_n = max(2, int(args.snapshots))
        ts_slice = overdose_ts.tail(snap_n).copy()
        if not ts_slice.empty:
            # Normalise simple EWS over time using the observed series (0–1)
            vmax = float(pd.to_numeric(ts_slice["value"], errors="coerce").max() or 0.0)
            for _, row in ts_slice.iterrows():
                dt = pd.to_datetime(row["date"], errors="coerce")
                if pd.isna(dt):
                    continue
                val = float(pd.to_numeric(row["value"], errors="coerce") or 0.0)
                ews_t = float(val / vmax) if vmax > 0 else 0.0
                g_t = build_risk_graph(
                    state_name=state,
                    overdose_value=val,
                    ews_value=ews_t,
                    median_income=float(median_income) if median_income is not None else None,
                    nida_row=nida_row,
                    social_signals=social_summary if social_summary else None,
                    trend_signals=trend_summary if trend_summary else None,
                    timestamp=dt.to_pydatetime(),
                )
                tg.add_snapshot(dt.to_pydatetime(), g_t, metadata={"state": state})

        tg.get_evolution_summary().to_csv(out_dir / f"temporal_summary_{state}.csv", index=False)
        tg.compute_edge_weight_changes().to_csv(out_dir / f"temporal_edges_{state}.csv", index=False)

        # ── Policy simulation ─────────────────────────────────────────────
        if simulate_policy:
            base_rate = float(latest_overdose)
            pop = population if population is not None else 1_000_000
            sim = run_simulation(base_rate, args.intervention, horizon=24, population=float(pop))
            # Save baseline + mean trajectory
            merged = sim.baseline.merge(sim.scenario_mean, on="month", suffixes=("_baseline", "_scenario"))
            merged.to_csv(out_dir / f"policy_{state}_{args.intervention}.csv", index=False)
            compare_interventions(base_rate, horizon=24, population=float(pop)).to_csv(
                out_dir / f"policy_compare_{state}.csv", index=False
            )

        # ── Narrative ─────────────────────────────────────────────────────
        narrative = generate_template_narrative(
            state=state,
            ews_row=risk_df.iloc[0].to_dict(),
            fusion_row=fusion_df.iloc[0].to_dict(),
            forecast_tail=forecast_df.tail(4),
            anomalies=anomalies_df,
        )
        with open(out_dir / f"narrative_{state}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "title": narrative.title,
                    "summary": narrative.summary,
                    "bullets": narrative.bullets,
                    "caveats": narrative.caveats,
                },
                f,
                indent=2,
            )

        state_graphs[state] = g

    # ── Multi-state exports ─────────────────────────────────────────────────
    if len(states) > 1:
        fusion_table = multi_state_fusion_table(fusion_inputs)
        fusion_table.to_csv(out_dir / "fusion_all_states.csv", index=False)
        nat_graph = multi_state_graph({s: state_graphs[s] for s in states})  # type: ignore[arg-type]
        graph_to_node_frame(nat_graph).to_csv(out_dir / "graph_nodes_national.csv", index=False)
        graph_to_edge_frame(nat_graph).to_csv(out_dir / "graph_edges_national.csv", index=False)

    print("Pipeline completed.")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
