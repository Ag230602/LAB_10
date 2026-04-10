from __future__ import annotations

"""Pipeline CLI entrypoint.

This is the primary runnable for generating all outputs under `data/cache/`.
Kept in `apps/` as the user-facing application layer.

Backwards compatible wrapper exists at `scripts/run_pipeline.py`.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data_sources.cdc_api import CDCAPIClient
from src.data_sources.census_api import CensusAPIClient
from src.data_sources.reddit_api import RedditAPIClient
from src.data_sources.nida_api import NIDASAMHSAClient
from src.data_sources.trends_api import TrendsClient
from src.features.sentiment import analyze_dataframe, aggregate_signal
from src.features.semantic_signals import score_semantic_similarity, aggregate_semantic_signal
from src.features.fusion import build_signal_dict, fuse, multi_state_fusion_table
from src.features.ews import compute_ews
from src.models.ensemble import ensemble_forecast
from src.models.anomaly import detect_anomalies
from src.models.policy_sim import compare_interventions, run_simulation
from src.graph.build_graph import (
    build_risk_graph,
    graph_to_edge_frame,
    graph_to_node_frame,
    multi_state_graph,
)
from src.graph.temporal_graph import TemporalGraph
from src.llm.risk_narrator import generate_template_narrative
from src.utils.ts import infer_monthly_series
from src.video.video_signals import extract_multi_video_signals


def _narrative_to_markdown(title: str, summary: str, bullets: List[str], caveats: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"# {title}".strip())
    lines.append("")
    if summary:
        lines.append(summary.strip())
        lines.append("")
    if bullets:
        lines.append("## Key points")
        for b in bullets:
            b = str(b).strip()
            if b:
                lines.append(f"- {b}")
        lines.append("")
    if caveats:
        lines.append("## Caveats")
        for c in caveats:
            c = str(c).strip()
            if c:
                lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

STATE_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="KS", help="Single state abbreviation (legacy)")
    parser.add_argument("--states", type=str, default="", help="Comma-separated list of state abbreviations")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--use-reddit", type=str, default="false")
    parser.add_argument("--use-trends", type=str, default="false")
    parser.add_argument(
        "--use-youtube",
        type=str,
        default="false",
        help="If true, extracts text-based signals from YouTube transcripts (no video download).",
    )
    parser.add_argument(
        "--youtube-urls",
        type=str,
        default="",
        help="Comma-separated list of YouTube video URLs to analyze via transcript.",
    )
    parser.add_argument(
        "--youtube-languages",
        type=str,
        default="en",
        help="Comma-separated language codes to try for transcripts (e.g., en,en-US).",
    )
    parser.add_argument(
        "--youtube-max-segments",
        type=int,
        default=400,
        help="Max transcript segments to score per YouTube video.",
    )
    parser.add_argument(
        "--use-video",
        type=str,
        default="false",
        help="If true, extracts non-identifying behavioral signals from video(s).",
    )
    parser.add_argument(
        "--video-paths",
        type=str,
        default="",
        help="Comma-separated list of local video file paths (mp4, mov, etc).",
    )
    parser.add_argument(
        "--allow-remote-video",
        type=str,
        default="false",
        help=(
            "If true, allows http(s) URLs in --video-paths and downloads them before processing. "
            "Only direct file URLs are supported (e.g., https://.../clip.mp4)."
        ),
    )
    parser.add_argument(
        "--remote-video-max-mb",
        type=int,
        default=500,
        help="Maximum size (MB) allowed for a remote video download.",
    )
    parser.add_argument(
        "--save-reddit-raw",
        type=str,
        default="false",
        help="If true, exports raw Reddit title/selftext. Default false for privacy.",
    )
    parser.add_argument(
        "--reddit-include-post-id",
        type=str,
        default="false",
        help="If true, keeps Reddit post IDs in memory/exports. Default false for privacy.",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--snapshots", type=int, default=6, help="Number of temporal graph snapshots")
    parser.add_argument("--simulate-policy", type=str, default="false")
    parser.add_argument("--intervention", type=str, default="naloxone_distribution")
    return parser.parse_args()


def _is_http_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in {"http", "https"} and bool(u.netloc)
    except Exception:
        return False


def _safe_filename_from_url(url: str) -> str:
    u = urlparse(url)
    name = Path(u.path).name or "remote_video"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if len(name) > 120:
        name = name[:120]
    return name


def _download_remote_video(
    url: str,
    *,
    out_dir: Path,
    max_mb: int,
) -> Path:
    """Download a remote video file to a local path.

    Notes
    -----
    - Intended for direct file URLs (e.g. .mp4 served over https).
    - Will reject obvious non-video responses (HTML pages, etc.).
    """
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _safe_filename_from_url(url)
    out_path = out_dir / fname

    # Stream download with basic size checks.
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(
                "Remote URL returned HTML, not a video file. "
                "Only direct video file URLs are supported (e.g., https://.../clip.mp4)."
            )

        clen = r.headers.get("Content-Length")
        if clen:
            try:
                n_bytes = int(clen)
                if n_bytes > max_mb * 1024 * 1024:
                    raise RuntimeError(f"Remote video too large ({n_bytes/1024/1024:.1f} MB > {max_mb} MB).")
            except ValueError:
                pass

        max_bytes = max_mb * 1024 * 1024
        written = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError(f"Remote video exceeded size limit ({max_mb} MB).")

    return out_path


def main() -> None:
    args = parse_args()
    state_abbr = args.state.upper()
    use_reddit = args.use_reddit.lower() == "true"
    use_trends = args.use_trends.lower() == "true"
    use_video = args.use_video.lower() == "true"
    use_youtube = args.use_youtube.lower() == "true"
    allow_remote_video = args.allow_remote_video.lower() == "true"
    simulate_policy = args.simulate_policy.lower() == "true"
    save_reddit_raw = args.save_reddit_raw.lower() == "true"
    reddit_include_post_id = args.reddit_include_post_id.lower() == "true"

    video_paths: List[str] = []
    if args.video_paths.strip():
        video_paths = [p.strip() for p in args.video_paths.split(",") if p.strip()]

    youtube_urls: List[str] = []
    if args.youtube_urls.strip():
        youtube_urls = [u.strip() for u in args.youtube_urls.split(",") if u.strip()]
    youtube_languages = [l.strip() for l in str(args.youtube_languages or "en").split(",") if l.strip()]

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

        overdose_values = pd.Series(pd.to_numeric(overdose_ts["value"], errors="coerce"), dtype="float64")

        latest_overdose = (
            float(overdose_values.dropna().tail(1).iloc[0])
            if not overdose_ts.empty
            else 0.0
        )
        trend_velocity = (
            float(overdose_values.diff().dropna().tail(6).mean() or 0.0)
            if len(overdose_ts) > 1
            else 0.0
        )

        # ── Census ────────────────────────────────────────────────────────
        state_fips = STATE_TO_FIPS.get(state, "20")
        census_ctx = census.get_state_context(state_fips=state_fips)
        census_ctx.to_csv(out_dir / f"census_context_{state}.csv", index=False)

        # Keep legacy file name for the dashboard
        census_df = census.get_state_population(state_fips=state_fips)
        census_df.to_csv(out_dir / f"census_{state}.csv", index=False)

        median_income = None
        if not census_df.empty and "B19013_001E" in census_df.columns:
            income_series = pd.Series(pd.to_numeric(census_df["B19013_001E"], errors="coerce"))
            median_income = income_series.fillna(0.0).iloc[0]

        population = None
        if not census_ctx.empty and "population" in census_ctx.columns:
            pop_series = pd.Series(pd.to_numeric(census_ctx["population"], errors="coerce"))
            pop_val = pop_series.fillna(0.0).iloc[0]
            population = float(pop_val) if pop_val > 0 else None

        poverty_rate = None
        if not census_ctx.empty and "poverty_rate_pct" in census_ctx.columns:
            pr_series = pd.Series(pd.to_numeric(census_ctx["poverty_rate_pct"], errors="coerce"))
            pr = pr_series.fillna(0.0).iloc[0]
            poverty_rate = float(pr)

        # ── NIDA/SAMHSA ───────────────────────────────────────────────────
        nida_df = nida.get_state_drug_stats(state)
        nida_df.to_csv(out_dir / f"nida_{state}.csv", index=False)
        nida_row = nida_df.iloc[0].to_dict() if not nida_df.empty else {}

        # ── Reddit / social signals ───────────────────────────────────────
        reddit_df = pd.DataFrame(columns=["title", "selftext"])  # default empty
        semantic_method = ""
        if use_reddit:
            reddit_df = reddit.search_posts(
                query="opioid OR fentanyl OR relapse OR overdose OR withdrawal OR naloxone",
                subreddit="all",
                limit=75,
                include_post_id=reddit_include_post_id,
            )
            if not reddit_df.empty:
                reddit_df["text"] = reddit_df["title"].fillna("") + " " + reddit_df["selftext"].fillna("")
                reddit_df = analyze_dataframe(reddit_df, text_col="text", include_matches=True)

                # Embedding-style semantic similarity signals (optional, local).
                sem_df, semantic_method = score_semantic_similarity(reddit_df["text"].fillna("").tolist())
                reddit_df = pd.concat([reddit_df.reset_index(drop=True), sem_df.reset_index(drop=True)], axis=1)
                semantic_summary = aggregate_semantic_signal(reddit_df)

                # Privacy-first export: no raw text by default.
                export_cols = [
                    c
                    for c in reddit_df.columns
                    if c
                    not in {
                        "title",
                        "selftext",
                        "text",
                        "id",
                    }
                ]
                if save_reddit_raw:
                    export_cols = list(reddit_df.columns)
                if reddit_include_post_id and not save_reddit_raw:
                    # Even if IDs are included, keep them out of the default export.
                    export_cols = [c for c in export_cols if c != "id"]

                reddit_df[export_cols].to_csv(out_dir / f"reddit_{state}.csv", index=False)
                # NOTE: social summaries are computed later (after optional YouTube transcript scoring)

                # Method comparison (lexicon vs semantic) – aggregates only.
                compare = {
                    "state": state,
                    "semantic_method": semantic_method or "unknown",
                    "lexicon_composite_risk_mean": float(social_summary.get("composite_risk", 0.0)),
                    "semantic_composite_risk_mean": float(semantic_summary.get("semantic_composite_risk", 0.0)),
                    "lexicon_substance_mean": float(social_summary.get("substance_score", 0.0)),
                    "semantic_substance_mean": float(semantic_summary.get("semantic_substance_score", 0.0)),
                    "lexicon_distress_mean": float(social_summary.get("distress_score", 0.0)),
                    "semantic_distress_mean": float(semantic_summary.get("semantic_distress_score", 0.0)),
                    "n_posts": int(len(reddit_df)),
                }
                pd.DataFrame([compare]).to_csv(out_dir / f"reddit_method_compare_{state}.csv", index=False)

        # ── YouTube transcript signals (optional, no video download) ─────
        youtube_scored_df = pd.DataFrame()
        youtube_status: Dict[str, object] = {
            "use_youtube": bool(use_youtube),
            "n_urls": int(len(youtube_urls)),
            "error": "",
        }
        if use_youtube:
            if not youtube_urls:
                youtube_status["error"] = "No YouTube URLs provided (pass --youtube-urls)."
            else:
                try:
                    from src.video.youtube_signals import extract_youtube_transcript_signals

                    yt_res = extract_youtube_transcript_signals(
                        youtube_urls,
                        languages=youtube_languages or ["en"],
                        max_segments=int(args.youtube_max_segments),
                        include_text=False,
                    )
                    youtube_status.update(yt_res.status)
                    youtube_status["semantic_method"] = yt_res.semantic_method

                    yt_per_video = yt_res.per_video.copy()
                    yt_per_video.to_csv(out_dir / f"youtube_{state}.csv", index=False)

                    youtube_scored_df = yt_res.per_segment.copy()

                except Exception as exc:
                    youtube_status["error"] = str(exc)

            try:
                with open(out_dir / f"youtube_status_{state}.json", "w", encoding="utf-8") as f:
                    json.dump(youtube_status, f, indent=2)
            except Exception as exc:
                print(f"Could not write YouTube status metadata: {exc}")

        # ── Google Trends ────────────────────────────────────────────────
        trends_df = pd.DataFrame()
        trend_summary: Dict[str, float] = {}
        if use_trends:
            trends_df = trends.get_substance_trend_summary(timeframe="today 12-m", geo="US")
            trends_df.to_csv(out_dir / f"trends_{state}.csv", index=False)
            if not trends_df.empty:
                for cat in ["opioid", "stimulant", "treatment"]:
                    sub = trends_df[trends_df["category"] == cat]
                    trend_summary[f"trends_{cat}_mean"] = (
                        float(pd.to_numeric(sub["mean_interest"], errors="coerce").mean()) if not sub.empty else 0.0
                    )

        # ── Social fusion frame (Reddit + optional YouTube transcript) ───
        social_frames: List[pd.DataFrame] = []
        social_cols = [
            "substance_score",
            "distress_score",
            "urgency_score",
            "composite_risk",
            "semantic_substance_score",
            "semantic_distress_score",
            "semantic_help_seeking_score",
            "semantic_composite_risk",
        ]
        if not reddit_df.empty:
            present_cols = [c for c in social_cols if c in reddit_df.columns]
            if present_cols:
                social_frames.append(pd.DataFrame(reddit_df.loc[:, list(present_cols)]))
        if not youtube_scored_df.empty:
            present_cols = [c for c in social_cols if c in youtube_scored_df.columns]
            if present_cols:
                social_frames.append(pd.DataFrame(youtube_scored_df.loc[:, list(present_cols)]))

        social_for_fusion = pd.concat(social_frames, ignore_index=True) if social_frames else pd.DataFrame()
        social_summary = aggregate_signal(social_for_fusion) if not social_for_fusion.empty else {}
        semantic_summary = aggregate_semantic_signal(social_for_fusion) if not social_for_fusion.empty else {}

        # ── Fusion & EWS ─────────────────────────────────────────────────
        sig_dict = build_signal_dict(
            cdc_df=county_df,
            nida_df=nida_df,
            census_df=census_ctx,
            reddit_df=social_for_fusion if not social_for_fusion.empty else None,
            trends_df=trends_df,
        )

        # ── Video behavioral signals (optional) ───────────────────────────
        video_window_df = pd.DataFrame()
        video_summary: Dict[str, float] = {}
        video_status: Dict[str, object] = {
            "use_video": bool(use_video),
            "n_paths": int(len(video_paths)),
            "allow_remote_video": bool(allow_remote_video),
            "processed": False,
            "n_windows": 0,
            "wrote_windows_csv": False,
            "error": "",
        }

        if use_video:
            if not video_paths:
                video_status["error"] = "No video paths provided (pass --video-paths)."
            else:
                try:
                    # Resolve remote URLs into downloaded local files (optional).
                    resolved_paths: List[str] = []
                    downloaded: List[str] = []
                    for vp in video_paths:
                        if _is_http_url(vp):
                            if not allow_remote_video:
                                raise RuntimeError(
                                    "Remote video URL provided but --allow-remote-video is false. "
                                    "Re-run with --allow-remote-video true and use a direct video file URL."
                                )
                            local = _download_remote_video(
                                vp,
                                out_dir=Path("data/cache") / "_remote_videos" / state,
                                max_mb=int(args.remote_video_max_mb),
                            )
                            resolved_paths.append(str(local))
                            downloaded.append(vp)
                        else:
                            resolved_paths.append(vp)

                    video_status["resolved_paths"] = resolved_paths
                    if downloaded:
                        video_status["downloaded_urls"] = downloaded

                    video_window_df, video_summary = extract_multi_video_signals(resolved_paths)
                    video_status["processed"] = True
                    video_status["n_windows"] = int(len(video_window_df))

                    # Always write a readable CSV if video was attempted so the dashboard can tell
                    # "ran but zero windows" apart from "not enabled".
                    video_out = out_dir / f"video_windows_{state}.csv"
                    expected_cols = [
                        "video_name",
                        "window_index",
                        "start_sec",
                        "end_sec",
                        "activity_mean",
                        "activity_std",
                        "anomaly_score",
                        "low_light_frac",
                        "scene_change_rate",
                    ]
                    if video_window_df.empty:
                        pd.DataFrame(columns=expected_cols).to_csv(video_out, index=False)
                    else:
                        # Ensure stable column ordering if possible.
                        for c in expected_cols:
                            if c not in video_window_df.columns:
                                video_window_df[c] = None
                        video_window_df[expected_cols].to_csv(video_out, index=False)

                    video_status["wrote_windows_csv"] = True

                    # Add summary scalars to fusion/EWS inputs
                    for k, v in video_summary.items():
                        if k.startswith("video_"):
                            sig_dict[k] = float(v)
                except Exception as exc:
                    msg = str(exc)
                    video_status["error"] = msg
                    print(f"Video processing skipped/failed: {msg}")

            # Write status metadata regardless of success.
            try:
                with open(out_dir / f"video_status_{state}.json", "w", encoding="utf-8") as f:
                    json.dump(video_status, f, indent=2)
            except Exception as exc:
                print(f"Could not write video status metadata: {exc}")

        # add explicitly computed signals
        sig_dict["trend_velocity"] = trend_velocity
        fusion_inputs[state] = sig_dict

        fusion_result = fuse(sig_dict)
        fusion_df = pd.DataFrame(
            [
                {
                    "state": state,
                    "fusion_score": fusion_result.fusion_score,
                    "alert_level": fusion_result.alert_level,
                    "confidence": fusion_result.confidence,
                    **{f"domain_{k}": v for k, v in fusion_result.domain_scores.items()},
                }
            ]
        )
        fusion_df.to_csv(out_dir / f"fusion_{state}.csv", index=False)

        risk_row = {
            "state": state,
            "trend_velocity": trend_velocity,
            "distress_mentions": float(social_summary.get("distress_score", 0.0) * 5.0),
            "substance_mentions": float(social_summary.get("substance_score", 0.0) * 5.0),
            "opioid_misuse_pct": float(nida_row.get("opioid_misuse_pct", 0.0)),
            "social_composite": float(social_summary.get("composite_risk", 0.0)),
            "semantic_social_composite": float(semantic_summary.get("semantic_composite_risk", 0.0)),
            "trends_interest": float(trend_summary.get("trends_opioid_mean", 0.0)),
            "poverty_rate_pct": poverty_rate if poverty_rate is not None else 0.0,
            "overdose_rate_per_100k": float(latest_overdose),
            "video_activity_mean": float(sig_dict.get("video_activity_mean", 0.0) or 0.0),
            "video_anomaly_score": float(sig_dict.get("video_anomaly_score", 0.0) or 0.0),
            "video_low_light_frac": float(sig_dict.get("video_low_light_frac", 0.0) or 0.0),
            "video_scene_change_rate": float(sig_dict.get("video_scene_change_rate", 0.0) or 0.0),
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
        series = pd.Series(pd.to_numeric(overdose_ts["value"], errors="coerce")).fillna(0.0)
        forecast_df = ensemble_forecast(series.tail(60), horizon=args.horizon)
        forecast_df.to_csv(out_dir / f"forecast_{state}.csv", index=False)

        # Anomalies on recent history
        recent = overdose_ts.tail(max(24, args.horizon * 2)).copy()
        recent["value"] = pd.Series(pd.to_numeric(recent["value"], errors="coerce"), index=recent.index).fillna(0.0)
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
            slice_values = pd.Series(pd.to_numeric(ts_slice["value"], errors="coerce"))
            vmax = float(slice_values.max() or 0.0)
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

        md = _narrative_to_markdown(
            title=narrative.title,
            summary=narrative.summary,
            bullets=narrative.bullets,
            caveats=narrative.caveats,
        )
        (out_dir / f"narrative_{state}.md").write_text(md, encoding="utf-8")

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
