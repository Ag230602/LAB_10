from __future__ import annotations

"""Streamlit dashboard entrypoint.

Run with:
  streamlit run apps/dashboard.py

Backwards compatible wrapper exists at `scripts/app.py`.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DMARG Dashboard", layout="wide")
st.title("Dynamic Multimodal Addiction Risk Graph (DMARG)")
st.caption("API-first demo for the NSF NRT AI challenge")

state = st.sidebar.text_input("State abbreviation", value="KS").upper()
use_national = st.sidebar.checkbox("Show national outputs if available", value=False)
base = Path("data/cache")

risk_path = base / f"risk_{state}.csv"
forecast_path = base / f"forecast_{state}.csv"
cdc_path = base / f"cdc_county_{state}.csv"
edge_path = base / f"graph_edges_{state}.csv"
node_path = base / f"graph_nodes_{state}.csv"
fusion_path = base / f"fusion_{state}.csv"
anomaly_path = base / f"anomalies_{state}.csv"
narrative_path = base / f"narrative_{state}.json"
policy_compare_path = base / f"policy_compare_{state}.csv"
temporal_summary_path = base / f"temporal_summary_{state}.csv"
temporal_edges_path = base / f"temporal_edges_{state}.csv"
video_windows_path = base / f"video_windows_{state}.csv"

if not risk_path.exists():
    st.warning(
        f"Run `PYTHONPATH=. .venv/bin/python apps/pipeline.py --state {state} --use-reddit false --use-trends false` first."
    )
    st.stop()

risk_df = pd.read_csv(risk_path)
forecast_df = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
cdc_df = pd.read_csv(cdc_path) if cdc_path.exists() else pd.DataFrame()
edge_df = pd.read_csv(edge_path) if edge_path.exists() else pd.DataFrame()
node_df = pd.read_csv(node_path) if node_path.exists() else pd.DataFrame()
fusion_df = pd.read_csv(fusion_path) if fusion_path.exists() else pd.DataFrame()
anomaly_df = pd.read_csv(anomaly_path) if anomaly_path.exists() else pd.DataFrame()
temporal_df = pd.read_csv(temporal_summary_path) if temporal_summary_path.exists() else pd.DataFrame()
temporal_edges_df = pd.read_csv(temporal_edges_path) if temporal_edges_path.exists() else pd.DataFrame()
video_windows_df = pd.read_csv(video_windows_path) if video_windows_path.exists() else pd.DataFrame()

if use_national:
    nat_edges = base / "graph_edges_national.csv"
    nat_nodes = base / "graph_nodes_national.csv"
    if nat_edges.exists() and nat_nodes.exists():
        edge_df = pd.read_csv(nat_edges)
        node_df = pd.read_csv(nat_nodes)

top_cols = st.columns(5)
top_cols[0].metric("State", state if not use_national else "National")
top_cols[1].metric("EWS", round(float(risk_df["ews"].iloc[0]), 3))
if "ews_alert_level" in risk_df.columns:
    top_cols[2].metric("Alert", str(risk_df["ews_alert_level"].iloc[0]))
else:
    top_cols[2].metric("Alert", "N/A")

trend = float(risk_df["trend_velocity"].iloc[0]) if "trend_velocity" in risk_df.columns else 0.0
top_cols[3].metric("Trend Velocity", round(trend, 3))
if not fusion_df.empty and "fusion_score" in fusion_df.columns:
    top_cols[4].metric("Fusion Score", round(float(fusion_df["fusion_score"].iloc[0]), 3))
else:
    top_cols[4].metric("Fusion Score", "N/A")

tabs = st.tabs(
    [
        "Overview",
        "Temporal",
        "Forecast",
        "Anomalies",
        "Knowledge Graph",
        "Policy Simulation",
        "Video",
        "Narrative",
        "Raw Data",
    ]
)

with tabs[0]:
    st.subheader("Risk domains")
    domain_cols = [c for c in risk_df.columns if c.startswith("ews_") and c not in ("ews", "ews_alert_level")]
    if domain_cols:
        dom = risk_df[domain_cols].iloc[0].to_dict()
        dom_df = pd.DataFrame({"domain": list(dom.keys()), "score": list(dom.values())})
        dom_df["domain"] = dom_df["domain"].str.replace("ews_", "", regex=False).str.title()
        fig = px.bar(dom_df, x="domain", y="score", title="Domain EWS components", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Domain columns not found in risk output.")

    if not fusion_df.empty:
        st.subheader("Fusion")
        st.dataframe(fusion_df, use_container_width=True)

with tabs[1]:
    st.subheader("Temporal graph summary")
    if not temporal_df.empty:
        tmp = temporal_df.copy()
        if "timestamp" in tmp.columns:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
        y = "mean_ews" if "mean_ews" in tmp.columns else ("max_ews" if "max_ews" in tmp.columns else None)
        if y:
            hover_mode = st.selectbox(
                "Hover behavior",
                options=["closest", "x unified"],
                index=0,
                help="Use 'closest' to avoid a plot-wide crosshair/vertical cursor line.",
            )
            fig = px.line(tmp, x="timestamp", y=y, markers=True, title=f"{y} over snapshots")
            fig.update_layout(hovermode=hover_mode)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tmp, use_container_width=True)
    else:
        st.info("No temporal summary found. Re-run pipeline with `--snapshots` > 1.")

    st.subheader("Edge trajectories")
    if not temporal_edges_df.empty and {"source", "target", "timestamp", "weight"}.issubset(temporal_edges_df.columns):
        tdf = temporal_edges_df.copy()
        tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
        tdf = tdf.dropna(subset=["timestamp"])
        if "relation" not in tdf.columns:
            tdf["relation"] = "related_to"

        rels = ["(all)"] + sorted([r for r in tdf["relation"].dropna().unique().tolist()])
        selected_rel = st.selectbox("Relation filter", options=rels, index=0)
        if selected_rel != "(all)":
            tdf = tdf[tdf["relation"] == selected_rel]

        rank_metric = st.radio(
            "Rank edges by",
            options=["Absolute change (first → last)", "Volatility (std dev)", "Both"],
            horizontal=True,
            index=0,
        )
        top_n = int(st.slider("Top N edges", min_value=10, max_value=200, value=25, step=5))

        tdf = tdf.sort_values(["source", "target", "relation", "timestamp"])
        keys = ["source", "target", "relation"]
        grouped = tdf.groupby(keys)

        start = grouped["weight"].first().reset_index(name="start_weight")
        end = grouped["weight"].last().reset_index(name="end_weight")
        stats = (
            grouped["weight"]
            .agg(
                min_weight="min",
                max_weight="max",
                mean_weight="mean",
                std_weight="std",
                n_points="count",
            )
            .reset_index()
        )

        edge_summary = start.merge(end, on=keys).merge(stats, on=keys)
        edge_summary["delta"] = edge_summary["end_weight"] - edge_summary["start_weight"]
        edge_summary["abs_delta"] = edge_summary["delta"].abs()
        edge_summary["std_weight"] = edge_summary["std_weight"].fillna(0.0)

        if rank_metric in ("Absolute change (first → last)", "Both"):
            by_change = edge_summary.sort_values(["abs_delta", "std_weight"], ascending=False)
            st.caption("Edges ranked by absolute change between first and last snapshot.")
            st.dataframe(by_change.head(top_n), use_container_width=True)
        else:
            by_change = edge_summary

        if rank_metric in ("Volatility (std dev)", "Both"):
            by_vol = edge_summary.sort_values(["std_weight", "abs_delta"], ascending=False)
            st.caption("Edges ranked by volatility (standard deviation of weights across snapshots).")
            st.dataframe(by_vol.head(top_n), use_container_width=True)
        else:
            by_vol = edge_summary

        chooser_base = by_change if rank_metric != "Volatility (std dev)" else by_vol
        top_keys = chooser_base.head(top_n)[keys]
        options = [f"{r.source} → {r.target} ({r.relation})" for r in top_keys.itertuples(index=False)]
        if options:
            selected = st.selectbox("Select an edge to plot", options=options, index=0)
            picked = top_keys.iloc[options.index(selected)].to_dict()
            sel = tdf[
                (tdf["source"] == picked["source"])
                & (tdf["target"] == picked["target"])
                & (tdf["relation"] == picked["relation"])
            ].copy()
            fig = px.line(sel, x="timestamp", y="weight", markers=True, title=f"Edge weight over time: {selected}")
            fig.update_layout(
                hovermode="closest",
                legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sel, use_container_width=True)
    else:
        st.info("No temporal edges found. Re-run pipeline with `--snapshots` > 1.")

with tabs[2]:
    st.subheader("Near-term forecast")
    if not forecast_df.empty:
        fig = go.Figure()
        if "step" not in forecast_df.columns:
            forecast_df = forecast_df.reset_index().rename(columns={"index": "step"})
        if "forecast" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(x=forecast_df["step"], y=forecast_df["forecast"], mode="lines+markers", name="Forecast")
            )
        if "lower_ci" in forecast_df.columns and "upper_ci" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["step"],
                    y=forecast_df["upper_ci"],
                    mode="lines",
                    name="Upper CI",
                    line=dict(width=1, dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["step"],
                    y=forecast_df["lower_ci"],
                    mode="lines",
                    name="Lower CI",
                    line=dict(width=1, dash="dot"),
                )
            )
        fig.update_layout(title="Ensemble forecast", xaxis_title="Step", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True)
    else:
        st.info("No forecast found. Run the pipeline.")

with tabs[3]:
    st.subheader("Spike / anomaly detection")
    if not anomaly_df.empty and "value" in anomaly_df.columns:
        plot_df = anomaly_df.copy()
        if "date" in plot_df.columns:
            plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
            x = "date"
        else:
            plot_df = plot_df.reset_index().rename(columns={"index": "idx"})
            x = "idx"
        fig = px.line(plot_df, x=x, y="value", title="Overdose series (recent)")
        st.plotly_chart(fig, use_container_width=True)
        if "anomaly_score" in plot_df.columns:
            fig2 = px.line(plot_df, x=x, y="anomaly_score", title="Anomaly score", range_y=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(anomaly_df, use_container_width=True)
    else:
        st.info("No anomalies output found.")

with tabs[4]:
    st.subheader("Graph nodes")
    if not node_df.empty:
        st.dataframe(node_df, use_container_width=True)
    else:
        st.info("No node export found.")

    st.subheader("Graph edges")
    if not edge_df.empty:
        st.dataframe(edge_df, use_container_width=True)
    else:
        st.info("No edge export found.")

with tabs[5]:
    st.subheader("Policy simulations")
    if policy_compare_path.exists():
        comp = pd.read_csv(policy_compare_path)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("Policy simulation outputs not found. Re-run pipeline with `--simulate-policy true`. ")

with tabs[6]:
    st.subheader("Video-derived behavioral signals (non-identifying)")
    if video_windows_df.empty:
        st.info(
            "No video window signals found. Run the pipeline with `--use-video true --video-paths ...` "
            "(requires `opencv-python`)."
        )
    else:
        vdf = video_windows_df.copy()

        if {"start_sec", "end_sec"}.issubset(vdf.columns):
            vdf["mid_sec"] = (
                pd.to_numeric(vdf["start_sec"], errors="coerce") + pd.to_numeric(vdf["end_sec"], errors="coerce")
            ) / 2.0
            x = "mid_sec"
            x_label = "Time (sec)"
        else:
            x = "window_index" if "window_index" in vdf.columns else None
            x_label = "Window"
            if x is None:
                vdf = vdf.reset_index().rename(columns={"index": "idx"})
                x = "idx"

        metrics = [
            c
            for c in [
                "activity_mean",
                "anomaly_score",
                "low_light_frac",
                "scene_change_rate",
            ]
            if c in vdf.columns
        ]

        top = st.columns(4)
        if "activity_mean" in vdf.columns:
            top[0].metric("Activity (mean)", round(float(pd.to_numeric(vdf["activity_mean"], errors="coerce").mean()), 3))
        if "anomaly_score" in vdf.columns:
            top[1].metric("Anomaly (mean)", round(float(pd.to_numeric(vdf["anomaly_score"], errors="coerce").mean()), 3))
        if "low_light_frac" in vdf.columns:
            top[2].metric("Low-light (mean)", round(float(pd.to_numeric(vdf["low_light_frac"], errors="coerce").mean()), 3))
        if "scene_change_rate" in vdf.columns:
            top[3].metric(
                "Scene change (mean)", round(float(pd.to_numeric(vdf["scene_change_rate"], errors="coerce").mean()), 3)
            )

        if metrics:
            selected = st.multiselect(
                "Metrics to plot",
                options=metrics,
                default=[c for c in ["activity_mean", "anomaly_score"] if c in metrics] or metrics[:1],
            )
            for m in selected:
                fig = px.line(vdf, x=x, y=m, markers=True, title=f"{m} over {x_label}")
                fig.update_layout(hovermode="closest")
                st.plotly_chart(fig, use_container_width=True)

        st.caption("Window-level signals only (no IDs, no raw frames, no person identification).")
        st.dataframe(vdf, use_container_width=True)

with tabs[7]:
    st.subheader("Risk narrative")
    if narrative_path.exists():
        with open(narrative_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.markdown(f"### {data.get('title','Risk Brief')}")
        st.write(data.get("summary", ""))
        st.markdown("**Key points**")
        for b in data.get("bullets", []):
            st.write(f"- {b}")
        with st.expander("Caveats"):
            for c in data.get("caveats", []):
                st.write(f"- {c}")
    else:
        st.info("Narrative not found. Run the pipeline.")

with tabs[8]:
    st.subheader("CDC sample")
    if not cdc_df.empty:
        st.dataframe(cdc_df.head(50), use_container_width=True)
    st.subheader("Risk output")
    st.dataframe(risk_df, use_container_width=True)
    if not fusion_df.empty:
        st.subheader("Fusion output")
        st.dataframe(fusion_df, use_container_width=True)

with st.expander("Project idea"):
    st.write(
        "DMARG combines CDC public-health signals, Census context, NIDA/SAMHSA indicators, optional social + search signals, "
        "and a dynamic temporal knowledge graph to produce early warnings, forecasts, and policy simulations."
    )
