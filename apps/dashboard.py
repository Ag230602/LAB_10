from __future__ import annotations

"""Streamlit dashboard entrypoint.

Run with:
  streamlit run apps/dashboard.py

Backwards compatible wrapper exists at `scripts/app.py`.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import networkx as nx

    _NX_AVAILABLE = True
except Exception:
    nx = None  # type: ignore[assignment]
    _NX_AVAILABLE = False

try:
    from pyvis.network import Network

    _PYVIS_AVAILABLE = True
except Exception:
    Network = None  # type: ignore[assignment]
    _PYVIS_AVAILABLE = False

st.set_page_config(page_title="DMARG Dashboard", layout="wide")
st.title("Dynamic Multimodal Addiction Risk Graph (DMARG)")
st.caption("API-first demo for the NSF NRT AI challenge")

# Bump this string whenever you want a quick visual confirmation that
# the Streamlit server is running the latest code.
DASHBOARD_BUILD = "2026-04-09-kgviz-pipeline-controls"

st.sidebar.caption(f"Build: {DASHBOARD_BUILD}")
with st.sidebar.expander("Runtime info", expanded=False):
    st.code(
        "\n".join(
            [
                f"dashboard_file: {Path(__file__).resolve()}",
                f"python: {sys.executable}",
                f"cwd: {Path.cwd()}",
            ]
        )
    )

state = st.sidebar.text_input("State abbreviation", value="KS").upper()
use_national = st.sidebar.checkbox("Show national outputs if available", value=False)

# Default; may be overridden by the sidebar expander below.
auto_run_missing = True


def _run_pipeline_from_dashboard(
    *,
    state_abbr: str,
    use_reddit: bool,
    use_trends: bool,
    use_video: bool,
    video_paths: str,
    snapshots: int,
    horizon: int,
    allow_remote_video: bool = False,
    remote_video_max_mb: int = 500,
    use_youtube: bool = False,
    youtube_urls: str = "",
    youtube_languages: str = "en",
    youtube_max_segments: int = 400,
) -> tuple[bool, str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "apps.pipeline",
        "--state",
        state_abbr,
        "--use-reddit",
        "true" if use_reddit else "false",
        "--use-trends",
        "true" if use_trends else "false",
        "--use-video",
        "true" if use_video else "false",
        "--video-paths",
        video_paths or "",
        "--allow-remote-video",
        "true" if allow_remote_video else "false",
        "--remote-video-max-mb",
        str(int(remote_video_max_mb)),
        "--use-youtube",
        "true" if use_youtube else "false",
        "--youtube-urls",
        youtube_urls or "",
        "--youtube-languages",
        youtube_languages or "en",
        "--youtube-max-segments",
        str(int(youtube_max_segments)),
        "--snapshots",
        str(int(snapshots)),
        "--horizon",
        str(int(horizon)),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        out = out.strip()
        ok = proc.returncode == 0
        if not out:
            out = f"Pipeline exited with code {proc.returncode}."
        return ok, out
    except Exception as exc:
        return False, str(exc)


def _save_uploaded_video(*, state_abbr: str, uploaded) -> str:
    """Persist an uploaded video to disk and return the local path."""
    out_dir = Path("data/cache") / "_uploads" / state_abbr
    out_dir.mkdir(parents=True, exist_ok=True)

    # Streamlit UploadedFile has .name and .getbuffer().
    name = getattr(uploaded, "name", "uploaded_video.mp4")
    safe_name = "".join([c if c.isalnum() or c in {".", "_", "-"} else "_" for c in str(name)])
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{safe_name}"

    with open(out_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return str(out_path)


with st.sidebar.expander("Generate / refresh outputs", expanded=False):
    st.caption("Runs the pipeline and refreshes the dashboard outputs.")
    auto_run_missing = st.checkbox("Auto-run if outputs missing", value=auto_run_missing)
    run_reddit = st.checkbox("Use Reddit", value=False)
    run_trends = st.checkbox("Use Google Trends", value=False)
    run_video = st.checkbox("Use Video", value=False)
    video_paths = ""
    if run_video:
        video_paths = st.text_input("Video paths (comma-separated)", value="")
    snapshots = st.slider("Temporal snapshots", min_value=1, max_value=24, value=6)
    horizon = st.slider("Forecast horizon", min_value=3, max_value=24, value=12)

    if st.button("Run pipeline now", type="primary"):
        with st.spinner("Running pipeline…"):
            ok, logs = _run_pipeline_from_dashboard(
                state_abbr=state,
                use_reddit=run_reddit,
                use_trends=run_trends,
                use_video=run_video,
                video_paths=video_paths,
                snapshots=snapshots,
                horizon=horizon,
            )
        if ok:
            st.success("Pipeline finished. Reloading outputs…")
            st.rerun()
        else:
            st.error("Pipeline failed. See logs below.")
            st.code(logs[:8000])

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
video_status_path = base / f"video_status_{state}.json"
youtube_path = base / f"youtube_{state}.csv"
youtube_status_path = base / f"youtube_status_{state}.json"

if not risk_path.exists():
    # If the user runs the dashboard before generating outputs, offer an automatic path.
    if auto_run_missing:
        with st.spinner("No outputs found for this state. Running pipeline…"):
            ok, logs = _run_pipeline_from_dashboard(
                state_abbr=state,
                use_reddit=False,
                use_trends=False,
                use_video=False,
                video_paths="",
                snapshots=6,
                horizon=12,
            )
        if ok and risk_path.exists():
            st.rerun()
        st.warning("Pipeline did not produce outputs. See logs.")
        st.code(logs[:8000])
        st.stop()

    st.warning(
        f"No outputs found for {state}. Use the sidebar 'Generate / refresh outputs' to run the pipeline."
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

video_status: dict = {}
if video_status_path.exists():
    try:
        with open(video_status_path, "r", encoding="utf-8") as f:
            video_status = json.load(f) or {}
    except Exception:
        video_status = {}

youtube_df = pd.read_csv(youtube_path) if youtube_path.exists() else pd.DataFrame()
youtube_status: dict = {}
if youtube_status_path.exists():
    try:
        with open(youtube_status_path, "r", encoding="utf-8") as f:
            youtube_status = json.load(f) or {}
    except Exception:
        youtube_status = {}

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
    st.subheader("Interactive knowledge graph")

    if node_df.empty or edge_df.empty:
        st.info("No graph export found. Re-run the pipeline to generate graph_nodes / graph_edges.")
    elif not _PYVIS_AVAILABLE and _NX_AVAILABLE:
        st.caption("PyVis not available; using Plotly fallback visualization.")

        e = edge_df.copy()
        n = node_df.copy()
        if "weight" in e.columns:
            w_series = pd.to_numeric(e["weight"], errors="coerce")
            e["weight"] = pd.Series(w_series).fillna(0.0)
        else:
            e["weight"] = 1.0
        if "relation" not in e.columns:
            e["relation"] = "related_to"
        if "risk_score" in n.columns:
            n["risk_score"] = pd.Series(pd.to_numeric(n["risk_score"], errors="coerce")).fillna(0.0)
        else:
            n["risk_score"] = 0.0

        rel_options = ["(all)"] + sorted([str(r) for r in e["relation"].dropna().unique().tolist()])
        selected_rel = st.selectbox("Relation", options=rel_options, index=0)
        max_edges = int(st.slider("Max edges", min_value=10, max_value=500, value=75, step=5))

        if selected_rel != "(all)":
            e = e[e["relation"].astype(str) == selected_rel]
        e = e.sort_values(by=["weight"], ascending=False).head(max_edges)  # type: ignore[call-arg]

        assert nx is not None
        G = nx.DiGraph()
        for row in n.itertuples(index=False):
            node_id = str(getattr(row, "node"))
            G.add_node(
                node_id,
                label=str(getattr(row, "label", node_id)),
                node_type=str(getattr(row, "node_type", "")),
                risk_score=float(getattr(row, "risk_score", 0.0) or 0.0),
                color=str(getattr(row, "color", "#4e79a7")),
            )
        for row in e.itertuples(index=False):
            G.add_edge(
                str(getattr(row, "source")),
                str(getattr(row, "target")),
                relation=str(getattr(row, "relation", "related_to")),
                weight=float(getattr(row, "weight", 0.0) or 0.0),
            )

        if len(G) == 0:
            st.info("No edges after filtering.")
        else:
            pos = nx.spring_layout(G, seed=42)

            edge_x: list[float] = []
            edge_y: list[float] = []
            edge_w: list[float] = []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, np.nan]
                edge_y += [y0, y1, np.nan]
                edge_w.append(float(d.get("weight", 0.0) or 0.0))

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(120,120,120,0.45)"),
                hoverinfo="none",
            )

            node_x: list[float] = []
            node_y: list[float] = []
            node_sizes: list[float] = []
            node_colors: list[str] = []
            node_text: list[str] = []
            for node_id, attrs in G.nodes(data=True):
                x, y = pos[node_id]
                node_x.append(float(x))
                node_y.append(float(y))
                rs = float(attrs.get("risk_score", 0.0) or 0.0)
                node_sizes.append(12 + rs * 26.0)
                node_colors.append(str(attrs.get("color", "#4e79a7")))
                node_text.append(
                    f"{attrs.get('label', node_id)}<br>type: {attrs.get('node_type','')}<br>risk_score: {rs:.3f}"
                )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[str(a.get("label", nid)) for nid, a in G.nodes(data=True)],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="rgba(20,20,20,0.6)")),
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                height=700,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Graph tables (nodes / edges)"):
            st.subheader("Graph nodes")
            st.dataframe(node_df, use_container_width=True)
            st.subheader("Graph edges")
            st.dataframe(edge_df, use_container_width=True)

    elif not _PYVIS_AVAILABLE and not _NX_AVAILABLE:
        st.warning("Graph visualization requires `pyvis` or `networkx` for fallback. Showing tables only.")
        st.dataframe(node_df, use_container_width=True)
        st.dataframe(edge_df, use_container_width=True)

    else:
        e = edge_df.copy()
        n = node_df.copy()

        if "weight" in e.columns:
            w_series = pd.to_numeric(e["weight"], errors="coerce")
            e["weight"] = pd.Series(w_series).fillna(0.0)
        else:
            e["weight"] = 1.0

        if "relation" not in e.columns:
            e["relation"] = "related_to"

        rel_options = ["(all)"] + sorted([str(r) for r in e["relation"].dropna().unique().tolist()])
        selected_rel = st.selectbox("Relation", options=rel_options, index=0)

        max_w = float(e["weight"].max() or 1.0)
        min_w = float(st.slider("Minimum edge weight", min_value=0.0, max_value=max(0.01, max_w), value=0.0))

        max_edges = int(st.slider("Max edges", min_value=10, max_value=500, value=75, step=5))
        physics = st.checkbox("Enable physics (drag + settle)", value=True)
        show_edge_labels = st.checkbox("Show edge labels", value=False)

        if selected_rel != "(all)":
            e = e[e["relation"].astype(str) == selected_rel]
        e = e[e["weight"] >= min_w]
        e = e.sort_values(by=["weight"], ascending=False).head(max_edges)  # type: ignore[call-arg]

        keep_nodes = set(e["source"].astype(str).tolist()) | set(e["target"].astype(str).tolist())
        n = n[n["node"].astype(str).isin(keep_nodes)].copy()

        assert Network is not None
        net = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")

        if physics:
            net.force_atlas_2based(
                gravity=-40,
                central_gravity=0.01,
                spring_length=120,
                spring_strength=0.02,
                damping=0.4,
                overlap=0.2,
            )
        else:
            net.toggle_physics(False)

        if "risk_score" in n.columns:
            n["risk_score"] = pd.to_numeric(n["risk_score"], errors="coerce").fillna(0.0)
        else:
            n["risk_score"] = 0.0

        for row in n.itertuples(index=False):
            node_id = str(getattr(row, "node"))
            label = str(getattr(row, "label", node_id))
            node_type = str(getattr(row, "node_type", ""))
            risk_score = float(getattr(row, "risk_score", 0.0) or 0.0)
            color = str(getattr(row, "color", "#4e79a7"))

            size = 14 + (risk_score * 22.0)
            title = f"{label}<br>type: {node_type}<br>risk_score: {risk_score:.3f}"
            net.add_node(node_id, label=label, title=title, color=color, size=size)

        for row in e.itertuples(index=False):
            src = str(getattr(row, "source"))
            tgt = str(getattr(row, "target"))
            rel = str(getattr(row, "relation", "related_to"))
            w = float(getattr(row, "weight", 0.0) or 0.0)
            title = f"{rel}<br>weight: {w:.3f}"
            net.add_edge(src, tgt, value=max(w, 0.001), title=title, label=(rel if show_edge_labels else ""))

        html = net.generate_html()
        components.html(html, height=700, scrolling=True)

        with st.expander("Graph tables (nodes / edges)"):
            st.subheader("Graph nodes")
            st.dataframe(node_df, use_container_width=True)
            st.subheader("Graph edges")
            st.dataframe(edge_df, use_container_width=True)

with tabs[5]:
    st.subheader("Policy simulations")
    if policy_compare_path.exists():
        comp = pd.read_csv(policy_compare_path)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("Policy simulation outputs not found. Re-run pipeline with `--simulate-policy true`. ")

with tabs[6]:
    st.subheader("Video-derived behavioral signals (non-identifying)")

    with st.expander("Analyze a video now", expanded=False):
        st.caption(
            "Paste a direct video file URL (e.g., .../clip.mp4) to analyze it. "
            "YouTube links are typically web pages/streams (not direct files) and cannot be analyzed directly with OpenCV."
        )

        yt_url = st.text_input("YouTube link (view only)", value="")
        if yt_url.strip():
            try:
                st.video(yt_url.strip())
            except Exception:
                st.info("Could not render that link in Streamlit.")

        st.divider()
        st.subheader("YouTube transcript signals (no video download)")
        st.caption(
            "Computes text-based risk proxy signals from the public transcript (when available). "
            "Requires `youtube-transcript-api` in your environment."
        )
        yt_analyze_url = st.text_input("YouTube URL to analyze (transcript)", value="")
        yt_langs = st.text_input("Transcript languages to try", value="en")
        yt_max = st.slider("Max transcript segments", min_value=50, max_value=2000, value=400, step=50)
        if yt_analyze_url.strip():
            if st.button("Analyze YouTube transcript (run pipeline)"):
                with st.spinner("Fetching transcript + running pipeline…"):
                    ok, logs = _run_pipeline_from_dashboard(
                        state_abbr=state,
                        use_reddit=False,
                        use_trends=False,
                        use_video=False,
                        video_paths="",
                        snapshots=6,
                        horizon=12,
                        use_youtube=True,
                        youtube_urls=yt_analyze_url.strip(),
                        youtube_languages=yt_langs.strip() or "en",
                        youtube_max_segments=int(yt_max),
                    )
                if ok:
                    st.success("Done. Reloading outputs…")
                    st.rerun()
                else:
                    st.error("Pipeline failed. See logs below.")
                    st.code(logs[:8000])

        remote_url = st.text_input("Direct video file URL to analyze (mp4/mov)", value="")
        remote_max_mb = st.slider("Max download size (MB)", min_value=50, max_value=2000, value=500, step=50)
        if remote_url.strip():
            st.caption("Tip: the URL must directly serve a video file (Content-Type video/*).")
            if st.button("Analyze URL video (download + run pipeline)"):
                with st.spinner("Downloading + running pipeline…"):
                    ok, logs = _run_pipeline_from_dashboard(
                        state_abbr=state,
                        use_reddit=False,
                        use_trends=False,
                        use_video=True,
                        video_paths=remote_url.strip(),
                        snapshots=6,
                        horizon=12,
                        allow_remote_video=True,
                        remote_video_max_mb=int(remote_max_mb),
                    )
                if ok:
                    st.success("Done. Reloading outputs…")
                    st.rerun()
                else:
                    st.error("Pipeline failed. See logs below.")
                    st.code(logs[:8000])

        uploaded = st.file_uploader(
            "Upload a video to analyze (mp4/mov)",
            type=["mp4", "mov", "m4v"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            if st.button("Analyze uploaded video (run pipeline)"):
                local_path = _save_uploaded_video(state_abbr=state, uploaded=uploaded)
                with st.spinner("Running pipeline with uploaded video…"):
                    ok, logs = _run_pipeline_from_dashboard(
                        state_abbr=state,
                        use_reddit=False,
                        use_trends=False,
                        use_video=True,
                        video_paths=local_path,
                        snapshots=6,
                        horizon=12,
                    )
                if ok:
                    st.success("Done. Reloading outputs…")
                    st.rerun()
                else:
                    st.error("Pipeline failed. See logs below.")
                    st.code(logs[:8000])

    # Always show aggregate scalars (they exist even when video is not enabled).
    scalar_cols = [
        c
        for c in [
            "video_activity_mean",
            "video_anomaly_score",
            "video_low_light_frac",
            "video_scene_change_rate",
        ]
        if c in risk_df.columns
    ]
    if scalar_cols:
        vals = risk_df.iloc[0][scalar_cols].to_dict()
        cols = st.columns(len(scalar_cols))
        for i, k in enumerate(scalar_cols):
            try:
                cols[i].metric(k, round(float(pd.to_numeric(vals.get(k), errors="coerce") or 0.0), 4))
            except Exception:
                cols[i].metric(k, "N/A")

    if video_windows_path.exists() and video_windows_df.empty and not video_status:
        st.caption("Note: video window CSV exists but no status metadata found (older pipeline run).")

    if video_windows_df.empty:
        # Distinguish: (a) video disabled, (b) enabled but misconfigured, (c) enabled but failed.
        if video_status:
            if video_status.get("error"):
                st.warning(
                    "Video processing failed. "
                    f"Error: {video_status.get('error')}\n\n"
                    "If you want video signals, install `opencv-python` and re-run the pipeline with "
                    "`--use-video true --video-paths /path/to/video.mp4`. "
                    "For direct remote file URLs, add `--allow-remote-video true` (MP4/Mov links only)."
                )
            elif int(video_status.get("n_paths", 0) or 0) <= 0:
                st.info(
                    "Video processing was enabled, but no `--video-paths` were provided. "
                    "Re-run the pipeline with `--use-video true --video-paths /path/to/video.mp4`. "
                    "(Direct remote file URLs supported with `--allow-remote-video true`.)"
                )
            else:
                st.info(
                    "Video processing ran, but no window-level rows were produced. "
                    "This can happen with very short/invalid videos or decode issues. "
                    "Try a longer video and ensure `opencv-python` is installed."
                )
        else:
            st.info(
                "No video window signals were generated for this state. "
                "This usually means the pipeline was run with `--use-video false` (default).\n\n"
                "To enable video signals: run the pipeline with `--use-video true --video-paths /path/to/video.mp4` "
                "(requires `opencv-python`). Direct remote file URLs also work with `--allow-remote-video true`."
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
            s = pd.Series(pd.to_numeric(vdf["activity_mean"], errors="coerce"))
            top[0].metric("Activity (mean)", round(float(s.mean()), 3))
        if "anomaly_score" in vdf.columns:
            s = pd.Series(pd.to_numeric(vdf["anomaly_score"], errors="coerce"))
            top[1].metric("Anomaly (mean)", round(float(s.mean()), 3))
        if "low_light_frac" in vdf.columns:
            s = pd.Series(pd.to_numeric(vdf["low_light_frac"], errors="coerce"))
            top[2].metric("Low-light (mean)", round(float(s.mean()), 3))
        if "scene_change_rate" in vdf.columns:
            s = pd.Series(pd.to_numeric(vdf["scene_change_rate"], errors="coerce"))
            top[3].metric("Scene change (mean)", round(float(s.mean()), 3))

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

    st.subheader("YouTube transcript summary")
    if youtube_status:
        with st.expander("YouTube transcript status", expanded=False):
            st.json(youtube_status)
    if not youtube_df.empty:
        st.dataframe(youtube_df, use_container_width=True)
    else:
        st.caption("No YouTube transcript outputs found for this state.")

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
