"""
Multimodal Knowledge Graph Builder
────────────────────────────────────
Constructs a rich directed knowledge graph integrating:
  • Location nodes (state/county)
  • Drug-type nodes (opioid, stimulant, alcohol, etc.)
  • Demographic nodes (poverty, income, unemployment)
  • Social signal nodes (Reddit, Google Trends)
  • Risk / EWS nodes
  • Event nodes (overdose death spike, policy change)

Edge semantics encode causal/correlational relations:
  has_risk, has_event, has_context, influences, co_occurs_with,
  worsened_by, mitigated_by
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd


# ── Node type palette (used for colour-coding in the dashboard) ──────────────
NODE_COLORS: Dict[str, str] = {
    "location":    "#4e79a7",
    "drug":        "#f28e2b",
    "risk_score":  "#e15759",
    "event":       "#76b7b2",
    "context":     "#59a14f",
    "social":      "#edc948",
    "trend":       "#b07aa1",
    "policy":      "#ff9da7",
    "demographic": "#9c755f",
}


def build_risk_graph(
    state_name: str,
    overdose_value: float,
    ews_value: float,
    median_income: Optional[float] = None,
    # New optional enrichment inputs
    nida_row: Optional[Dict] = None,
    social_signals: Optional[Dict] = None,
    trend_signals: Optional[Dict] = None,
    drug_breakdown: Optional[Dict[str, float]] = None,
    timestamp: Optional[datetime] = None,
) -> nx.DiGraph:
    """
    Builds a directed knowledge graph for a single state / time-step.

    Parameters
    ----------
    state_name      : state abbreviation (e.g., "KS")
    overdose_value  : mean overdose deaths (CDC)
    ews_value       : aggregate early warning score 0–1
    median_income   : Census median household income (optional)
    nida_row        : dict of NIDA/SAMHSA signals (optional)
    social_signals  : dict of Reddit/text signal scores (optional)
    trend_signals   : dict of Google Trends scores (optional)
    drug_breakdown  : dict of {drug_name: death_count} (optional)
    timestamp       : snapshot datetime (optional, defaults to now)
    """
    ts = timestamp or datetime.utcnow()
    g = nx.DiGraph()

    # ── State (location) node ─────────────────────────────────────────────────
    g.add_node(
        state_name,
        node_type="location",
        label=state_name,
        color=NODE_COLORS["location"],
        risk_score=ews_value,
        timestamp=ts.isoformat(),
    )

    # ── Overdose event node ───────────────────────────────────────────────────
    ov_node = f"{state_name}_overdose"
    g.add_node(
        ov_node,
        node_type="event",
        label="Overdose Deaths",
        color=NODE_COLORS["event"],
        value=float(overdose_value),
        risk_score=min(float(overdose_value) / 500.0, 1.0),
        timestamp=ts.isoformat(),
    )
    g.add_edge(state_name, ov_node, relation="has_event",
               weight=round(min(float(overdose_value) / 100.0, 1.0), 3))

    # ── EWS risk-score node ───────────────────────────────────────────────────
    ews_node = f"{state_name}_ews"
    g.add_node(
        ews_node,
        node_type="risk_score",
        label="EWS",
        color=NODE_COLORS["risk_score"],
        value=float(ews_value),
        risk_score=float(ews_value),
        timestamp=ts.isoformat(),
    )
    g.add_edge(state_name, ews_node, relation="has_risk", weight=round(float(ews_value), 3))

    # ── Income / demographic context ──────────────────────────────────────────
    if median_income is not None:
        inc_node = f"{state_name}_income"
        inc_risk = float(max(0.0, 1.0 - (median_income - 30_000) / 90_000))
        g.add_node(
            inc_node,
            node_type="demographic",
            label="Median Income",
            color=NODE_COLORS["demographic"],
            value=float(median_income),
            risk_score=round(inc_risk, 3),
            timestamp=ts.isoformat(),
        )
        g.add_edge(state_name, inc_node, relation="has_context", weight=round(inc_risk, 3))
        g.add_edge(inc_node, ews_node, relation="influences", weight=round(inc_risk * 0.15, 3))

    # ── NIDA / SAMHSA signals ─────────────────────────────────────────────────
    if nida_row:
        nida_node = f"{state_name}_nida"
        opioid_pct = float(nida_row.get("opioid_misuse_pct", 0.0))
        illicit_pct = float(nida_row.get("illicit_drug_use_pct", 0.0))
        nida_risk = min((opioid_pct / 9.0 * 0.6 + illicit_pct / 25.0 * 0.4), 1.0)
        g.add_node(
            nida_node,
            node_type="context",
            label="NIDA/SAMHSA",
            color=NODE_COLORS["context"],
            value=nida_risk,
            risk_score=round(nida_risk, 3),
            opioid_misuse_pct=opioid_pct,
            illicit_drug_use_pct=illicit_pct,
            treatment_need_pct=float(nida_row.get("treatment_need_pct", 0.0)),
            timestamp=ts.isoformat(),
        )
        g.add_edge(state_name, nida_node, relation="has_context", weight=round(nida_risk, 3))
        g.add_edge(nida_node, ews_node, relation="influences", weight=round(nida_risk * 0.25, 3))

    # ── Drug-type breakdown ───────────────────────────────────────────────────
    if drug_breakdown:
        total = sum(drug_breakdown.values()) or 1.0
        for drug, count in drug_breakdown.items():
            drug_node = f"{state_name}_{drug.lower().replace(' ', '_')}"
            rel_risk = count / total
            g.add_node(
                drug_node,
                node_type="drug",
                label=drug.title(),
                color=NODE_COLORS["drug"],
                value=float(count),
                risk_score=round(rel_risk, 3),
                timestamp=ts.isoformat(),
            )
            g.add_edge(ov_node, drug_node, relation="attributed_to",
                       weight=round(rel_risk, 3))
            g.add_edge(drug_node, ews_node, relation="co_occurs_with",
                       weight=round(rel_risk * 0.30, 3))

    # ── Social signal node ────────────────────────────────────────────────────
    if social_signals:
        soc_node = f"{state_name}_social"
        composite = float(social_signals.get("composite_risk", 0.0))
        g.add_node(
            soc_node,
            node_type="social",
            label="Social Signals",
            color=NODE_COLORS["social"],
            value=composite,
            risk_score=round(composite, 3),
            substance_score=float(social_signals.get("substance_score", 0.0)),
            distress_score=float(social_signals.get("distress_score", 0.0)),
            urgency_score=float(social_signals.get("urgency_score", 0.0)),
            timestamp=ts.isoformat(),
        )
        g.add_edge(state_name, soc_node, relation="has_social_signal",
                   weight=round(composite, 3))
        g.add_edge(soc_node, ews_node, relation="influences",
                   weight=round(composite * 0.20, 3))

    # ── Google Trends node ────────────────────────────────────────────────────
    if trend_signals:
        trend_node = f"{state_name}_trends"
        trend_risk = float(trend_signals.get("trends_opioid_mean", 50.0)) / 100.0
        g.add_node(
            trend_node,
            node_type="trend",
            label="Search Trends",
            color=NODE_COLORS["trend"],
            value=trend_risk,
            risk_score=round(trend_risk, 3),
            timestamp=ts.isoformat(),
        )
        g.add_edge(state_name, trend_node, relation="has_trend",
                   weight=round(trend_risk, 3))
        g.add_edge(trend_node, ews_node, relation="influences",
                   weight=round(trend_risk * 0.10, 3))

    return g


def multi_state_graph(state_graphs: Dict[str, nx.DiGraph]) -> nx.DiGraph:
    """
    Merges individual state graphs into a single national knowledge graph.
    Adds inter-state edges when two bordering states both exceed a risk threshold.
    """
    combined = nx.DiGraph()
    for state, g in state_graphs.items():
        combined.add_nodes_from(g.nodes(data=True))
        combined.add_edges_from(g.edges(data=True))

    # Add high-risk linkages between states (simplified – based on EWS proximity)
    ews_scores = {}
    for node, data in combined.nodes(data=True):
        if data.get("node_type") == "risk_score":
            state = str(node).split("_")[0]
            ews_scores[state] = float(data.get("value", 0.0))

    states = list(ews_scores.keys())
    for i, s1 in enumerate(states):
        for s2 in states[i + 1:]:
            if ews_scores[s1] > 0.5 and ews_scores[s2] > 0.5:
                weight = round((ews_scores[s1] + ews_scores[s2]) / 2, 3)
                combined.add_edge(s1, s2, relation="co_elevated_risk", weight=weight)

    return combined


def graph_to_edge_frame(graph: nx.Graph) -> pd.DataFrame:
    """Converts any graph to a tidy edge DataFrame."""
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append({
            "source":   str(u),
            "target":   str(v),
            "relation": data.get("relation", "related_to"),
            "weight":   round(float(data.get("weight", 1.0)), 4),
        })
    return pd.DataFrame(rows)


def graph_to_node_frame(graph: nx.Graph) -> pd.DataFrame:
    """Converts graph nodes to a tidy DataFrame."""
    rows = []
    for node, data in graph.nodes(data=True):
        rows.append({
            "node":       str(node),
            "node_type":  data.get("node_type", "unknown"),
            "label":      data.get("label", str(node)),
            "risk_score": round(float(data.get("risk_score", data.get("value", 0.0))), 4),
            "color":      data.get("color", "#cccccc"),
        })
    return pd.DataFrame(rows)
