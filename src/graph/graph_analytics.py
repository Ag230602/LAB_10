"""
Graph Analytics Module
───────────────────────
Provides higher-order analyses on NetworkX graphs:

  • Centrality measures (degree, betweenness, eigenvector, closeness)
  • Community detection (greedy modularity / Louvain approximation)
  • Risk propagation simulation (adapted SIR model on graph)
  • High-risk node identification
  • Graph summary statistics
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd


# ── Centrality ────────────────────────────────────────────────────────────────

def compute_centrality(g: nx.Graph) -> pd.DataFrame:
    """
    Returns a DataFrame with centrality measures for every node.

    Columns: node, degree_centrality, betweenness_centrality,
             closeness_centrality, eigenvector_centrality, node_type
    """
    if g.number_of_nodes() == 0:
        return pd.DataFrame()

    deg   = nx.degree_centrality(g)
    btwn  = nx.betweenness_centrality(g, normalized=True, weight="weight")
    clos  = nx.closeness_centrality(g)

    # Eigenvector may fail on disconnected graphs – fall back gracefully
    try:
        eig = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in g.nodes()}

    rows = []
    for node in g.nodes():
        attrs = g.nodes[node]
        rows.append({
            "node":                   str(node),
            "degree_centrality":      round(deg.get(node, 0.0), 4),
            "betweenness_centrality": round(btwn.get(node, 0.0), 4),
            "closeness_centrality":   round(clos.get(node, 0.0), 4),
            "eigenvector_centrality": round(eig.get(node, 0.0), 4),
            "node_type":              attrs.get("node_type", "unknown"),
            "risk_score":             attrs.get("risk_score", attrs.get("value", 0.0)),
        })
    df = pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)
    return df.reset_index(drop=True)


# ── Community detection ───────────────────────────────────────────────────────

def detect_communities(g: nx.Graph) -> Dict[str, int]:
    """
    Assigns community IDs to each node using greedy modularity optimisation.

    Returns {node_id: community_id}
    """
    if g.number_of_nodes() < 2:
        return {str(n): 0 for n in g.nodes()}

    # Greedy modularity (fast, no extra package needed)
    undirected = g.to_undirected() if g.is_directed() else g
    try:
        communities = nx.community.greedy_modularity_communities(undirected, weight="weight")
        mapping: Dict[str, int] = {}
        for cid, community in enumerate(communities):
            for node in community:
                mapping[str(node)] = cid
        return mapping
    except Exception:
        return {str(n): 0 for n in g.nodes()}


def community_summary(g: nx.Graph) -> pd.DataFrame:
    """Returns a DataFrame summarising each detected community."""
    comm_map = detect_communities(g)
    rows = []
    communities: Dict[int, List] = {}
    for node, cid in comm_map.items():
        communities.setdefault(cid, []).append(node)

    for cid, members in communities.items():
        sub = g.subgraph([n for n in g.nodes() if str(n) in members])
        risk_vals = [
            sub.nodes[n].get("risk_score", sub.nodes[n].get("value", 0.0))
            for n in sub.nodes()
        ]
        rows.append({
            "community_id":  cid,
            "n_members":     len(members),
            "members":       ", ".join(sorted(members)[:8]),
            "mean_risk":     round(sum(risk_vals) / len(risk_vals), 4) if risk_vals else 0.0,
            "max_risk":      round(max(risk_vals), 4) if risk_vals else 0.0,
            "density":       round(nx.density(sub), 4) if sub.number_of_nodes() > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("mean_risk", ascending=False).reset_index(drop=True)


# ── Risk propagation simulation ───────────────────────────────────────────────

def simulate_risk_propagation(
    g: nx.Graph,
    seed_nodes: List[str],
    beta: float = 0.15,
    decay: float = 0.05,
    steps: int = 10,
) -> pd.DataFrame:
    """
    Simulates how a risk signal spreads across the graph from *seed_nodes*,
    using a simplified SIR-inspired model:

      r_t+1(v) = (1 - decay) * r_t(v)
                 + beta * mean(r_t(u) for u in neighbours(v))

    Parameters
    ----------
    g          : knowledge graph (nodes must have a 'risk_score' attribute)
    seed_nodes : nodes where the initial risk originates
    beta       : transmission coefficient (0–1)
    decay      : per-step risk decay
    steps      : number of propagation steps

    Returns
    -------
    DataFrame with columns: step, node, risk_score
    """
    # Initialise risk scores
    risk: Dict[str, float] = {}
    for node in g.nodes():
        attrs = g.nodes[node]
        base = float(attrs.get("risk_score", attrs.get("value", 0.0)))
        risk[str(node)] = base

    # Seed nodes get a boost
    for sn in seed_nodes:
        if sn in risk:
            risk[sn] = min(1.0, risk[sn] + 0.5)

    records = [{"step": 0, "node": n, "risk_score": v} for n, v in risk.items()]

    for step in range(1, steps + 1):
        new_risk = {}
        for node in g.nodes():
            sn = str(node)
            neighbours = list(g.neighbors(node))
            if neighbours:
                nb_risk = sum(risk.get(str(nb), 0.0) for nb in neighbours) / len(neighbours)
            else:
                nb_risk = 0.0
            new_risk[sn] = min(1.0, (1 - decay) * risk.get(sn, 0.0) + beta * nb_risk)

        risk = new_risk
        records.extend({"step": step, "node": n, "risk_score": round(v, 4)} for n, v in risk.items())

    return pd.DataFrame(records)


# ── Graph summary ─────────────────────────────────────────────────────────────

def graph_summary(g: nx.Graph) -> Dict:
    """Returns a concise summary dict for display in a dashboard."""
    if g.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0}

    undirected = g.to_undirected() if g.is_directed() else g
    components = list(nx.connected_components(undirected))

    risk_vals = [
        g.nodes[n].get("risk_score", g.nodes[n].get("value", 0.0))
        for n in g.nodes()
    ]

    return {
        "n_nodes":           g.number_of_nodes(),
        "n_edges":           g.number_of_edges(),
        "density":           round(nx.density(g), 4),
        "n_components":      len(components),
        "largest_component": max(len(c) for c in components),
        "avg_degree":        round(sum(dict(g.degree()).values()) / g.number_of_nodes(), 2),
        "mean_risk":         round(sum(risk_vals) / len(risk_vals), 4) if risk_vals else 0.0,
        "max_risk":          round(max(risk_vals), 4) if risk_vals else 0.0,
    }


def get_high_risk_nodes(
    g: nx.Graph,
    top_n: int = 5,
    risk_attr: str = "risk_score",
) -> List[Tuple[str, float]]:
    """Returns the top-N nodes sorted by risk_attr descending."""
    scored = [
        (str(n), float(g.nodes[n].get(risk_attr, g.nodes[n].get("value", 0.0))))
        for n in g.nodes()
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
