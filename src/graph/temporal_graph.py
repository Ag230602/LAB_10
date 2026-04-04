"""
Temporal Knowledge Graph
─────────────────────────
Maintains a sequence of time-stamped NetworkX graph snapshots.
Each snapshot captures the risk state at a given point in time.

Key capabilities
────────────────
  • add_snapshot(ts, graph)          – append a time-indexed graph
  • get_evolution()                  – DataFrame of node attributes over time
  • compute_edge_weight_changes()    – per-edge weight trajectory
  • graph_diff(t1, t2)               – structural diff between two snapshots
  • export_animation_frames()        – list of (ts, edge_df) for animation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd


@dataclass
class GraphSnapshot:
    timestamp: datetime
    graph: nx.DiGraph
    metadata: Dict = field(default_factory=dict)


class TemporalGraph:
    """
    Ordered collection of directed graph snapshots.

    Each snapshot represents the knowledge graph at one time step
    (e.g., one month of CDC overdose data + social signals).
    """

    def __init__(self) -> None:
        self._snapshots: List[GraphSnapshot] = []

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_snapshot(
        self,
        timestamp: datetime,
        graph: nx.DiGraph,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Append a new time-indexed snapshot (maintains time-sorted order)."""
        snap = GraphSnapshot(
            timestamp=timestamp,
            graph=graph.copy(),
            metadata=metadata or {},
        )
        self._snapshots.append(snap)
        self._snapshots.sort(key=lambda s: s.timestamp)

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def timestamps(self) -> List[datetime]:
        return [s.timestamp for s in self._snapshots]

    def get_snapshot(self, index: int) -> Optional[GraphSnapshot]:
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None

    def latest(self) -> Optional[GraphSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    # ── Analysis ──────────────────────────────────────────────────────────────

    def get_node_attribute_evolution(self, node_id: str, attr: str) -> pd.DataFrame:
        """
        Returns a time-series DataFrame of a single node attribute across all
        snapshots.

        Columns: timestamp, value
        """
        rows = []
        for snap in self._snapshots:
            val = snap.graph.nodes.get(node_id, {}).get(attr, None)
            rows.append({"timestamp": snap.timestamp, "value": val})
        return pd.DataFrame(rows)

    def get_evolution_summary(self) -> pd.DataFrame:
        """
        Returns a tidy DataFrame summarising key graph statistics for each snapshot.

        Columns: timestamp, n_nodes, n_edges, density,
                 mean_ews, max_ews, mean_risk_score
        """
        rows = []
        for snap in self._snapshots:
            g = snap.graph
            n = g.number_of_nodes()
            e = g.number_of_edges()
            density = nx.density(g) if n > 1 else 0.0

            ews_values = [
                d.get("ews", d.get("value", 0.0))
                for _, d in g.nodes(data=True)
                if d.get("node_type") in ("risk_score", "state")
            ]
            mean_ews = sum(ews_values) / len(ews_values) if ews_values else 0.0
            max_ews  = max(ews_values) if ews_values else 0.0

            risk_values = [
                d.get("risk_score", 0.0)
                for _, d in g.nodes(data=True)
                if "risk_score" in d
            ]
            mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0

            rows.append({
                "timestamp":       snap.timestamp,
                "n_nodes":         n,
                "n_edges":         e,
                "density":         round(density, 4),
                "mean_ews":        round(mean_ews, 4),
                "max_ews":         round(max_ews, 4),
                "mean_risk_score": round(mean_risk, 4),
                **snap.metadata,
            })
        return pd.DataFrame(rows)

    def compute_edge_weight_changes(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-edge weight trajectories.
        Columns: source, target, timestamp, weight
        """
        rows = []
        for snap in self._snapshots:
            for u, v, data in snap.graph.edges(data=True):
                rows.append({
                    "source":    u,
                    "target":    v,
                    "timestamp": snap.timestamp,
                    "weight":    float(data.get("weight", 1.0)),
                    "relation":  data.get("relation", "related_to"),
                })
        return pd.DataFrame(rows)

    def graph_diff(self, idx_a: int, idx_b: int) -> Dict:
        """
        Structural diff between two snapshots at *idx_a* and *idx_b*.

        Returns a dict with:
          new_nodes, removed_nodes, new_edges, removed_edges,
          edge_weight_changes
        """
        snap_a = self.get_snapshot(idx_a)
        snap_b = self.get_snapshot(idx_b)
        if snap_a is None or snap_b is None:
            return {}

        nodes_a = set(snap_a.graph.nodes())
        nodes_b = set(snap_b.graph.nodes())
        edges_a = set((u, v) for u, v in snap_a.graph.edges())
        edges_b = set((u, v) for u, v in snap_b.graph.edges())

        weight_changes = []
        common_edges = edges_a & edges_b
        for u, v in common_edges:
            w_a = snap_a.graph.edges[u, v].get("weight", 1.0)
            w_b = snap_b.graph.edges[u, v].get("weight", 1.0)
            if abs(w_b - w_a) > 1e-6:
                weight_changes.append({"edge": (u, v), "old": w_a, "new": w_b, "delta": w_b - w_a})

        return {
            "new_nodes":          list(nodes_b - nodes_a),
            "removed_nodes":      list(nodes_a - nodes_b),
            "new_edges":          list(edges_b - edges_a),
            "removed_edges":      list(edges_a - edges_b),
            "edge_weight_changes": weight_changes,
        }

    def export_animation_frames(self) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Returns a list of (timestamp, edge_DataFrame) tuples suitable for
        building an animated graph visualisation.
        """
        frames = []
        for snap in self._snapshots:
            rows = [
                {
                    "source":   u,
                    "target":   v,
                    "weight":   data.get("weight", 1.0),
                    "relation": data.get("relation", "related_to"),
                }
                for u, v, data in snap.graph.edges(data=True)
            ]
            frames.append((snap.timestamp, pd.DataFrame(rows)))
        return frames

    def to_flat_dataframe(self) -> pd.DataFrame:
        """
        Flattens all snapshots into a single long-format DataFrame.
        Columns: timestamp, node, node_type, attribute_key, attribute_value
        """
        rows = []
        for snap in self._snapshots:
            for node, attrs in snap.graph.nodes(data=True):
                for k, v in attrs.items():
                    rows.append({
                        "timestamp":  snap.timestamp,
                        "node":       str(node),
                        "node_type":  attrs.get("node_type", "unknown"),
                        "attr_key":   k,
                        "attr_value": v,
                    })
        return pd.DataFrame(rows)
