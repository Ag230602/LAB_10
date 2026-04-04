"""Semantic / embedding-style text signals.

Goal
----
Provide an embedding-based (or embedding-like) alternative to pure lexicon rules.
This module intentionally avoids any user identification and works at an
aggregate/population level.

Implementation
--------------
- If `sentence_transformers` is installed, uses SBERT embeddings + cosine similarity.
- Otherwise falls back to TF-IDF vectors (sklearn) with the same cosine similarity API.

Outputs are continuous scores in [0, 1] for:
- `semantic_substance_score`
- `semantic_distress_score`
- `semantic_help_seeking_score`
- `semantic_composite_risk`

This is not a supervised classifier by default (no labels required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SemanticSeedSet:
    substance: List[str]
    distress: List[str]
    help_seeking: List[str]


DEFAULT_SEEDS = SemanticSeedSet(
    substance=[
        "using fentanyl",
        "opioid withdrawal",
        "relapse after detox",
        "overdose risk",
        "drinking too much alcohol",
        "meth cravings",
    ],
    distress=[
        "I feel hopeless",
        "panic and anxiety",
        "can't cope anymore",
        "I want to give up",
        "I'm depressed",
    ],
    help_seeking=[
        "I need help",
        "how to get treatment",
        "rehab options",
        "naloxone / narcan access",
        "support group",
    ],
)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a @ b.T) / (a_norm * b_norm.T)


class _Embedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class _SBERTEmbedder(_Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


class _TfidfEmbedder(_Embedder):
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=25_000,
        )
        self._fit = False

    def fit(self, corpus: Sequence[str]) -> None:
        self._vectorizer.fit(list(corpus))
        self._fit = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fit:
            self.fit(texts)
        mat = self._vectorizer.transform(list(texts))
        return mat.toarray()


def build_embedder(prefer_sbert: bool = True) -> Tuple[_Embedder, str]:
    """Returns (embedder, method_name)."""
    if prefer_sbert:
        try:
            return _SBERTEmbedder(), "sbert"
        except Exception:
            pass
    return _TfidfEmbedder(), "tfidf"


def score_semantic_similarity(
    texts: Sequence[str],
    seeds: Optional[SemanticSeedSet] = None,
    prefer_sbert: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Compute semantic similarity to seed phrase sets.

    For each text, score is the max cosine similarity to any seed in the set,
    rescaled into [0, 1].
    """
    seed_set = seeds or DEFAULT_SEEDS
    clean_texts = [str(t or "") for t in texts]

    embedder, method = build_embedder(prefer_sbert=prefer_sbert)

    # Fit TF-IDF on union corpus to keep the space consistent.
    if isinstance(embedder, _TfidfEmbedder):
        embedder.fit(clean_texts + seed_set.substance + seed_set.distress + seed_set.help_seeking)

    text_vecs = embedder.encode(clean_texts)

    def _max_sim(seed_phrases: List[str]) -> np.ndarray:
        seed_vecs = embedder.encode(seed_phrases)
        sims = _cosine_sim_matrix(text_vecs, seed_vecs)
        out = sims.max(axis=1) if sims.size else np.zeros(len(clean_texts))
        # cosine similarity can be [-1,1] in TF-IDF; clamp to [0,1]
        return np.clip(out, 0.0, 1.0)

    sub = _max_sim(seed_set.substance)
    dis = _max_sim(seed_set.distress)
    help_ = _max_sim(seed_set.help_seeking)

    composite = np.clip(0.40 * sub + 0.35 * dis + 0.25 * help_, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "semantic_substance_score": np.round(sub, 4),
            "semantic_distress_score": np.round(dis, 4),
            "semantic_help_seeking_score": np.round(help_, 4),
            "semantic_composite_risk": np.round(composite, 4),
        }
    )
    return df, method


def aggregate_semantic_signal(df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate semantic signals into means for fusion/EWS."""
    keys = [
        "semantic_substance_score",
        "semantic_distress_score",
        "semantic_help_seeking_score",
        "semantic_composite_risk",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(pd.to_numeric(df.get(k, pd.Series(dtype=float)), errors="coerce").mean()) if not df.empty else 0.0
    return out
