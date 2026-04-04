"""
Advanced sentiment & text-signal analysis
──────────────────────────────────────────
Layers:
  1. VADER compound sentiment score
  2. Substance-specific term density  (extended lexicon)
  3. Distress-signal term density     (extended lexicon)
  4. Urgency score                    (crisis language)
  5. Composite risk-text score        (weighted aggregate)

Falls back gracefully when vaderSentiment is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# ── VADER ─────────────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except Exception:
    _vader = None
    _VADER_AVAILABLE = False

# ── Extended lexicons ─────────────────────────────────────────────────────────
SUBSTANCE_TERMS: Dict[str, float] = {
    # weight = signal strength (0.5 = moderate, 1.0 = strong)
    "fentanyl": 1.0, "carfentanil": 1.0, "heroin": 1.0, "meth": 0.9,
    "methamphetamine": 1.0, "cocaine": 0.9, "crack": 0.8, "oxy": 0.8,
    "oxycontin": 0.9, "opioid": 0.9, "opioids": 0.9, "xanax": 0.7,
    "percocet": 0.7, "vicodin": 0.7, "benzodiazepine": 0.8, "benzo": 0.7,
    "overdose": 1.0, "od": 0.7, "narcotics": 0.8, "substance abuse": 0.9,
    "withdrawal": 0.8, "detox": 0.7, "narcan": 0.8, "naloxone": 0.9,
    "relapse": 0.9, "using again": 0.9, "got high": 0.7, "shooting up": 1.0,
    "snorting": 0.8, "fix": 0.6, "dope": 0.7, "stash": 0.6, "supply": 0.4,
    "tolerance": 0.5, "dependence": 0.7, "addiction": 0.9, "addict": 0.8,
}

DISTRESS_TERMS: Dict[str, float] = {
    "anxious": 0.5, "anxiety": 0.6, "depressed": 0.7, "depression": 0.7,
    "hopeless": 0.9, "hopelessness": 0.9, "suicidal": 1.0, "suicide": 1.0,
    "panic": 0.7, "lonely": 0.5, "isolated": 0.6, "craving": 0.8,
    "desperate": 0.8, "pain": 0.4, "suffering": 0.7, "helpless": 0.8,
    "struggling": 0.6, "can't stop": 0.8, "out of control": 0.9,
    "hit rock bottom": 0.9, "rock bottom": 0.9, "need help": 0.8,
    "want to die": 1.0, "no way out": 0.9, "giving up": 0.8,
    "lost everything": 0.8, "homeless": 0.6, "broke": 0.3,
}

URGENCY_TERMS: Dict[str, float] = {
    "emergency": 1.0, "911": 1.0, "ems": 0.9, "overdosing": 1.0,
    "not breathing": 1.0, "blue lips": 1.0, "unresponsive": 1.0,
    "passed out": 0.9, "dying": 1.0, "death": 0.8, "died": 0.9,
    "found dead": 1.0, "narcan needed": 1.0, "call 911": 1.0,
    "cpr": 0.9, "ambulance": 0.9, "hospital": 0.6, "er visit": 0.7,
}


@dataclass
class SentimentResult:
    vader_compound: float = 0.0          # -1 (negative) to +1 (positive)
    substance_score: float = 0.0         # 0–1 weighted term density
    distress_score: float = 0.0          # 0–1 weighted term density
    urgency_score: float = 0.0           # 0–1 weighted term density
    composite_risk: float = 0.0          # 0–1 aggregate risk-text score
    n_tokens: int = 0
    matched_substance_terms: List[str] = field(default_factory=list)
    matched_distress_terms: List[str] = field(default_factory=list)
    matched_urgency_terms: List[str] = field(default_factory=list)


def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    return re.findall(r"[a-z']+", text.lower())


def _score_lexicon(tokens: List[str], text: str, lexicon: Dict[str, float]) -> tuple[float, List[str]]:
    """
    Matches single-word tokens AND multi-word phrases in *text*.
    Returns (weighted_density, matched_terms).
    """
    token_set = set(tokens)
    total_weight = 0.0
    matched: List[str] = []

    for term, weight in lexicon.items():
        if " " in term:                          # multi-word phrase match
            if term in text:
                total_weight += weight
                matched.append(term)
        else:                                    # single-word token match
            if term in token_set:
                total_weight += weight
                matched.append(term)

    n = max(len(tokens), 1)
    density = min(total_weight / n, 1.0)
    return density, matched


def analyze(text: str) -> SentimentResult:
    """
    Full sentiment + signal analysis of a single text string.
    """
    clean = (text or "").strip()
    tokens = _tokenize(clean)
    n = len(tokens)

    # 1. VADER
    vader_compound = 0.0
    if _VADER_AVAILABLE and _vader is not None and clean:
        vader_compound = _vader.polarity_scores(clean)["compound"]

    # 2. Lexicon scores
    sub_score, sub_terms = _score_lexicon(tokens, clean.lower(), SUBSTANCE_TERMS)
    dis_score, dis_terms = _score_lexicon(tokens, clean.lower(), DISTRESS_TERMS)
    urg_score, urg_terms = _score_lexicon(tokens, clean.lower(), URGENCY_TERMS)

    # 3. Composite risk-text score
    # Very negative VADER + high substance + high distress → high risk
    # vader_compound is typically negative for distress text → invert sign
    vader_risk = max(-vader_compound, 0.0)   # 0 (positive) to 1 (very negative)
    composite = (
        0.30 * vader_risk
        + 0.30 * sub_score
        + 0.25 * dis_score
        + 0.15 * urg_score
    )
    composite = min(composite, 1.0)

    return SentimentResult(
        vader_compound=vader_compound,
        substance_score=sub_score,
        distress_score=dis_score,
        urgency_score=urg_score,
        composite_risk=composite,
        n_tokens=n,
        matched_substance_terms=sub_terms,
        matched_distress_terms=dis_terms,
        matched_urgency_terms=urg_terms,
    )


def analyze_dataframe(
    df: pd.DataFrame,
    text_col: str,
    prefix: str = "",
    include_matches: bool = False,
) -> pd.DataFrame:
    """
    Applies *analyze()* to every row of a DataFrame and appends result columns.

    New columns added (all prefixed with *prefix* if provided):
      vader_compound, substance_score, distress_score, urgency_score,
      composite_risk, n_tokens
    """
    out = df.copy()
    results = out[text_col].fillna("").apply(analyze)

    col = (prefix + "{}") if prefix else "{}"
    out[col.format("vader_compound")]  = results.apply(lambda r: r.vader_compound)
    out[col.format("substance_score")] = results.apply(lambda r: r.substance_score)
    out[col.format("distress_score")]  = results.apply(lambda r: r.distress_score)
    out[col.format("urgency_score")]   = results.apply(lambda r: r.urgency_score)
    out[col.format("composite_risk")]  = results.apply(lambda r: r.composite_risk)
    out[col.format("n_tokens")]        = results.apply(lambda r: r.n_tokens)

    if include_matches:
        # Store matched lexicon terms as a lightweight evidence trail.
        # Avoids raw text persistence while still supporting explainability.
        out[col.format("matched_substance_terms")] = results.apply(lambda r: ", ".join(r.matched_substance_terms))
        out[col.format("matched_distress_terms")] = results.apply(lambda r: ", ".join(r.matched_distress_terms))
        out[col.format("matched_urgency_terms")] = results.apply(lambda r: ", ".join(r.matched_urgency_terms))
    return out


def aggregate_signal(df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """
    Returns a summary dict with mean signal values from a text-scored DataFrame.
    Useful for feeding into the fusion / EWS layer.
    """
    col = (prefix + "{}") if prefix else "{}"
    keys = ["vader_compound", "substance_score", "distress_score",
            "urgency_score", "composite_risk"]
    result: Dict[str, float] = {}
    for k in keys:
        c = col.format(k)
        result[k] = float(df[c].mean()) if c in df.columns and not df.empty else 0.0
    return result
