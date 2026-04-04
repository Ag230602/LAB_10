from __future__ import annotations

import re
from dataclasses import dataclass
import pandas as pd

SUBSTANCE_TERMS = {
    "opioid", "fentanyl", "heroin", "meth", "methamphetamine", "alcohol",
    "cocaine", "relapse", "withdrawal", "overdose", "oxy", "xanax"
}

DISTRESS_TERMS = {
    "anxious", "anxiety", "depressed", "hopeless", "panic", "lonely",
    "relapse", "craving", "desperate", "withdrawal", "pain"
}


@dataclass
class TextSignalResult:
    substance_mentions: int
    distress_mentions: int
    text_length: int


def extract_text_signals(text: str) -> TextSignalResult:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    substance_count = sum(token in SUBSTANCE_TERMS for token in tokens)
    distress_count = sum(token in DISTRESS_TERMS for token in tokens)
    return TextSignalResult(
        substance_mentions=int(substance_count),
        distress_mentions=int(distress_count),
        text_length=len(tokens),
    )


def add_text_signal_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    signals = out[text_col].fillna("").apply(extract_text_signals)
    out["substance_mentions"] = signals.apply(lambda x: x.substance_mentions)
    out["distress_mentions"] = signals.apply(lambda x: x.distress_mentions)
    out["text_length"] = signals.apply(lambda x: x.text_length)
    return out
