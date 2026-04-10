"""YouTube URL -> transcript/metadata -> privacy-safe text signals.

This module intentionally does NOT download video.

What it does
------------
- Accepts one or more YouTube video URLs.
- Fetches lightweight metadata via YouTube oEmbed.
- When available, fetches the public transcript using `youtube-transcript-api`.
- Scores each transcript segment using existing text+semantic signal code.
- Returns per-video summary rows and per-segment score rows.

Privacy / storage
-----------------
- By default, callers should NOT persist raw transcript text.
- This module returns scored frames; caller can drop `text` before saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from src.features.sentiment import analyze_dataframe, aggregate_signal
from src.features.semantic_signals import score_semantic_similarity, aggregate_semantic_signal

try:
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

    _YTA_AVAILABLE = True
except Exception:
    YouTubeTranscriptApi = None  # type: ignore[assignment]
    _YTA_AVAILABLE = False


@dataclass(frozen=True)
class YouTubeSignalsResult:
    per_video: pd.DataFrame
    per_segment: pd.DataFrame
    semantic_method: str
    status: Dict[str, object]


def parse_youtube_video_id(url: str) -> str:
    """Extract a YouTube video id from common URL forms."""
    u = urlparse(url.strip())
    host = (u.netloc or "").lower()
    path = (u.path or "").strip("/")

    if host.endswith("youtu.be"):
        # youtu.be/<id>
        vid = path.split("/")[0]
        return vid

    if "youtube.com" in host:
        # youtube.com/watch?v=<id>
        q = parse_qs(u.query or "")
        if "v" in q and q["v"]:
            return str(q["v"][0])
        # youtube.com/shorts/<id> or /embed/<id>
        parts = path.split("/")
        if parts and parts[0] in {"shorts", "embed"} and len(parts) > 1:
            return parts[1]

    raise ValueError("Could not parse YouTube video id from URL.")


def fetch_oembed_metadata(url: str) -> Dict[str, object]:
    """Fetch basic metadata via YouTube oEmbed."""
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=20,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _require_transcript_api() -> None:
    if not _YTA_AVAILABLE or YouTubeTranscriptApi is None:
        raise RuntimeError(
            "YouTube transcript support requires `youtube-transcript-api`. "
            "Install with `pip install youtube-transcript-api` (and re-run)."
        )


def fetch_transcript_segments(video_id: str, *, languages: Sequence[str]) -> List[Dict[str, object]]:
    """Fetch transcript segments for a YouTube video id.

    Returns list of dicts with keys: text, start, duration.
    """
    _require_transcript_api()
    assert YouTubeTranscriptApi is not None

    segments_obj: object
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        # Older youtube-transcript-api versions exposed a classmethod.
        segments_obj = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))  # type: ignore[attr-defined]
    else:
        # Newer youtube-transcript-api versions (>=1.x) use an instance API.
        api = YouTubeTranscriptApi()  # type: ignore[call-arg]
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=list(languages))  # type: ignore[attr-defined]
            segments_obj = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
        elif hasattr(api, "list"):
            transcript = api.list(video_id).find_transcript(list(languages))  # type: ignore[attr-defined]
            fetched = transcript.fetch(preserve_formatting=False)
            segments_obj = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
        else:
            raise RuntimeError(
                "Installed `youtube-transcript-api` does not provide a supported transcript fetch API. "
                "Please upgrade/downgrade the package."
            )

    segments = segments_obj or []

    # Normalize types + keep only keys we use.
    out: List[Dict[str, object]] = []
    for s in segments:  # type: ignore[assignment]
        if isinstance(s, dict):
            text = str(s.get("text", "") or "")
            start = float(s.get("start", 0.0) or 0.0)
            duration = float(s.get("duration", 0.0) or 0.0)
        else:
            text = str(getattr(s, "text", "") or "")
            start = float(getattr(s, "start", 0.0) or 0.0)
            duration = float(getattr(s, "duration", 0.0) or 0.0)

        out.append(
            {
                "text": text.replace("\n", " ").strip(),
                "start": start,
                "duration": duration,
            }
        )
    return out


def extract_youtube_transcript_signals(
    youtube_urls: Sequence[str],
    *,
    languages: Sequence[str] = ("en",),
    max_segments: int = 400,
    include_text: bool = False,
) -> YouTubeSignalsResult:
    """Extract privacy-safe text signals from one or more YouTube URLs."""

    urls = [u.strip() for u in youtube_urls if str(u or "").strip()]
    status: Dict[str, object] = {
        "n_urls": int(len(urls)),
        "languages": list(languages),
        "max_segments": int(max_segments),
        "transcript_api_available": bool(_YTA_AVAILABLE),
        "errors": [],
    }

    per_video_rows: List[Dict[str, object]] = []
    segment_frames: List[pd.DataFrame] = []
    semantic_method = ""

    for url in urls:
        row: Dict[str, object] = {
            "url": url,
            "video_id": "",
            "title": "",
            "author_name": "",
            "author_url": "",
            "transcript_available": False,
            "n_segments": 0,
        }

        try:
            vid = parse_youtube_video_id(url)
            row["video_id"] = vid

            meta = fetch_oembed_metadata(url)
            row["title"] = str(meta.get("title", "") or "")
            row["author_name"] = str(meta.get("author_name", "") or "")
            row["author_url"] = str(meta.get("author_url", "") or "")

            segments = fetch_transcript_segments(vid, languages=languages)
            if max_segments > 0:
                segments = segments[: int(max_segments)]

            row["transcript_available"] = bool(segments)
            row["n_segments"] = int(len(segments))

            if not segments:
                per_video_rows.append(row)
                continue

            seg_df = pd.DataFrame(segments)
            seg_df.insert(0, "url", url)
            seg_df.insert(1, "video_id", vid)
            seg_df = seg_df.reset_index().rename(columns={"index": "segment_index"})

            # Score lexicon-based text signals.
            scored = analyze_dataframe(seg_df, text_col="text", include_matches=False)

            # Score semantic similarity signals.
            sem_df, method = score_semantic_similarity(scored["text"].fillna("").tolist())
            semantic_method = semantic_method or method
            scored = pd.concat([scored.reset_index(drop=True), sem_df.reset_index(drop=True)], axis=1)

            # Summaries for this video.
            lex_sum = aggregate_signal(scored)
            sem_sum = aggregate_semantic_signal(scored)

            row.update({f"lex_{k}": float(v) for k, v in lex_sum.items()})
            row.update({f"sem_{k}": float(v) for k, v in sem_sum.items()})
            row["semantic_method"] = method

            if not include_text and "text" in scored.columns:
                scored = scored.drop(columns=["text"])

            per_video_rows.append(row)
            segment_frames.append(scored)

        except Exception as exc:
            errs = status.get("errors")
            if isinstance(errs, list):
                errs.append({"url": url, "error": str(exc)})
            per_video_rows.append({**row, "error": str(exc)})

    per_video_df = pd.DataFrame(per_video_rows)
    per_segment_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()

    return YouTubeSignalsResult(
        per_video=per_video_df,
        per_segment=per_segment_df,
        semantic_method=semantic_method,
        status=status,
    )
