"""Video → structured behavioral signals (skeleton).

Mission fit
-----------
- Uses single or multiple videos as input.
- Detects behavior patterns, anomalies, and environmental context.
- Converts video into structured signals (activity trend, anomaly score, etc.).
- Works at non-identifiable, population level.

Design choices
--------------
- Avoids any biometric identification (no faces, no re-identification, no tracking IDs).
- Operates on coarse frame-difference / motion energy statistics.
- Exports only aggregated window-level metrics.

Dependencies
------------
- Optional: `opencv-python` for video decoding.
  If OpenCV is not available, this module raises a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False


@dataclass
class VideoWindowSignals:
    window_index: int
    start_sec: float
    end_sec: float
    activity_mean: float
    activity_std: float
    anomaly_score: float
    low_light_frac: float
    scene_change_rate: float


def _require_cv2() -> None:
    if not _CV2_AVAILABLE or cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for video processing. Install with `pip install opencv-python`."
        )


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[arg-type]


def extract_video_signals(
    video_path: str | Path,
    *,
    sample_fps: float = 2.0,
    window_sec: float = 10.0,
    resize_width: int = 320,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Extract window-level behavioral signals from a single video.

    Signals (per window)
    --------------------
    - activity_mean/std: mean/std of normalized frame-difference magnitude
    - anomaly_score: activity z-score clamped to [0, 1] (per window)
    - low_light_frac: fraction of frames in window below brightness threshold
    - scene_change_rate: fraction of large frame changes in window

    Returns
    -------
    (window_df, summary_dict)
    """
    _require_cv2()

    p = Path(video_path)
    cap = cv2.VideoCapture(str(p))  # type: ignore[attr-defined]
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {p}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)  # type: ignore[attr-defined]
    if native_fps <= 0:
        native_fps = 30.0

    step = max(int(round(native_fps / max(sample_fps, 0.1))), 1)
    brightness_thresh = 35.0
    scene_thresh = 18.0

    # Streaming accumulation
    diffs: List[float] = []
    brightness: List[float] = []
    big_changes: List[int] = []

    rows: List[VideoWindowSignals] = []

    prev_gray: Optional[np.ndarray] = None
    idx = 0
    sampled = 0

    def flush_window(window_index: int, start_sec: float, end_sec: float) -> None:
        if not diffs:
            return
        a = np.asarray(diffs, dtype=float)
        b = np.asarray(brightness, dtype=float)
        bc = np.asarray(big_changes, dtype=float) if big_changes else np.zeros(len(a))

        activity_mean = float(np.clip(a.mean(), 0.0, 1.0))
        activity_std = float(np.clip(a.std(), 0.0, 1.0))
        low_light_frac = float(np.mean(b < brightness_thresh)) if len(b) else 0.0
        scene_change_rate = float(np.mean(bc)) if len(bc) else 0.0

        # provisional anomaly score computed later; put placeholder
        rows.append(
            VideoWindowSignals(
                window_index=window_index,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                activity_mean=activity_mean,
                activity_std=activity_std,
                anomaly_score=0.0,
                low_light_frac=low_light_frac,
                scene_change_rate=scene_change_rate,
            )
        )

    # Track window boundaries in sampled-time
    window_index = 0
    window_start_sample = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step != 0:
            idx += 1
            continue

        # Resize for speed and to reduce identifying detail
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / float(w)
            frame = cv2.resize(frame, (resize_width, int(h * scale)))  # type: ignore[attr-defined]

        gray = _to_gray(frame)
        gray_f = gray.astype(np.float32)

        # brightness proxy
        brightness.append(float(gray_f.mean()))

        if prev_gray is None:
            prev_gray = gray
            diffs.append(0.0)
            big_changes.append(0)
        else:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float32)  # type: ignore[attr-defined]
            prev_gray = gray
            # Normalized motion energy proxy
            motion = float(np.clip(diff.mean() / 255.0, 0.0, 1.0))
            diffs.append(motion)
            big_changes.append(1 if (diff.mean() > scene_thresh) else 0)

        sampled += 1

        # Window flush condition in terms of sampled seconds
        sampled_time_sec = sampled / max(sample_fps, 0.1)
        if sampled_time_sec >= (window_index + 1) * window_sec:
            start_sec = window_index * window_sec
            end_sec = (window_index + 1) * window_sec
            flush_window(window_index, start_sec, end_sec)
            window_index += 1
            diffs.clear()
            brightness.clear()
            big_changes.clear()

        idx += 1

    cap.release()

    if diffs:
        start_sec = window_index * window_sec
        end_sec = (window_index + 1) * window_sec
        flush_window(window_index, start_sec, end_sec)

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        summary = {
            "video_activity_mean": 0.0,
            "video_anomaly_score": 0.0,
            "video_low_light_frac": 0.0,
            "video_scene_change_rate": 0.0,
            "n_windows": 0,
        }
        return df, summary

    # Window-level anomaly: z-score on activity_mean
    mu = float(df["activity_mean"].mean())
    sigma = float(df["activity_mean"].std() or 0.0)
    if sigma <= 1e-9:
        df["anomaly_score"] = 0.0
    else:
        z = (df["activity_mean"] - mu).abs() / sigma
        df["anomaly_score"] = np.clip(z / 4.0, 0.0, 1.0).round(4)

    summary = {
        "video_activity_mean": float(np.clip(df["activity_mean"].mean(), 0.0, 1.0)),
        "video_anomaly_score": float(np.clip(df["anomaly_score"].mean(), 0.0, 1.0)),
        "video_low_light_frac": float(np.clip(df["low_light_frac"].mean(), 0.0, 1.0)),
        "video_scene_change_rate": float(np.clip(df["scene_change_rate"].mean(), 0.0, 2.0)),
        "n_windows": int(len(df)),
    }
    return df, summary


def extract_multi_video_signals(
    video_paths: Sequence[str | Path],
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Aggregate signals across multiple videos.

    Returns a concatenated window DataFrame with a `video_name` column plus a
    simple summary (mean across videos).
    """
    frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    for vp in video_paths:
        df, summ = extract_video_signals(vp, **kwargs)
        df = df.copy()
        df.insert(0, "video_name", Path(vp).name)
        frames.append(df)
        summaries.append(summ)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not summaries:
        return all_df, {
            "video_activity_mean": 0.0,
            "video_anomaly_score": 0.0,
            "video_low_light_frac": 0.0,
            "video_scene_change_rate": 0.0,
            "n_windows": 0,
            "n_videos": 0,
        }

    # Mean-of-means aggregation
    def mean_key(k: str) -> float:
        vals = [float(s.get(k, 0.0) or 0.0) for s in summaries]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "video_activity_mean": mean_key("video_activity_mean"),
        "video_anomaly_score": mean_key("video_anomaly_score"),
        "video_low_light_frac": mean_key("video_low_light_frac"),
        "video_scene_change_rate": mean_key("video_scene_change_rate"),
        "n_windows": int(sum(int(s.get("n_windows", 0) or 0) for s in summaries)),
        "n_videos": int(len(summaries)),
    }
    return all_df, summary
