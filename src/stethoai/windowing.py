"""Fixed half-hop windows and explicit annotation-overlap labels (D-033)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_interval(segments: pd.DataFrame, start: float, end: float) -> str | None:
    """Positive overlap only, including record-level labels over their recorded span.

    No overlap = unknown (excluded). Q/X have precedence. Unannotated time is
    ignored, never independently used as evidence of Normal.
    """
    hit = segments.loc[(segments.seg_start_s < end) & (segments.seg_end_s > start)]
    labels = set(hit.label)
    if not labels or labels & {"Q", "X"}:
        return None
    crackle = bool(labels & {"C", "B"})
    wheeze = bool(labels & {"W", "B"})
    return "B" if crackle and wheeze else "C" if crackle else "W" if wheeze else "N"


def window_bounds(samples: int, seconds: int, sr: int = 4000) -> list[tuple[int, int]]:
    width = int(seconds * sr)
    hop = width // 2
    assert samples > 0 and width > 0 and hop > 0
    starts = list(range(0, max(samples - width, 0) + 1, hop))
    if starts[-1] + width < samples:
        starts.append(starts[-1] + hop)
    return [(start, min(start + width, samples)) for start in starts]


def make_windows(signal: np.ndarray, segments: pd.DataFrame, seconds: int):
    windows = []
    for start, end in window_bounds(len(signal), seconds):
        label = label_interval(segments, start / 4000, end / 4000)
        if label is None:
            continue
        padded = np.zeros(seconds * 4000, dtype=np.float32)
        padded[:end-start] = signal[start:end]
        windows.append((start / 4000, end / 4000, label, padded))
    return windows
