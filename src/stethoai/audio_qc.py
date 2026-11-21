"""Recording-level audio metadata and conservative ingestion QC flags."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


CLIPPING_ABS = 0.999
CLIPPING_FRACTION = 0.01
SILENCE_RMS = 1e-4
MIN_DURATION_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_audio(path: Path) -> dict[str, Any]:
    """Inspect one audio file without resampling or modifying source data."""
    with sf.SoundFile(path) as audio:
        sample_count = 0
        clipped_count = 0
        sum_squares = 0.0
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            sample_count += block.size
            clipped_count += int(np.count_nonzero(np.abs(block) >= CLIPPING_ABS))
            sum_squares += float(np.sum(np.square(block, dtype=np.float64)))
        duration_s = audio.frames / audio.samplerate
        rms = (sum_squares / sample_count) ** 0.5 if sample_count else 0.0
        clipping_fraction = clipped_count / sample_count if sample_count else 0.0
        reasons: list[str] = []
        if clipping_fraction >= CLIPPING_FRACTION:
            reasons.append("clipping")
        if rms <= SILENCE_RMS:
            reasons.append("silence")
        if duration_s < MIN_DURATION_S:
            reasons.append("too_short")
        return {
            "sha256": _sha256(path),
            "orig_sr": audio.samplerate,
            "orig_channels": audio.channels,
            "orig_subtype": audio.subtype,
            "duration_s": round(duration_s, 9),
            "qc_status": "fail" if reasons else "pass",
            "qc_reason": ";".join(reasons),
        }

