"""Shared manifest contract for all source datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = (
    "segment_id",
    "dataset",
    "recording_id",
    "patient_id",
    "audio_path",
    "source_label",
    "label",
    "label_level",
    "seg_start_s",
    "seg_end_s",
    "device",
    "filter_mode",
    "site_code",
    "age_group",
    "split",
    "sha256",
    "orig_sr",
    "orig_channels",
    "orig_subtype",
    "duration_s",
    "qc_status",
    "qc_reason",
)


def make_segment_id(dataset: str, recording_id: str, start_s: float) -> str:
    """Return the stable dataset/recording/start identifier from the schema."""
    start_ms = int(round(start_s * 1000))
    return f"{dataset}__{recording_id}__{start_ms}"


def relative_audio_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def make_row(
    *,
    dataset: str,
    recording_id: str,
    patient_id: str,
    audio_path: Path,
    project_root: Path,
    source_label: str,
    label: str,
    label_level: str,
    seg_start_s: float,
    seg_end_s: float | None,
    device: str,
    filter_mode: str,
    site_code: str,
    age_group: str,
    split: str,
) -> dict[str, Any]:
    return {
        "segment_id": make_segment_id(dataset, recording_id, seg_start_s),
        "dataset": dataset,
        "recording_id": recording_id,
        "patient_id": patient_id,
        "audio_path": relative_audio_path(audio_path, project_root),
        "source_label": source_label,
        "label": label,
        "label_level": label_level,
        "seg_start_s": seg_start_s,
        "seg_end_s": seg_end_s,
        "device": device,
        "filter_mode": filter_mode,
        "site_code": site_code,
        "age_group": age_group,
        "split": split,
    }

