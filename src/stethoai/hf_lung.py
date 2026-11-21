"""Loader for HF_Lung_V1 respiratory-phase and adventitious events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import make_row


@dataclass(frozen=True)
class Event:
    kind: str
    start: float
    end: float


def _seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _recording_fields(recording_id: str) -> tuple[str, str, str]:
    if recording_id.startswith("steth_"):
        return recording_id.split("_")[1], "Littmann 3200", "unknown"
    if recording_id.startswith("trunc_"):
        encoded = recording_id[len("trunc_") :].rsplit("_", 1)[0]
        site_match = re.search(r"-(L\d+)$", encoded)
        site = site_match.group(1) if site_match else "unknown"
        return encoded[:10], "HF-Type-1", site
    raise ValueError(f"unrecognized HF_Lung filename: {recording_id}")


def _label(active: list[Event]) -> str:
    kinds = {event.kind for event in active}
    unknown = kinds - {"D", "Wheeze", "Rhonchi", "Stridor"}
    if unknown:
        raise ValueError(f"unrecognized HF_Lung event labels: {sorted(unknown)}")
    if "Stridor" in kinds:
        return "X"
    has_crackle = "D" in kinds
    has_continuous = bool(kinds & {"Wheeze", "Rhonchi"})
    if has_crackle and has_continuous:
        return "B"
    if has_crackle:
        return "C"
    if has_continuous:
        return "W"
    return "N"


def _parse_events(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        kind, start, end = line.split()
        events.append(Event(kind, _seconds(start), _seconds(end)))
    return events


def _merged_coverage(events: list[Event]) -> list[tuple[float, float]]:
    intervals = sorted((event.start, event.end) for event in events if event.end > event.start)
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def load_hf_lung(root: Path, project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for audio_path in sorted((root / split).glob("*.wav")):
            recording_id = audio_path.stem
            patient_id, device, site = _recording_fields(recording_id)
            events = _parse_events(audio_path.with_name(f"{recording_id}_label.txt"))
            phases = [event for event in events if event.kind in {"I", "E"}]
            adventitious = [event for event in events if event.kind not in {"I", "E"}]
            if not events:
                rows.append(
                    make_row(
                        dataset="hf_lung_v1",
                        recording_id=recording_id,
                        patient_id=patient_id,
                        audio_path=audio_path,
                        project_root=project_root,
                        source_label="",
                        label="N",
                        label_level="record",
                        seg_start_s=0.0,
                        seg_end_s=None,
                        device=device,
                        filter_mode="unknown",
                        site_code=site,
                        age_group="adult",
                        split=split,
                    )
                )
                continue
            coverage = _merged_coverage(phases if phases else adventitious)
            for coverage_start, coverage_end in coverage:
                overlaps = [
                    event
                    for event in adventitious
                    if event.start < coverage_end and event.end > coverage_start
                ]
                boundaries = {coverage_start, coverage_end}
                for event in overlaps:
                    boundaries.add(max(coverage_start, event.start))
                    boundaries.add(min(coverage_end, event.end))
                ordered = sorted(boundaries)
                for start, end in zip(ordered, ordered[1:]):
                    if end <= start:
                        continue
                    active = [
                        event for event in overlaps if event.start < end and event.end > start
                    ]
                    active_phases = sorted(
                        {
                            phase.kind
                            for phase in phases
                            if phase.start < end and phase.end > start
                        }
                    )
                    source = "+".join([*active_phases, *sorted({e.kind for e in active})])
                    rows.append(
                        make_row(
                            dataset="hf_lung_v1",
                            recording_id=recording_id,
                            patient_id=patient_id,
                            audio_path=audio_path,
                            project_root=project_root,
                            source_label=source,
                            label=_label(active),
                            label_level="event",
                            seg_start_s=start,
                            seg_end_s=end,
                            device=device,
                            filter_mode="unknown",
                            site_code=site,
                            age_group="adult",
                            split=split,
                        )
                    )
    return rows

