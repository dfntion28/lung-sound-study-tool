"""Loader for ICBHI 2017 cycle annotations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import make_row


LABELS = {("0", "0"): "N", ("1", "0"): "C", ("0", "1"): "W", ("1", "1"): "B"}
DEVICES = {
    "AKGC417L": "AKG C417L",
    "LittC2SE": "Littmann Classic II SE",
    "Litt3200": "Littmann 3200",
    "Meditron": "WelchAllyn Meditron",
}


def _age_groups(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            age = float(parts[1])
        except ValueError:
            result[parts[0]] = "unknown"
        else:
            result[parts[0]] = "pediatric" if age < 18 else "adult"
    return result


def _split_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_recording: dict[str, str] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        recording_id, split = line.split()
        patient_id = recording_id.split("_", 1)[0]
        source_splits[patient_id].add(split)
        by_recording[recording_id] = split
    by_patient = {
        patient_id: "test" if "test" in splits else "train"
        for patient_id, splits in source_splits.items()
    }
    return by_recording, by_patient


def load_icbhi(root: Path, project_root: Path) -> list[dict[str, Any]]:
    database = root / "ICBHI_final_database"
    by_recording, by_patient = _split_maps(root / "ICBHI_challenge_train_test.txt")
    ages = _age_groups(root / "ICBHI_Challenge_demographic_information.txt")
    rows: list[dict[str, Any]] = []
    for audio_path in sorted(database.glob("*.wav")):
        recording_id = audio_path.stem
        parts = recording_id.split("_")
        patient_id = parts[0]
        split = by_patient.get(patient_id)
        if split is None:
            raise ValueError(f"ICBHI recording has no official split: {recording_id}")
        annotation_path = database / f"{recording_id}.txt"
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            start, end, crackle, wheeze = line.split()
            source_label = f"crackles={crackle},wheezes={wheeze}"
            rows.append(
                make_row(
                    dataset="icbhi2017",
                    recording_id=recording_id,
                    patient_id=patient_id,
                    audio_path=audio_path,
                    project_root=project_root,
                    source_label=source_label,
                    label=LABELS[(crackle, wheeze)],
                    label_level="cycle",
                    seg_start_s=float(start),
                    seg_end_s=float(end),
                    device=DEVICES.get(parts[-1], parts[-1]),
                    filter_mode="unknown",
                    site_code=parts[2],
                    age_group=ages.get(patient_id, "unknown"),
                    split=split,
                )
            )
    return rows

