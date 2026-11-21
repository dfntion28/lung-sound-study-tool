"""Loader for Fraiwan 2021 filter-specific record WAVs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from .manifest import make_row


FILTERS = {"B": "bell", "D": "diaphragm", "E": "extended"}


def _label(source_label: str) -> str:
    if source_label == "N":
        return "N"
    if source_label == "Bronchial":
        return "X"
    has_crackle = " C" in f" {source_label}" or source_label in {"C", "Crep"}
    has_wheeze = "W" in source_label
    if has_crackle and has_wheeze:
        return "B"
    if has_crackle:
        return "C"
    if has_wheeze:
        return "W"
    raise ValueError(f"unrecognized Fraiwan sound type: {source_label}")


def _annotations(path: Path) -> dict[int, tuple[float, str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows = [
            row
            for row in worksheet.iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        ]
        return {
            index: (float(row[0]), str(row[2]).strip(), str(row[3]).strip())
            for index, row in enumerate(rows, 1)
        }
    finally:
        workbook.close()


def load_fraiwan(root: Path, project_root: Path) -> list[dict[str, Any]]:
    annotations = _annotations(root / "Data annotation.xlsx")
    rows: list[dict[str, Any]] = []
    for audio_path in sorted(root.glob("*.wav")):
        match = re.match(r"([BDE])_?P(\d+)$", audio_path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"unrecognized Fraiwan filename: {audio_path.name}")
        filter_code, patient_number_text = match.groups()
        patient_number = int(patient_number_text)
        if patient_number not in annotations:
            raise ValueError(f"Fraiwan P{patient_number} has no annotation row")
        age, site, source_label = annotations[patient_number]
        rows.append(
            make_row(
                dataset="fraiwan2021",
                recording_id=audio_path.stem,
                patient_id=f"P{patient_number}",
                audio_path=audio_path,
                project_root=project_root,
                source_label=source_label,
                label=_label(source_label),
                label_level="record",
                seg_start_s=0.0,
                seg_end_s=None,
                device="Littmann 3200",
                filter_mode=FILTERS[filter_code.upper()],
                site_code=site,
                age_group="pediatric" if age < 18 else "adult",
                split="external",
            )
        )
    return rows

