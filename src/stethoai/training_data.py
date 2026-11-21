"""Fail-closed train admission and per-recording 4 kHz mono cache (Phase 3)."""
from __future__ import annotations

import hashlib
import io
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf

ADULT_SOURCES = ("icbhi2017", "hf_lung_v1")
CLASSES = ("N", "C", "W", "B")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adult_training_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Use only train adult-source recordings with no Q/X or failed QC rows."""
    train = manifest.loc[(manifest.split == "train") & manifest.dataset.isin(ADULT_SOURCES)].copy()
    assert (train.split == "train").all(), "train rows only"
    bad = train.loc[~train.label.isin(CLASSES) | train.qc_status.ne("pass"),
                    ["dataset", "recording_id"]].drop_duplicates()
    keys = set(map(tuple, bad.to_numpy()))
    keep = [(d, r) not in keys for d, r in zip(train.dataset, train.recording_id)]
    return train.loc[keep].reset_index(drop=True)


class TrainingAudio:
    """Guard raw AND cached reads against the complete manifest before any I/O.

    An authorized record must match its manifest identity, path, hash and patient.
    The manifest is trusted project metadata, not permission supplied by the caller.
    """

    def __init__(self, root: Path, manifest: pd.DataFrame):
        self.root = root.resolve()
        self.cache = self.root / "data/processed/features/audio"
        admitted = adult_training_rows(manifest)
        locked = manifest.loc[manifest.split != "train"]
        locked_paths = {self._path(p) for p in locked.audio_path}
        train_paths = {self._path(p) for p in admitted.audio_path}
        assert not train_paths & locked_paths, "train path aliases non-train audio"
        used_locked = locked.loc[locked.split != "excluded"]
        locked_patients = set(zip(used_locked.dataset, used_locked.patient_id.astype(str)))
        train_patients = set(zip(admitted.dataset, admitted.patient_id.astype(str)))
        assert not train_patients & locked_patients, "train patient overlaps locked patient"
        self.allowed = {self._identity(r) for r in admitted.to_dict("records")}
        self.audit: list[dict] = []

    def _path(self, path) -> Path:
        resolved = (self.root / str(path)).resolve()
        assert resolved.is_relative_to(self.root), "audio path escapes project root"
        return resolved

    def _identity(self, row) -> tuple:
        return (str(row["dataset"]), str(row["recording_id"]), str(row["patient_id"]),
                self._path(row["audio_path"]), str(row["sha256"]))

    def load(self, row) -> np.ndarray:
        # These checks deliberately precede cache lookup and ALL file opening.
        assert row["split"] == "train", "only train audio may be read in Phase 3"
        identity = self._identity(row)
        assert identity in self.allowed, "audio is not in the admitted train manifest"
        path = identity[3]
        key = hashlib.sha256((str(path) + str(row["sha256"]) + "mono4000_v1").encode()).hexdigest()
        cached = self.cache / f"{key}.npy"
        self.cache.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            signal = np.load(cached, allow_pickle=False)
            access = "cache"
        else:
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["sha256"], "raw audio hash changed"
            channels, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            signal = channels.mean(axis=1)
            if sr != 4000:
                divisor = gcd(sr, 4000)
                signal = resample_poly(signal, 4000 // divisor, sr // divisor)
            signal = signal.astype(np.float32)
            assert signal.size and np.isfinite(signal).all(), "invalid train audio"
            np.save(cached, signal, allow_pickle=False)
            access = "raw"
        assert signal.ndim == 1 and np.isfinite(signal).all(), "invalid cached train audio"
        self.audit.append(dict(dataset=row["dataset"], recording_id=row["recording_id"],
                               audio_path=str(row["audio_path"]), split="train", access=access))
        return signal
