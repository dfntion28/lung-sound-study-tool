"""Train-only deterministic feature extraction and resumable recording caches."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .features import FEATURE_CONFIG, summarize_features
from .training_data import CLASSES, TrainingAudio, adult_training_rows
from .windowing import label_interval, make_windows

WINDOWS = (3, 5, 8)
LABEL_RULE_VERSION = "D033_positive_overlap_whole_record_QX_exclusion_v1"


def extract_training_features(root, manifest, manifest_hash, progress=print):
    audio = TrainingAudio(root, manifest)  # Validate against locked metadata before any cache reads.
    admitted = adult_training_rows(manifest)
    assert len(admitted) and admitted.split.eq("train").all(), "train only feature extraction"
    cache = root / "data/processed/features/record_features"
    cache.mkdir(parents=True, exist_ok=True)
    grouped = list(admitted.groupby(["dataset", "recording_id"], sort=True))
    records, feature_audit = [], []
    arrays = {w: [] for w in WINDOWS}
    metadata = {w: [] for w in WINDOWS}
    for index, ((dataset, rec_id), segments) in enumerate(grouped):
        record = segments.iloc[0].to_dict()
        assert record["split"] == "train"
        assert segments.patient_id.astype(str).nunique() == 1
        assert segments.audio_path.nunique() == 1 and segments.sha256.nunique() == 1
        truth = label_interval(segments, 0, float(record["duration_s"]))
        assert truth in CLASSES, "recording lacks usable truth"
        records.append(dict(recording_key=f"{dataset}::{rec_id}", dataset=dataset,
                            recording_id=rec_id, patient_key=f"{dataset}::{record['patient_id']}",
                            label=CLASSES.index(truth), annotation_rows=len(segments)))
        fingerprint = json.dumps([manifest_hash, dataset, rec_id, record["sha256"],
                                  FEATURE_CONFIG, LABEL_RULE_VERSION], sort_keys=True)
        path = cache / f"{hashlib.sha256(fingerprint.encode()).hexdigest()}.npz"
        reused = path.exists()
        if not reused:
            signal = audio.load(record)
            payload = {}
            for w in WINDOWS:
                windows = make_windows(signal, segments, w)
                assert windows, f"no annotated windows for {dataset}/{rec_id} at {w}s"
                payload[f"x{w}"] = np.vstack([summarize_features(item[3]) for item in windows])
                payload[f"meta{w}"] = np.array([[start,end,CLASSES.index(label)]
                                                 for start,end,label,_ in windows])
            np.savez_compressed(path, **payload)
        # Features are train-derived; the admission assertion applies on cache hits as well.
        assert segments.split.eq("train").all()
        with np.load(path, allow_pickle=False) as stored:
            for w in WINDOWS:
                x, meta = stored[f"x{w}"], stored[f"meta{w}"]
                assert x.shape == (len(meta), 200) and np.isfinite(x).all()
                arrays[w].append(x)
                metadata[w].extend([(index, float(s), float(e), int(y)) for s,e,y in meta])
        feature_audit.append(dict(dataset=dataset, recording_id=rec_id, split="train",
                                  access="cache" if reused else "extracted"))
        if (index+1) % 250 == 0 or index == len(grouped)-1:
            progress(f"features: {index+1}/{len(grouped)} recordings")
    (root / "logs").mkdir(exist_ok=True)
    pd.DataFrame(audio.audit, columns=["dataset","recording_id","audio_path","split","access"]).to_csv(
        root / "logs/baseline_audio_access.csv", index=False)
    pd.DataFrame(feature_audit).to_csv(root / "logs/baseline_feature_access.csv", index=False)
    sets = {}
    for w in WINDOWS:
        meta = pd.DataFrame(metadata[w], columns=["record_index","start_s","end_s","label"])
        sets[w] = (np.vstack(arrays[w]), meta)
    return pd.DataFrame(records), sets, admitted
