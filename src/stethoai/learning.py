"""Patient-grouped folds, fold-local scaling and balanced baseline models."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
MODELS = ("logistic", "random_forest", "linear_svm")
MODEL_SETTINGS = {
    "logistic": dict(C=1., class_weight="balanced", solver="lbfgs", max_iter=2000),
    "random_forest": dict(n_estimators=100, max_depth=20, min_samples_leaf=2,
                          max_features="sqrt", class_weight="balanced_subsample", n_jobs=1,
                          random_state=SEED),
    "linear_svm": dict(C=1., class_weight="balanced", dual="auto", max_iter=10000,
                       random_state=SEED, calibration="sigmoid", inner_grouped_folds=3,
                       calibration_ensemble=False),
}


def grouped_folds(records, n_splits=5):
    assert records.recording_key.is_unique, "duplicate recording"
    assert records.patient_key.notna().all(), "missing patient"
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(records)), records.label, records.patient_key))
    for train, val in folds:
        assert not set(records.iloc[train].patient_key) & set(records.iloc[val].patient_key)
        assert set(records.iloc[train].label) == {0, 1, 2, 3}, "training fold lacks class"
    return folds


def make_estimator(name, x, y, groups, quick=False):
    if name == "logistic":
        return Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(**MODEL_SETTINGS[name]))])
    if name == "random_forest":
        settings = dict(MODEL_SETTINGS[name])
        if quick:
            settings["n_estimators"] = 8
        return Pipeline([("model", RandomForestClassifier(**settings))])
    if name == "linear_svm":
        settings = {k: v for k, v in MODEL_SETTINGS[name].items()
                    if k not in {"calibration", "inner_grouped_folds", "calibration_ensemble"}}
        base = Pipeline([("scale", StandardScaler()), ("model", LinearSVC(**settings))])
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                     .split(x, y, groups))
        for train, val in inner:
            assert not set(groups[train]) & set(groups[val]), "calibration patient leakage"
            assert set(y[train]) == {0,1,2,3}, "calibration training fold lacks class"
        return CalibratedClassifierCV(base, method="sigmoid", cv=inner, ensemble=False, n_jobs=1)
    raise ValueError(f"unknown model: {name}")


def fit_oof(name, x, y, recording_indices, records, *, quick=False, progress=None):
    """Each validation patient's windows are predicted once; no OOF scaler reuse."""
    assert len(x) == len(y) == len(recording_indices)
    groups = records.patient_key.to_numpy()[recording_indices]
    oof = np.full((len(y), 4), np.nan)
    visits = np.zeros(len(y), dtype=int)
    diagnostics = []
    for fold, (train_records, val_records) in enumerate(grouped_folds(records)):
        train = np.flatnonzero(np.isin(recording_indices, train_records))
        val = np.flatnonzero(np.isin(recording_indices, val_records))
        assert not set(groups[train]) & set(groups[val]), "outer patient leakage"
        assert len(train) and len(val)
        started = time.monotonic()
        model = make_estimator(name, x[train], y[train], groups[train], quick=quick)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x[train], y[train])
        assert np.array_equal(model.classes_, np.arange(4)), "class order changed"
        oof[val] = model.predict_proba(x[val])
        visits[val] += 1
        diagnostic = dict(fold=fold, train_windows=len(train), validation_windows=len(val),
                          train_records=len(train_records), validation_records=len(val_records),
                          train_patients=len(set(groups[train])), validation_patients=len(set(groups[val])),
                          patient_overlap=0, validation_record_indices=val_records.tolist(),
                          training_class_windows=np.bincount(y[train], minlength=4).tolist(),
                          warnings=[str(w.message) for w in caught],
                          elapsed_s=round(time.monotonic()-started, 3))
        if isinstance(model, Pipeline) and "scale" in model.named_steps:
            diagnostic["scaler_mean"] = model.named_steps["scale"].mean_.tolist()
        if name == "linear_svm":
            diagnostic["inner_calibration"] = [dict(train_windows=len(a), validation_windows=len(b),
                                                     patient_overlap=len(set(groups[train][a]) & set(groups[train][b])))
                                                 for a,b in model.cv]
        diagnostics.append(diagnostic)
        if progress:
            progress(f"{name} fold {fold+1}/5: train={len(train)} val={len(val)} "
                     f"seconds={diagnostic['elapsed_s']} warnings={diagnostic['warnings']}")
    assert (visits == 1).all() and np.isfinite(oof).all(), "incomplete/duplicate OOF predictions"
    return oof, diagnostics
