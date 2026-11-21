"""Recording aggregation, abstention-aware metrics and deterministic OOF selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .training_data import CLASSES

AGGREGATIONS = ("mean", "max", "vote")
THRESHOLDS = (0., .3, .4, .5, .6, .7, .8, .9, .95)


def aggregate(probabilities, recording_indices, n_records, rule):
    assert len(probabilities) == len(recording_indices)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert set(recording_indices) == set(range(n_records)), "recording has no window prediction"
    output = np.zeros((n_records, 4), dtype=float)
    if rule == "mean":
        np.add.at(output, recording_indices, probabilities)
    elif rule == "max":
        np.maximum.at(output, recording_indices, probabilities)
    elif rule == "vote":
        np.add.at(output, (recording_indices, probabilities.argmax(axis=1)), 1)
    else:
        raise ValueError(f"unknown aggregation: {rule}")
    output /= output.sum(axis=1, keepdims=True)
    return output


def score_predictions(truth, probabilities, threshold=0.):
    assert len(truth) > 0 and len(truth) == len(probabilities)
    predicted = probabilities.argmax(axis=1)
    retained = probabilities.max(axis=1) >= threshold
    predicted = np.where(retained, predicted, 4)  # U is output only; FN for its true class.
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(4), zero_division=0)
    abnormal = truth != 0
    normal = truth == 0
    correct = predicted == truth
    se = float(correct[abnormal].mean()) if abnormal.any() else None
    sp = float(correct[normal].mean()) if normal.any() else None
    return dict(n=len(truth), retained=int(retained.sum()), coverage=float(retained.mean()),
                accuracy=float(correct.mean()),
                retained_accuracy=float(correct[retained].mean()) if retained.any() else None,
                macro_f1=float(f1.mean()),
                per_class={label: dict(precision=float(precision[i]), recall=float(recall[i]),
                                       f1=float(f1[i]), support=int(support[i]))
                           for i,label in enumerate(CLASSES)},
                confusion_matrix=confusion_matrix(truth, predicted, labels=np.arange(5))[:4].tolist(),
                icbhi_sensitivity=se, icbhi_specificity=sp,
                icbhi_score=(se+sp)/2 if se is not None and sp is not None else None)


def threshold_curve(truth, probabilities):
    return [dict(threshold=t, **score_predictions(truth, probabilities, t)) for t in THRESHOLDS]


def select_result(results):
    # Ties: coverage, shorter window, model name, aggregation name, lower threshold.
    return sorted(results, key=lambda r: (-r["macro_f1"], -r["coverage"], r["window_s"],
                                          r["model"], r["aggregation"], r["threshold"]))[0]


def choose_abstain_threshold(truth, probabilities):
    """D-039 proposed operating rule, using train-only recording OOF predictions.

    Grid includes both endpoints. Fallback ties take the lowest threshold.
    This does not select a model, alter its probabilities or freeze an operating policy.
    """
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities, dtype=float)
    if (truth.ndim != 1 or not len(truth) or
            probabilities.shape != (len(truth), 4) or
            not np.isin(truth, np.arange(4)).all() or
            not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError("expected nonempty N/C/W/B truth and normalized finite four-class probabilities")
    curve = [dict(threshold=i/100, **score_predictions(truth, probabilities, i/100))
             for i in range(101)]
    a0 = curve[0]["retained_accuracy"]
    feasible = [point for point in curve if point["coverage"] >= .70]
    targets = [point for point in feasible if point["retained_accuracy"] >= a0 + .05]
    chosen = targets[0] if targets else min(feasible, key=lambda p: (-p["retained_accuracy"], p["threshold"]))
    return dict(chosen, branch="target" if targets else "fallback", a0=a0,
                target_accuracy=a0+.05, coverage_floor=.70, curve=curve)
