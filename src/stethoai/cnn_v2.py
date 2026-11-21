"""v2 CNN: same output contract as v1 (3 s windows, 1.5 s hop, vote, N/C/W/B + Uncertain), new training recipe.

Changes from v1: gain-invariant loop-padded input (features_v2), on-the-fly waveform augmentation (augment),
source-balanced window sampling, optional extra wheeze weight, early stopping on the source-macro F1 of inner
validation patients, and a source-robust Uncertain threshold. Architecture is v1's SmallCNN, unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .augment import AUGMENT_CONFIG, augment
from .cnn import SmallCNN, seed_everything
from .evaluation import score_predictions
from .features_v2 import FEATURES_V2, logmel_batch, pad_window
from .training_data import CLASSES
from .windowing import window_bounds

WIDTH = FEATURES_V2['window_samples']
GROUPS = ('icbhi', 'hf_steth', 'hf_trunc')
ARMS = {
    'A': dict(balance_alpha=0., augment=False, wheeze_factor=1.),
    'B': dict(balance_alpha=.5, augment=True, wheeze_factor=1.),
    'C': dict(balance_alpha=.5, augment=True, wheeze_factor=1.5),
}
TRAIN_CONFIG = dict(version=1, batch_size=32, learning_rate=1e-3, epoch_windows=32000, max_epochs=15, patience=3,
                    min_delta=1e-4, monitor='inner_source_macro_f1_vote', seed=42, threads=4)


class WaveStore:
    """Memory-mapped float16 training waveforms plus record and labelled-window tables."""

    def __init__(self, folder: Path):
        folder = Path(folder)
        self.waves = np.load(folder / 'train_waves_f16.npy', mmap_mode='r', allow_pickle=False)
        self.records = pd.read_csv(folder / 'train_records.csv', keep_default_na=False)
        self.windows = pd.read_csv(folder / 'train_windows.csv')
        assert self.records.record_index.tolist() == list(range(len(self.records)))
        self.offsets = self.records.offset.to_numpy()
        self.lengths = self.records.length.to_numpy()
        self.win_record = self.windows.record_index.to_numpy()
        self.win_start = self.windows.start.to_numpy()
        self.win_label = self.windows.label.to_numpy()
        self.win_group = self.records.group.to_numpy()[self.win_record]

    def signal(self, record_index: int) -> np.ndarray:
        o, n = self.offsets[record_index], self.lengths[record_index]
        return np.asarray(self.waves[o:o + n], dtype=np.float32)

    def window(self, w: int, rng: np.random.Generator | None = None) -> np.ndarray:
        r, start = self.win_record[w], int(self.win_start[w])
        length = int(self.lengths[r])
        if rng is not None:
            jitter = int(rng.uniform(-1, 1) * AUGMENT_CONFIG['start_jitter_s'] * 4000)
            start = int(np.clip(start + jitter, 0, length - 1))
        o = self.offsets[r]
        return pad_window(np.asarray(self.waves[o + start:o + min(start + WIDTH, length)], dtype=np.float32))


def sampling_weights(groups: np.ndarray, alpha: float) -> np.ndarray:
    """Per-window weight n_group**-alpha: 0 = natural mix, 0.5 = square-root balance, 1 = equal groups."""
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(float) ** (-alpha)
    return weights / weights.sum()


def class_weights(labels: np.ndarray, weights: np.ndarray, wheeze_factor: float) -> torch.Tensor:
    expected = np.bincount(labels, weights=weights, minlength=4)
    if (expected <= 0).any():
        raise ValueError('fitting subset lacks a class')
    w = 1 / (4 * expected / expected.sum())
    w[2] *= wheeze_factor
    return torch.tensor(w, dtype=torch.float32)


def images(waves: list[np.ndarray]) -> torch.Tensor:
    return logmel_batch(torch.from_numpy(np.stack(waves)).double()).float()


def predict_windows(model, waves: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(waves), batch_size):
            out.append(torch.softmax(model(images(waves[i:i + batch_size])), dim=1).numpy())
    return np.concatenate(out) if out else np.empty((0, 4))


def vote_fractions(probabilities: np.ndarray) -> np.ndarray:
    return np.bincount(probabilities.argmax(axis=1), minlength=4) / len(probabilities)


def source_macro_f1(truth: np.ndarray, fractions: np.ndarray, groups: np.ndarray) -> dict:
    """Recording macro-F1 (threshold 0) per source group and their unweighted mean (the v2 selection metric)."""
    per = {g: score_predictions(truth[groups == g], fractions[groups == g])['macro_f1']
           for g in GROUPS if (groups == g).any()}
    return dict(per_group=per, source_macro=float(np.mean(list(per.values()))))


def train_epoch(model, store, fit_windows, probabilities, criterion, optimizer, rng, generator, use_augment):
    model.train()
    draw = torch.multinomial(torch.from_numpy(probabilities), TRAIN_CONFIG['epoch_windows'], replacement=True,
                             generator=generator).numpy()
    total = 0.
    for i in range(0, len(draw), TRAIN_CONFIG['batch_size']):
        batch = fit_windows[draw[i:i + TRAIN_CONFIG['batch_size']]]
        waves = [store.window(w, rng) for w in batch]
        if use_augment:
            waves = [augment(w, rng) for w in waves]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images(waves)), torch.from_numpy(store.win_label[batch]).long())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch)
    return total / len(draw)


def record_fractions(model, store, record_indices) -> np.ndarray:
    """Clean labelled-window vote fractions for each requested recording (v1 OOF convention)."""
    wanted = np.isin(store.win_record, record_indices)
    ws = np.flatnonzero(wanted)
    p = predict_windows(model, [store.window(w) for w in ws])
    remap = {int(r): k for k, r in enumerate(record_indices)}
    out = np.zeros((len(record_indices), 4))
    np.add.at(out, (np.array([remap[int(r)] for r in store.win_record[ws]]), p.argmax(axis=1)), 1)
    return out / out.sum(axis=1, keepdims=True)


def fit(store, fit_records, val_records, arm: dict, seed: int, epochs: int | None = None, progress=print):
    """Train on fit_records; early-stop on val_records' source-macro F1 unless a fixed epoch count is given."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    fit_windows = np.flatnonzero(np.isin(store.win_record, fit_records))
    probabilities = sampling_weights(store.win_group[fit_windows], arm['balance_alpha'])
    model = SmallCNN()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(store.win_label[fit_windows], probabilities,
                                                               arm['wheeze_factor']))
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    history, best, best_epoch, best_state, stale = [], -np.inf, 0, None, 0
    limit = epochs or TRAIN_CONFIG['max_epochs']
    for epoch in range(1, limit + 1):
        loss = train_epoch(model, store, fit_windows, probabilities, criterion, optimizer, rng, generator,
                           arm['augment'])
        entry = dict(epoch=epoch, train_loss=loss)
        if epochs is None:
            truth = store.records.label.to_numpy()[val_records]
            score = source_macro_f1(truth, record_fractions(model, store, val_records),
                                    store.records.group.to_numpy()[val_records])
            entry.update(inner=score)
            if score['source_macro'] > best + TRAIN_CONFIG['min_delta']:
                best, best_epoch, stale = score['source_macro'], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(entry)
        progress(json.dumps(entry))
        if epochs is None and stale >= TRAIN_CONFIG['patience']:
            break
    if epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, history


# ---------------- inference (the contract the team tool would use if v2 is adopted) ----------------

def load_v2(model_dir: Path):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    raw = (model_dir / config['model_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['model_sha256']:
        raise ValueError('v2 model hash mismatch')
    if config['features'] != FEATURES_V2:
        raise ValueError('v2 feature config mismatch')
    model = SmallCNN()
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return model, config


def output_label(fractions: np.ndarray, threshold: float) -> str:
    return CLASSES[int(np.argmax(fractions))] if float(np.max(fractions)) >= threshold else 'U'


def predict_signal(model, signal: np.ndarray, threshold: float) -> dict:
    """Deployment view: every 3 s / 1.5 s window of a 4 kHz recording, loop-padded, vote, Uncertain rule."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError('expected a finite non-empty 4 kHz mono signal')
    bounds = window_bounds(len(signal), 3)
    p = predict_windows(model, [pad_window(signal[s:e]) for s, e in bounds])
    fractions = vote_fractions(p)
    return dict(bounds=bounds, window_probabilities=p, vote=fractions, label=output_label(fractions, threshold))
