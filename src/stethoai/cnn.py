"""Option B: a small, randomly initialized CPU CNN on the Phase 3 log-mel recipe."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .features import logmel_spectrogram
from .learning import SEED

CNN_CONFIG = dict(version=2, channels=[16, 32, 64, 128], kernel_size=3,
                  dropout=.25, batch_size=32, learning_rate=.001,
                  default_epochs=20, patience=3, inner_fraction=.20,
                  monitor='inner_recording_macro_f1_mean_aggregation',
                  normalization='per_window_mean_std_floor_1e-6',
                  loss='inverse_fit_window_frequency_cross_entropy',
                  optimizer='Adam', seed=SEED, num_workers=0, threads=4)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def logmel_input(signal):
    image = logmel_spectrogram(signal)
    image = (image-image.mean()) / max(float(image.std()), 1e-6)
    return image[np.newaxis].astype(np.float32)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        incoming = 1
        for outgoing in CNN_CONFIG['channels']:
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)])
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(CNN_CONFIG['dropout']), nn.Linear(incoming, 4))

    def forward(self, x):
        return self.head(self.features(x))


def balanced_loss(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, range(4)).all():
        raise ValueError('invalid class labels')
    counts = np.bincount(labels.astype(int), minlength=4)
    if (counts == 0).any():
        raise ValueError('fitting subset lacks a class')
    return nn.CrossEntropyLoss(weight=torch.tensor(len(labels)/(4*counts), dtype=torch.float32))
