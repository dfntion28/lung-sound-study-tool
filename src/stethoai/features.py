"""Stateless log-mel/MFCC summaries; learned scaling belongs inside CV."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct
from scipy.signal import savgol_filter, stft

FEATURE_CONFIG = dict(version=1, sample_rate=4000, n_fft=256, hop_length=64,
                      n_mels=40, n_mfcc=20, fmin=0, fmax=2000,
                      mel_scale="HTK", mel_normalization="unit_area_discrete",
                      spectrum="scipy_stft_hann_spectrum_power_zero_boundary",
                      log="natural_floor_1e-12", mfcc="DCT_II_ortho",
                      deltas="MFCC_savgol_width9_poly2_deriv1_2_nearest",
                      summary="mean_population_std", dimensions=200)


@lru_cache(maxsize=1)
def _mel_bank() -> np.ndarray:
    frequencies = np.linspace(0, 2000, 129)
    points = 700 * (10**(np.linspace(0, 2595*np.log10(1+2000/700), 42)/2595)-1)
    filters = []
    for low, center, high in zip(points[:-2], points[1:-1], points[2:]):
        triangle = np.maximum(0, np.minimum((frequencies-low)/(center-low),
                                             (high-frequencies)/(high-center)))
        filters.append(triangle / triangle.sum())
    return np.asarray(filters, dtype=np.float32)


def logmel_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Shared stateless Phase 3/4 spectral recipe; no fitted parameters."""
    assert signal.ndim == 1 and len(signal) >= 256 and np.isfinite(signal).all()
    config = FEATURE_CONFIG
    _, _, spectrum = stft(signal, fs=config['sample_rate'], window="hann", nperseg=config['n_fft'],
                           noverlap=config['n_fft']-config['hop_length'], nfft=config['n_fft'],
                           boundary="zeros", padded=True)
    return np.log(np.maximum(_mel_bank() @ np.abs(spectrum)**2, 1e-12))


def summarize_features(signal: np.ndarray) -> np.ndarray:
    logmel = logmel_spectrogram(signal)
    mfcc = dct(logmel, type=2, axis=0, norm="ortho")[:20]
    delta = savgol_filter(mfcc, 9, 2, deriv=1, axis=1, mode="nearest")
    delta2 = savgol_filter(mfcc, 9, 2, deriv=2, axis=1, mode="nearest")
    result = np.concatenate([stat for block in (logmel, mfcc, delta, delta2)
                             for stat in (block.mean(axis=1), block.std(axis=1))])
    assert result.shape == (200,) and np.isfinite(result).all()
    return result.astype(np.float32)
