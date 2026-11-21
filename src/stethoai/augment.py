"""Training-only waveform augmentation for v2 (4 kHz, one 3 s window at a time).

Order: device EQ (a smooth random gain curve over 0-2 kHz), optional clipping, then coloured background noise at a
random SNR. Pure gain is not needed: the v2 input is gain-invariant by construction (features_v2).
"""
from __future__ import annotations

import numpy as np

AUGMENT_CONFIG = dict(version=1, eq_probability=.6, eq_points_hz=[0, 250, 500, 1000, 2000], eq_db=8.,
                      clip_probability=.1, clip_peak=[1., 2.], noise_probability=.6,
                      noise_colours={'white': 0., 'pink': 1., 'brown': 2.}, snr_db=[3., 30.],
                      start_jitter_s=.5)


def coloured_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-power noise with power spectrum proportional to 1/f**beta."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spectrum *= freqs ** (-beta / 2)
    noise = np.fft.irfft(spectrum, n)
    return noise / max(float(np.sqrt(np.mean(noise**2))), 1e-12)


def random_eq(wave: np.ndarray, rng: np.random.Generator, sr: int = 4000) -> np.ndarray:
    points = np.asarray(AUGMENT_CONFIG['eq_points_hz'], dtype=float)
    gains_db = rng.uniform(-AUGMENT_CONFIG['eq_db'], AUGMENT_CONFIG['eq_db'], len(points))
    freqs = np.fft.rfftfreq(len(wave), 1 / sr)
    curve = 10 ** (np.interp(freqs, points, gains_db) / 20)
    return np.fft.irfft(np.fft.rfft(wave) * curve, len(wave))


def add_noise(wave: np.ndarray, snr_db: float, beta: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(np.square(wave, dtype=np.float64)))
    if power <= 0:
        return wave.copy()
    return wave + coloured_noise(len(wave), beta, rng) * np.sqrt(power / 10 ** (snr_db / 10))


def augment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(wave, dtype=np.float64)
    if rng.random() < AUGMENT_CONFIG['eq_probability']:
        out = random_eq(out, rng)
    if rng.random() < AUGMENT_CONFIG['clip_probability']:
        peak = float(np.abs(out).max())
        if peak > 0:
            out = np.clip(out * rng.uniform(*AUGMENT_CONFIG['clip_peak']) / peak, -1, 1)
    if rng.random() < AUGMENT_CONFIG['noise_probability']:
        beta = list(AUGMENT_CONFIG['noise_colours'].values())[rng.integers(3)]
        out = add_noise(out, rng.uniform(*AUGMENT_CONFIG['snr_db']), beta, rng)
    assert np.isfinite(out).all()
    return out.astype(np.float32)
