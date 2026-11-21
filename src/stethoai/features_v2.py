"""v2 model input: gain-invariant log-mel with loop padding (fixes the v1 level/zero-padding sensitivity).

v1 padded short windows with zeros and floored log power at an absolute 1e-12, so a padded window's normalised
image depended on the recording level. v2 (a) fills a short window by repeating its own audio and (b) floors log
power relative to the window's peak mel power (80 dB range). A pure gain change then shifts every log value by the
same constant, which the per-window mean/std normalisation removes exactly. One torch implementation serves
training (batched, after waveform augmentation) and inference.
"""
from __future__ import annotations

import numpy as np
import torch

from .features import _mel_bank

FEATURES_V2 = dict(version=2, sample_rate=4000, window_samples=12000, n_fft=256, hop_length=64, n_mels=40,
                   mel_bank='features._mel_bank (HTK 0-2000 Hz, unit area)', stft='torch center zeros hann',
                   padding='loop_repeat_own_audio', log_floor='relative_1e-8_of_window_peak_plus_1e-30',
                   normalization='per_window_mean_std_floor_1e-6')
_RELATIVE_FLOOR = 1e-8


def pad_window(segment: np.ndarray, width: int = FEATURES_V2['window_samples']) -> np.ndarray:
    """Return exactly `width` samples; a short segment is repeated end to end (never zero-filled)."""
    segment = np.asarray(segment, dtype=np.float32)
    if segment.ndim != 1 or len(segment) == 0:
        raise ValueError('pad_window needs a non-empty 1-D segment')
    if len(segment) >= width:
        return segment[:width].copy()
    return np.tile(segment, int(np.ceil(width / len(segment))))[:width].copy()


def logmel_batch(waves: torch.Tensor) -> torch.Tensor:
    """(batch, samples) float waveforms -> (batch, 1, 40, frames) normalised log-mel images."""
    if waves.ndim != 2:
        raise ValueError('expected (batch, samples)')
    window = torch.hann_window(FEATURES_V2['n_fft'], periodic=False, dtype=waves.dtype)
    spec = torch.stft(waves, FEATURES_V2['n_fft'], FEATURES_V2['hop_length'], window=window, center=True,
                      pad_mode='constant', return_complex=True)
    bank = torch.from_numpy(_mel_bank()).to(waves.dtype)
    power = torch.einsum('mf,bft->bmt', bank, spec.abs()**2)
    peak = power.amax(dim=(1, 2), keepdim=True)
    image = torch.log(torch.maximum(power, peak*_RELATIVE_FLOOR + 1e-30))
    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    return ((image - mean) / std).unsqueeze(1)


def logmel_input_v2(window: np.ndarray) -> np.ndarray:
    """One already-padded 3 s window -> (1, 40, frames) float32 image."""
    wave = torch.from_numpy(np.asarray(window, dtype=np.float32)).double().unsqueeze(0)
    return logmel_batch(wave)[0].float().numpy()
