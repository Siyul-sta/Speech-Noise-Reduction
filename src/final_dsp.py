"""Fixed 12_1500 DSP inference; no evaluation or clean-data imports.

Channels are independent. Other rates use polyphase resampling to/from 16 kHz.
Short clips are edge-padded and trimmed. Hooks are process-local, not thread-safe.
"""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import ndimage, signal
from scipy.special import exp1

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'dsp_20261117'
for folder in (ROOT / 'psd_round', ROOT / 'r6'):
    sys.path.insert(0, str(folder))
import core
import pockets
from previous import Config, enhance as first_stage
from candidate import process
from noisy_estimators import estimate as noise_estimate

BASE = Config(**json.loads((ROOT / 'r6' / 'best_config.json').read_text())['config'])


def estimate_16k(x):
    """Same arithmetic and fixed settings as archived depth_round 12_1500."""
    def depth_rule(freq, speech_spec, noisy_spec, base_depth):
        speech = ndimage.gaussian_filter(abs(speech_spec)**2, (2, 1.5))
        noisy = ndimage.gaussian_filter(abs(noisy_spec)**2, (2, 1.5))
        background = np.quantile(noisy, .2, axis=1, keepdims=True)
        evidence = speech / (speech + background + 1e-18)
        weight = np.exp(-0.5 * (np.log2(np.maximum(freq, 1) / 1500.) / 0.6)**2)
        return np.maximum(base_depth[:, None] + 4 * (1 - evidence) + 12 * weight[:, None], 0)

    def gain(xi, gamma, floor):
        xi = np.maximum(xi, 1e-10)
        gamma = np.maximum(gamma, 1e-10)
        w = xi / (1 + xi)
        v = np.maximum(w * gamma, 1e-10)
        g = w * np.exp(.5 * exp1(v))
        return np.clip((1 - .25) * w + .25 * np.clip(g, floor, 1), floor, 1)

    def estimator(power, initial, full, actual, sr, nfft, hop):
        return noise_estimate(power, initial, full, 'local_15_0.5')

    gain.final_only = False
    saved = core.NOISE_ESTIMATOR, core.GAIN_RULE, pockets.DEPTH_RULE
    pockets.DEPTH_RULE = depth_rule
    try:
        core.NOISE_ESTIMATOR, core.GAIN_RULE = estimator, gain
        try:
            pilot = first_stage(x, 16000, replace(BASE, structure='fit_eq_erb'),
                                estimator='quantile', blend=1)
        finally:
            core.NOISE_ESTIMATOR, core.GAIN_RULE = saved[:2]
        return process(x, pilot, 'hi_floor_tone_1.0')
    finally:
        core.NOISE_ESTIMATOR, core.GAIN_RULE, pockets.DEPTH_RULE = saved


def _channel(x, rate):
    if not len(x) or np.max(np.abs(x)) < 1e-12:
        return x.copy()
    divisor = math.gcd(rate, 16000)
    internal = x if rate == 16000 else signal.resample_poly(x, 16000 // divisor, rate // divisor)
    n = len(internal)
    if n < 4096:
        internal = np.pad(internal, (0, 4096 - n), mode='edge')
    output = estimate_16k(internal)[:n]
    if rate != 16000:
        output = signal.resample_poly(output, rate // divisor, 16000 // divisor)
    return np.pad(output, (0, max(0, len(x) - len(output))))[:len(x)]


def enhance(x, sr=16000):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('Expected finite samples or samples x channels')
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError('Sample rate must be a positive integer')
    if x.ndim == 1:
        return _channel(x, int(sr))
    return np.column_stack([_channel(x[:, c], int(sr)) for c in range(x.shape[1])])
