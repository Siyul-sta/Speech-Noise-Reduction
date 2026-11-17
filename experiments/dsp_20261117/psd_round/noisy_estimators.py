"""Reference-free local lower-envelope estimators for first-pass DSP filtering."""
import numpy as np
from scipy import ndimage
from adaptive import relative_noise


def estimate(power, initial, full, mode):
    kind, span_text, strength_text = mode.split('_')
    span = int(span_text)
    strength = float(strength_text)
    smoothed = ndimage.gaussian_filter(power, (1.5, 1.0))
    local = ndimage.percentile_filter(smoothed, 20, size=(1, span), mode='nearest')
    reference = np.quantile(smoothed[:, full], .20, axis=1, keepdims=True)
    change = np.clip(local / np.maximum(reference, 1e-18), .25, 4)
    if kind == 'local':
        proposed = initial[:, None] * np.sqrt(change)
    elif kind == 'quiet':
        # Select quiet frames separately within each frequency neighborhood.
        band_energy = ndimage.uniform_filter1d(smoothed, 17, axis=0)
        threshold = np.quantile(band_energy[:, full], .20, axis=1, keepdims=True)
        quiet = band_energy <= threshold
        weights = ndimage.uniform_filter1d(quiet.astype(float), span, axis=1)
        weighted = ndimage.uniform_filter1d(smoothed * quiet, span, axis=1)
        proposed = np.where(weights > .05, weighted / np.maximum(weights, 1e-18), local)
        # Keep the original calibration while allowing local temporal changes.
        center = np.median(proposed[:, full], axis=1, keepdims=True)
        proposed = initial[:, None] * np.clip(proposed / np.maximum(center, 1e-18), .25, 4)
    else:
        raise ValueError(mode)
    baseline = relative_noise(power, initial, full)
    return np.exp((1-strength)*np.log(np.maximum(baseline, 1e-18)) + strength*np.log(np.maximum(proposed, 1e-18)))
