"""Frequency-selective restoration after noise suppression."""
import numpy as np
from scipy import ndimage, signal
DEPTH_RULE = None

def restore(x, pilot, mode):
    opts = dict(fs=16000, nperseg=1024, noverlap=768)
    freq, _, a = signal.stft(pilot, **opts)
    z = signal.stft(x, **opts)[2]
    logfreq = np.log2(np.maximum(freq, 1) / 500)
    primary = np.exp(-.5 * (logfreq / .32)**2)
    envelope = ndimage.gaussian_filter(abs(a)**2, (3, 1.5))
    background = np.quantile(ndimage.gaussian_filter(abs(z)**2, (3, 1.5)), .2, axis=1)[:, None]
    confidence = envelope / (envelope + background + 1.e-18)
    kind = mode.split('_')[1]
    if kind == 'slope':
        # Control the bass-side and treble-side skirts independently.
        low, high = map(float, mode.split('_')[2:])
        depth = np.where(freq <= 500, low, high)
        if DEPTH_RULE is None:
            response = (1 + (primary - 1) * depth / 20)[:, None]
        else:
            depth_matrix = DEPTH_RULE(freq, a, z, depth)
            response = 1 + (primary[:, None] - 1) * depth_matrix / 20
    elif kind == 'voice':
        center = int(mode.split('_')[2])
        second = np.exp(-.5 * (np.log2(np.maximum(freq, 1) / center) / .45)**2)
        response = primary[:, None] + (1 - primary[:, None]) * second[:, None] * (.15 + .55 * confidence)
    elif kind == 'flat':
        order = float(mode.split('_')[2])
        response = np.exp(-.5 * abs(logfreq / .32)**order)[:, None]
    else:
        # Use high-confidence speech outside the primary band for parallel restoration.
        level = float(mode.split('_')[2])
        response = primary[:, None] + (1-primary[:, None]) * level * confidence**2
    y = signal.istft(a * 10**((response-1)*20/20), **opts)[1][:len(x)]
    return y * np.clip(x @ y / max(y @ y, 1.e-18), .1, 8)
