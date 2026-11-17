"""Experimental coherent-tone cancellation around frozen submission 4187900.
Noisy-only inference. No training, reference audio, labels, or evaluation imports.
"""
import math
from pathlib import Path
import sys
import numpy as np
from scipy import signal, ndimage, fft, optimize
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.final_dsp import enhance as baseline


def stable_lines(x, y):
    """Fit globally coherent sinusoidal residuals by least squares, noisy only."""
    n = len(x)
    if n < 4096:
        return np.zeros_like(x)
    nfft = fft.next_fast_len(n * 4)
    residual = x-y
    power = abs(fft.rfft(residual * np.hanning(n), nfft))**2
    background = ndimage.median_filter(power, size=101, mode='nearest') + max(power.max()*1e-12, 1e-18)
    hz = fft.rfftfreq(nfft, 1/16000)
    peaks = signal.find_peaks(power, distance=12)[0]
    peaks = peaks[(hz[peaks]>100) & (hz[peaks]<6500) & (power[peaks]/background[peaks]>100)]
    peaks = sorted(peaks, key=lambda p: power[p], reverse=True)[:8]
    t = np.arange(n)/16000
    columns = []
    for peak in peaks:
        def objective(frequency):
            value = np.dot(residual, np.exp(-2j*np.pi*frequency*t))
            return -abs(value)**2
        frequency = optimize.minimize_scalar(objective, bounds=(hz[peak]-.5,hz[peak]+.5),
                                             method='bounded', options={'xatol':1e-5}).x
        oscillator = np.exp(-2j*np.pi*frequency*t)
        coefficients = [np.mean(residual[s]*oscillator[s]) for s in
                        [slice(k*n//6,(k+1)*n//6) for k in range(6)]]
        coherence = abs(np.mean(coefficients))**2 / (np.mean(abs(np.array(coefficients))**2)+1e-18)
        if coherence >= .9:
            columns.extend([np.cos(2*np.pi*frequency*t), np.sin(2*np.pi*frequency*t)])
    if not columns:
        return np.zeros_like(x)
    basis = np.column_stack(columns)
    return basis @ np.linalg.lstsq(basis, x, rcond=None)[0]



def _channel(x, sr):
    if not len(x) or np.max(abs(x)) < 1e-12:
        return x.copy()
    divisor = math.gcd(sr, 16000)
    internal = x if sr == 16000 else signal.resample_poly(x, 16000//divisor, sr//divisor)
    pilot = baseline(internal, 16000)
    tones = stable_lines(internal, pilot)
    y = baseline(internal-tones, 16000) if np.any(tones) else pilot
    if sr != 16000:
        y = signal.resample_poly(y, sr//divisor, 16000//divisor)
    return np.pad(y, (0, max(0,len(x)-len(y))))[:len(x)]


def enhance(x, sr=16000):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('Expected finite samples or samples x channels')
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError('Invalid sample rate')
    if x.ndim == 1:
        return _channel(x, int(sr))
    return np.column_stack([_channel(x[:, c], int(sr)) for c in range(x.shape[1])])
