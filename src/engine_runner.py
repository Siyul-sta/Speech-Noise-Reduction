"""Adapters for the two independently developed classical DSP engines."""
import numpy as np
from scipy import signal


def enhance_a(x, sr):
    """IMCRA + OM-LSA + modulation Kalman + fusion; noisy input only."""
    from .engines.speech_gain_engine import compute_gain_matrix
    if len(x) < 512:
        return np.asarray(x, dtype=np.float64).copy()
    _, _, z = signal.stft(x, fs=sr, window="hann", nperseg=512,
                          noverlap=384, boundary="zeros", padded=True)
    mag = np.abs(z)
    gain = compute_gain_matrix(mag, mag ** 2)
    if gain.shape != z.shape or not np.isfinite(gain).all():
        raise ValueError("Engine A returned an invalid gain matrix")
    _, y = signal.istft(z * gain, fs=sr, window="hann", nperseg=512,
                        noverlap=384, boundary=True)
    return _same_length(y, len(x))


def enhance_b(x, sr):
    """TSNR + HRNR + Wiener v3; noisy input only."""
    from .engines.tsnr_hrnr_v3 import enhance, Config
    return _same_length(enhance(x, sr, Config()), len(x))


def _same_length(y, n):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("Engine must return mono 1-D audio")
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = y[:n]
    if not np.isfinite(y).all():
        raise ValueError("Engine produced NaN or Inf")
    return y


def enhance(x, sr=16000, method="a"):
    """Apply one engine to each channel, retaining input shape and metadata."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError("Expected samples or samples x channels")
    if method not in ("a", "b"):
        raise ValueError("Choose method a or b; oracle auto-selection is prohibited")
    f = enhance_a if method == "a" else enhance_b
    if x.ndim == 1:
        return f(x, sr)
    return np.column_stack([f(x[:, c], sr) for c in range(x.shape[1])])
