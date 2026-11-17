"""Speech-evidence-gated high-band gain floor after the current DSP chain."""
import numpy as np
from scipy import signal, ndimage


def restore(x, y, mode):
    # mode: hi_floor_005 / _010 / _015. The floor is relative to noisy magnitude.
    floor = 0.01 if "tone" in mode or "track" in mode else int(mode.rsplit("_", 1)[1]) / 100.0
    kw = dict(fs=16000, nperseg=1024, noverlap=768)
    f, _, z = signal.stft(x, **kw)
    _, _, a = signal.stft(y, **kw)

    # A purely spectral, reference-free speech cue: output energy above its
    # per-frequency lower envelope. This avoids restoring high-band noise
    # uniformly across time.
    p = ndimage.gaussian_filter(np.abs(a) ** 2, (1.0, 1.5))
    bg = np.quantile(p, 0.20, axis=1, keepdims=True)
    excess_db = 10.0 * np.log10((p + 1e-18) / (bg + 1e-18))
    conf = np.clip((excess_db - 2.0) / 8.0, 0.0, 1.0)
    conf = ndimage.gaussian_filter(conf, (0.8, 0.8))

    # Restrict restoration to 750 Hz–6.5 kHz; leave low bands untouched.
    band = np.clip((f - 650.0) / 250.0, 0.0, 1.0)
    band *= np.clip((7000.0 - f) / 500.0, 0.0, 1.0)
    if "tone" in mode:
        power = ndimage.gaussian_filter(np.abs(z) ** 2, (0.5, 1.0))
        neighborhood = ndimage.uniform_filter1d(power, 17, axis=0)
        concentration = np.clip((power / (neighborhood + 1e-18) - 2) / 6, 0, 1)
        persistence = ndimage.uniform_filter1d(concentration, 19, axis=1)
        strength = float(mode.rsplit("_", 1)[1])
        conf *= 1 - strength * persistence
    if "track" in mode:
        # Local lower-envelope noise tracking with a power-domain Wiener gain.
        power = ndimage.gaussian_filter(np.abs(z) ** 2, (1.0, 1.0))
        noise = ndimage.percentile_filter(power, percentile=20, size=(1, 31))
        noise = ndimage.gaussian_filter(noise, (2.0, 1.0))
        posterior_gain = np.sqrt(np.clip(1 - noise / (power + 1e-18), .01, 1))
        strength = float(mode.rsplit("_", 1)[1])
        a *= posterior_gain ** strength
    local_floor = floor * conf * band[:, None]
    current = np.abs(a)
    source_magnitude = np.abs(z)
    if "noise" in mode:
        power = ndimage.gaussian_filter(np.abs(z) ** 2, (1.0, 1.0))
        noise = np.quantile(power, .20, axis=1, keepdims=True)
        if "local" in mode:
            noise = ndimage.percentile_filter(power, percentile=20, size=(1, 31))
        clean_fraction = np.clip(1 - noise / (power + 1e-18), .01, 1)
        source_magnitude *= np.sqrt(clean_fraction)
    target = np.maximum(current, local_floor * source_magnitude)
    # Keep the estimated clean phase; if it is numerically absent, use noisy phase.
    phase = np.divide(a, current, out=np.divide(z, np.maximum(np.abs(z), 1e-18)), where=current > 1e-12)
    out = target * phase
    return signal.istft(out, **kw)[1][:len(x)]
