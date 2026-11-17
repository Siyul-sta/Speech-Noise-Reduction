"""Noisy-only PSD candidate family built on the verified second-submission DSP.

No Clean/Reference/PESQ/evaluation data is imported by this module.
The baseline uses 12_1500; the known 0.25 SPP blend is a fixed control.
"""
import numpy as np
from scipy import ndimage
from src import final_dsp as dsp

EPS = 1e-18
ORIGINAL = dsp.noise_estimate
MODES = ("baseline", "spp25", "quiet_focus", "speech_guard", "down_guard", "nonstationary")


def mix_psd(power, baseline, spp, mode):
    """Return a PSD computed strictly from noisy STFT power and PSD estimates."""
    if mode == "baseline":
        return baseline
    if mode not in MODES:
        raise ValueError("Unknown mode: " + mode)

    baseline = np.maximum(np.asarray(baseline, float), EPS)
    spp = np.maximum(np.asarray(spp, float), EPS)
    power = np.maximum(np.asarray(power, float), EPS)
    if baseline.shape != spp.shape or baseline.shape != power.shape:
        raise ValueError("PSD shapes must match")
    log_ratio = np.log(spp) - np.log(baseline)
    if mode == "spp25":
        weight = .25
    else:
        gamma = np.clip(power / baseline, 0, 1e5)
        # Gerkmann-Hendriks-style posterior speech presence likelihood.
        xi = 10.0 ** 1.5
        speech_probability = 1.0 / (1.0 + (1.0 + xi) * np.exp(-gamma * xi / (1.0 + xi)))
        speech_probability = ndimage.gaussian_filter(speech_probability, (1.0, 1.0))
        if mode == "quiet_focus":
            # Trust SPP primarily where speech is unlikely.
            weight = .06 + .44 * (1.0 - speech_probability) ** 1.5
        elif mode == "speech_guard":
            # Do not introduce large positive PSD corrections in speech-heavy bins.
            weight = .25
            log_ratio = np.minimum(log_ratio, .12 + 1.2 * (1.0 - speech_probability))
        elif mode == "down_guard":
            # Retain baseline in bins where SPP overestimates the noise.
            weight = .25
            log_ratio = np.minimum(log_ratio, .0)
        elif mode == "nonstationary":
            # More SPP adaptation in frequency bands whose estimated PSD varies.
            variation = ndimage.gaussian_filter(
                abs(np.diff(np.log(baseline), axis=1, prepend=np.log(baseline[:, :1]))),
                (1.0, 3.0),
            )
            adaptive = np.clip(variation / .20, 0, 1)
            weight = (.10 + .40 * adaptive) * (1 - .6 * speech_probability)
        else:
            raise AssertionError(mode)
    candidate = baseline * np.exp(np.clip(weight * log_ratio, -3.0, 3.0))
    if not np.isfinite(candidate).all() or np.any(candidate <= 0):
        raise FloatingPointError("Non-finite/negative noise PSD")
    return candidate


def make_estimator(mode):
    """Installable hook called only with the observed Noisy spectrogram."""
    if mode not in MODES:
        raise ValueError(mode)

    def estimator(power, initial, full, ignored_mode):
        base = ORIGINAL(power, initial, full, "local_15_0.5")
        if mode == "baseline":
            return base
        spp = dsp.core.track_noise_spp(power, initial)
        return mix_psd(power, base, spp, mode)
    return estimator


def enhance(noisy, rate=16000, mode="spp25"):
    """Noisy-only single-channel enhancement, shares the original final DSP pipeline."""
    previous = dsp.noise_estimate
    dsp.noise_estimate = make_estimator(mode)
    try:
        return dsp.enhance(noisy, rate)
    finally:
        dsp.noise_estimate = previous
