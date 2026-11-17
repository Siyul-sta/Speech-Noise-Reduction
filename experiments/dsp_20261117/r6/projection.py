"""Match the output level to a second noisy-only speech estimate."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from previous import Config, enhance
from colour import color

def run(x, pilot, mode):
    config = json.loads(Path(__file__).with_name('selected_config.json').read_text())['config']
    reference = enhance(x, 16000, Config(**config), estimator='quantile', blend=1.)
    level = mode.split('_')[-1]
    if mode.startswith("v5proj_cross"):
        from resonance import run as resonance
        # Parallel narrow-band correction with the high-SNR reconstruction.
        shaped = resonance(x, .5 * (reference + pilot), 'v5res_500_0.32_' + level)
        return .5 * (reference + shaped)
    shaped = color(x, (reference + pilot) / 2, 'v5colour_500_' + level)
    if 'local' in mode:
        length = 4097
        window = signal.windows.hann(length)
        correlation = signal.fftconvolve(shaped * reference, window, mode='same')
        energy = signal.fftconvolve(shaped ** 2, window, mode='same')
        gain = np.clip(correlation / np.maximum(energy, 1.e-12), .5, 1.5)
    else:
        gain = np.clip(shaped @ reference / max(shaped @ shaped, 1.e-12), .5, 1.5)
    return shaped * gain
