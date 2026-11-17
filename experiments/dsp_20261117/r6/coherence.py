"""Subtract long-lived complex carriers before time-frequency gain estimation."""
import numpy as np
from scipy import signal, ndimage
from resonance import run as shape

def repair(x, pilot, mode):
    size, hop = 2048, 128
    if mode.startswith(('v5carrier_span','v5carrier_internal')):
        _, _, size_text, span_text, _ = mode.split('_')
        size = int(size_text)
        span = int(span_text)
    else:
        span = 37
    if mode.startswith('v5carrier_phase'):
        size, span = 1024, 19
    kw = dict(fs=16000, nperseg=size, noverlap=size-hop)
    _, _, spec = signal.stft(x, **kw)
    original = signal.stft(pilot, **kw)[2]
    residual = spec - original
    tracker = residual
    if mode.startswith('v5carrier_phase'):
        kind = mode.split('_')[2]
        if kind == 'raw':
            tracker = spec
        elif kind == 'speech':
            tracker = original
        elif kind == 'comp':
            energy = ndimage.uniform_filter1d(abs(original)**2, span, axis=1)
            projection = ndimage.uniform_filter1d((spec*original.conj()).real, span, axis=1)
            residual = spec - original*np.clip(projection/(energy+1.e-18), .5, 4)
            tracker = residual
    cross = tracker[:,1:] * tracker[:,:-1].conj()
    cross = np.pad(cross, ((0,0),(1,0)), mode='edge')
    cross = ndimage.uniform_filter1d(cross, span - 2, axis=1)
    carrier = np.exp(1j * np.cumsum(np.angle(cross), axis=1))
    # De-rotate each bin then average across time to isolate stable sinusoids.
    demod = residual * carrier.conj()
    pred = ndimage.uniform_filter1d(demod, span, axis=1) * carrier
    power = ndimage.uniform_filter1d(abs(residual)**2, span, axis=1)
    reliability = np.clip(abs(pred)**2 / (power+1.e-18), 0, 1)
    est = spec - reliability * pred
    speech = ndimage.gaussian_filter(abs(original)**2, (1,1))
    noise = np.quantile(ndimage.gaussian_filter(abs(est)**2,(1,1)), .18, axis=1)[:,None]
    gain = speech / (speech + noise + 1.e-18)
    if mode.startswith('v5carrier_pre'):
        from previous import enhance, Config
        from dynamic import apply
        from pathlib import Path
        import json
        cleaned = signal.istft(est, **kw)[1][:len(x)]
        cfg = Config(**json.loads(Path(__file__).with_name('pesq_config.json').read_text())['config'])
        second = enhance(cleaned, 16000, cfg, estimator='quantile', blend=1.)
        second = apply(cleaned, second, 'dyn_global')
        y = .35*pilot + .65*second
        if mode.endswith('plain'):
            return y
        return shape(x, y, 'v5res_500_0.32_20')
    if mode.startswith('v5carrier_path'):
        from pathspeech import reconstruct
        return reconstruct(x, original, spec, est, reliability, gain, kw, mode)
    blend = float(mode.split('_')[-1])
    out = original + blend * reliability * (est * np.sqrt(gain) - original)
    y = signal.istft(out, **kw)[1][:len(x)]
    if mode.startswith('v5carrier_internal'):
        return y
    if mode.startswith(('v5carrier_span','v5carrier_phase')):
        from pockets import restore
        return restore(x, y, 'v5r6_slope_24_16')
    if mode.startswith("v5carrier_bass"):
        from bass_control import combine
        return combine(x,y,mode)
    if mode.startswith(("v5carrier_fid","v5carrier_ring")):
        from fidelity import recover
        return recover(x,pilot,y,mode)
    if mode.startswith("v5carrier_adapt"):
        from equal_loudness import adapt
        return adapt(x,y,mode)
    if mode.startswith('v5carrier_contour'):
        from contour import equalize
        return equalize(x, y, mode)
    if mode.startswith('v5carrier_low'):
        from pockets import restore
        cutoff = int(mode.split('_')[2])
        focused = restore(x, y, 'v5r6_slope_24_16')
        # Complementary zero-phase crossover. Restore the low-frequency waveform
        # of the pilot, preserving the gain above the selected crossover.
        bass = signal.butter(4, cutoff, fs=16000, output='sos')
        source = pilot if 'pilot' in mode else y
        return focused + signal.sosfiltfilt(bass, source - focused)
    if mode.startswith('v5carrier_skirt'):
        from pockets import restore
        low, high = mode.split('_')[2:4]
        return restore(x, y, 'v5r6_slope_' + low + '_' + high)
    if mode.startswith('v5carrier_plain'):
        return y
    return shape(x, y, 'v5res_500_0.32_20')
