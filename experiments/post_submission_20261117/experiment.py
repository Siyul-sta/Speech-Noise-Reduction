"""Noisy-only postfilters on the frozen second-submission DSP baseline.

HPSS principle: FitzGerald (2010), Harmonic/Percussive Separation Using
Median Filtering, https://www.dafx.de/paper-archive/2010/DAFx10/DerryFitzGerald_DAFx10_P15.pdf
Soft attenuation here is an experimental speech adaptation, not the paper's
music-separation result. Clean references are accessed only by scoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, ndimage, fft, optimize
from pesq import pesq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.final_dsp import enhance


def stable_lines(x, y):
    """Fit globally coherent sinusoidal residuals by least squares, noisy only."""
    n = len(x)
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


def postfilter(x, y, mode):
    if mode == 'baseline':
        return y.copy()
    family, strength = mode.split('_')
    strength = float(strength)
    if family == 'track':
        # Track slowly changing narrowband residual carriers before rerunning DSP.
        kw = dict(fs=16000, nperseg=2048, noverlap=1920)
        z, a = signal.stft(x, **kw)[2], signal.stft(y, **kw)[2]
        residual = z-a
        span = 33
        cross = residual[:, 1:]*residual[:, :-1].conj()
        cross = np.pad(cross, ((0,0),(1,0)), mode='edge')
        cross = ndimage.uniform_filter1d(cross, span-2, axis=1)
        carrier = np.exp(1j*np.cumsum(np.angle(cross), axis=1))
        predicted = ndimage.uniform_filter1d(residual*carrier.conj(), span, axis=1)*carrier
        power = ndimage.uniform_filter1d(abs(residual)**2, span, axis=1)
        coherence = np.clip(abs(predicted)**2/(power+1e-18), 0, 1)
        neighborhood = ndimage.median_filter(power, size=(17,1), mode='nearest')
        prominence = np.clip((power/(neighborhood+1e-18)-5)/15, 0, 1)
        gate = np.clip((coherence-.8)/.2, 0, 1)*prominence
        tones = signal.istft(predicted*gate, **kw)[1][:len(x)]
        return enhance(x-strength*tones, 16000)
    if family in ('lines', 'linepost'):
        tones = stable_lines(x, y)
        if not np.any(tones):
            return y.copy()
        if family == 'lines':
            return enhance(x-strength*tones, 16000)
        # Project only the estimated stable noise carrier onto the final waveform.
        alpha = np.clip(np.dot(y, tones)/(np.dot(tones, tones)+1e-18), 0, 1)
        return y-strength*alpha*tones
    kw = dict(fs=16000, nperseg=1024, noverlap=768)
    a, z = signal.stft(y, **kw)[2], signal.stft(x, **kw)[2]
    magnitude = abs(a)
    if family == 'envelope':
        # Infer speech activity from the observable mixture's lower envelope.
        f = np.fft.rfftfreq(1024, 1/16000)
        band = (f >= 250) & (f <= 3500)
        power = ndimage.gaussian_filter(abs(z)**2, (1, 1))
        floor = np.quantile(power, .2, axis=1, keepdims=True)
        activity = ndimage.gaussian_filter1d(np.sum(np.maximum(power-floor, 0)[band], axis=0), 2)
        level = ndimage.gaussian_filter1d(np.sum(magnitude[band]**2, axis=0), 2)
        activity /= np.quantile(activity, .8) + 1e-18
        level /= np.quantile(level, .8) + 1e-18
        gain = np.clip(np.sqrt((activity+.02)/(level+.02)), .5, 2)**strength
        out = a * gain[None, :]
    elif family == 'compand':
        # Modulation-domain compression/expansion; fixed exponent, no fitting.
        level = ndimage.gaussian_filter1d(np.sum(magnitude**2, axis=0), 2)
        relative = level / (np.quantile(level, .8) + 1e-18)
        gain = np.clip(np.maximum(relative, .001)**strength, .5, 2)
        out = a * gain[None, :]
    elif family == 'hpss':
        # Horizontal persistence versus vertical broadband transients.
        horizontal = ndimage.median_filter(magnitude, size=(1, 9), mode='nearest')
        vertical = ndimage.median_filter(magnitude, size=(17, 1), mode='nearest')
        percussion = vertical**2 / (horizontal**2 + vertical**2 + 1e-18)
        out = a * (1 - strength * percussion)
    elif family == 'isolated':
        # Musical-noise peaks unsupported in a local time/frequency neighborhood.
        reference = ndimage.median_filter(magnitude, size=(3, 5), mode='nearest')
        spike = np.clip(1 - reference / (magnitude + 1e-18), 0, 1)
        out = a * (1 - strength * spike)
    elif family == 'phase':
        # Move processed phase towards the observable mixture phase, magnitude fixed.
        rotation = np.angle(z * a.conj())
        out = a * np.exp(1j * strength * rotation)
    else:
        raise ValueError(mode)
    return signal.istft(out, **kw)[1][:len(x)]


def pcm16(y):
    buffer = io.BytesIO()
    sf.write(buffer, y, 16000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return sf.read(buffer, dtype='float64')[0]


def score(task):
    data, cache, row, modes = task
    c, sr = sf.read(data / row['clean_path'], dtype='float64')
    x, rate = sf.read(data / row['noisy_path'], dtype='float64')
    if sr != 16000 or rate != sr or x.shape != c.shape:
        raise ValueError('Expected matched mono 16 kHz')
    key = hashlib.sha256(x.tobytes()).hexdigest()
    cached = cache / (key + '.npy')
    y = np.load(cached) if cached.exists() else enhance(x, sr)
    if not cached.exists():
        np.save(cached, y)
    rows = []
    for mode in modes:
        start = time.monotonic()
        w = pcm16(postfilter(x, y, mode))
        if w.shape != c.shape or not np.isfinite(w).all():
            raise ValueError('Invalid output')
        value = float(pesq(sr, c, w, 'wb'))
        snr = float(10 * np.log10(np.sum(c**2) / max(np.sum((c - w)**2), 1e-30)))
        rows.append(dict(segment_id=row['segment_id'], speaker=row['speaker'],
                         noise=row['noise_category'], target=row['target_snr_db'],
                         mode=mode, pesq=value, snr=snr, seconds=time.monotonic()-start))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--result', required=True, type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--modes', nargs='+', default=['baseline','hpss_0.15','hpss_0.35',
                  'isolated_0.15','isolated_0.35','phase_0.25','phase_0.5'])
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(a.manifest)
    tasks = [(a.data, a.cache, row, a.modes) for row in manifest.to_dict('records')]
    with ProcessPoolExecutor(a.workers) as pool:
        records = sum(pool.map(score, tasks), [])
    result = pd.DataFrame(records)
    a.result.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.result, index=False)
    summary = result.groupby('mode')[['pesq', 'snr']].mean()
    summary['delta_pesq'] = summary.pesq - summary.loc['baseline', 'pesq']
    summary.to_csv(a.result.with_name(a.result.stem + '_summary.csv'))
    print(summary.sort_values('pesq', ascending=False).to_string(), flush=True)
