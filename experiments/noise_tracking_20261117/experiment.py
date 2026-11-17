"""Noisy-only noise tracker candidates. Clean is read only by the evaluator."""
import argparse
import csv
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal
from pesq import pesq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import final_dsp as dsp

ORIGINAL = dsp.noise_estimate


def tracker(power, initial, full, mode, variant):
    baseline = ORIGINAL(power, initial, full, mode)
    if variant == 'baseline':
        return baseline
    spp = dsp.core.track_noise_spp(power, initial)
    # A spectral floor limits speech-contaminated long-term estimates.
    if variant.startswith('blend'):
        weight = float(variant.split('_')[1])
        return baseline**(1-weight) * spp**weight
    if variant == 'capped':
        return np.minimum(baseline, spp)
    if variant == 'residual':
        # Estimate residual noise using first-pass speech presence evidence.
        ratio = power / np.maximum(baseline, 1e-18)
        speech = ndimage.gaussian_filter(np.maximum(ratio-1, 0), (1, 1))
        probability = speech / (speech+3)
        quiet = (1-probability)**2
        numerator = ndimage.uniform_filter1d(power*quiet, 31, axis=1)
        denominator = ndimage.uniform_filter1d(quiet, 31, axis=1)
        local = numerator / np.maximum(denominator, .01)
        local = np.clip(local, baseline*.5, baseline*2)
        return np.sqrt(local*baseline)
    raise ValueError(variant)


def evaluate(job):
    row, root, variants = job
    clean, sr = sf.read(Path(root)/row['clean_path'])
    noisy, _ = sf.read(Path(root)/row['noisy_path'])
    result = {'id': row['segment_id']}
    for variant in variants:
        def estimate(p, i, f, m):
            return tracker(p, i, f, m, variant)
        dsp.noise_estimate = estimate
        try:
            output = dsp.enhance(noisy, sr)
        finally:
            dsp.noise_estimate = ORIGINAL
        stream = io.BytesIO()
        sf.write(stream, output, sr, format='WAV', subtype='PCM_16')
        stream.seek(0)
        output, _ = sf.read(stream)
        result[variant] = pesq(sr, clean, output, 'wb')
        result[variant+'_snr'] = 10*np.log10(np.sum(clean**2)/np.sum((clean-output)**2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.manifest, encoding='utf-8-sig')))
    with ProcessPoolExecutor(8) as pool, open(args.output, 'w') as stream:
        for index, result in enumerate(pool.map(evaluate, [(r,args.root,args.variants) for r in rows])):
            if index == 0:
                writer = csv.DictWriter(stream, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            stream.flush()
