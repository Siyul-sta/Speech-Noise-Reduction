"""Run fixed, reference-free final attenuation candidates on mono 16 kHz WAVs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from experiment import estimate


def render(task):
    source, destination, mode = task
    x, rate = sf.read(source, dtype='float64')
    if rate != 16000 or x.ndim != 1 or len(x) < 4096 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite mono 16 kHz input of at least 4096 samples: {source}')
    y = estimate(x, mode)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError(f'Invalid output: {source}')
    sf.write(destination, y, rate, subtype='PCM_16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['low_-2', 'low_2', 'high_-2', 'high_2', 'presence_-2', 'presence_2'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output directories must differ')
    files = sorted(args.input.glob('*.wav'))
    if not files:
        parser.error('No input WAV files')
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(render, [(p, args.output / p.name, args.mode) for p in files]))
