"""Run the reference-free PSD candidate on mono 16 kHz WAVs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from experiment import first_pass
from candidate import process


def render(task):
    source, destination, mode = task
    x, sr = sf.read(source, dtype='float64')
    if sr != 16000 or x.ndim != 1 or len(x) < 4096 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite mono 16 kHz WAV with at least 4096 samples: {source}')
    pilot = first_pass(x, mode)
    y = process(x, pilot, 'hi_floor_tone_1.0')
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError(f'Invalid output: {source}')
    sf.write(destination, y, sr, subtype='PCM_16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'local_15_0.5', 'quiet_31_0.5'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output directories must differ')
    paths = sorted(args.input.glob('*.wav'))
    if not paths:
        parser.error('No input WAVs found')
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(p, args.output / p.name, args.mode) for p in paths]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(render, tasks))
