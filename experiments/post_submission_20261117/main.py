"""Standalone experimental CLI; does not change the official root main.py."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from coherent_tones import enhance


def render(task):
    source, destination = task
    x, sr = sf.read(source, dtype='float64', always_2d=True)
    y = enhance(x, sr)
    if x.shape != y.shape or not np.isfinite(y).all():
        raise ValueError(f'Invalid output: {source}')
    sf.write(destination, y, sr, subtype='PCM_16')
    info = sf.info(destination)
    if (info.frames, info.channels, info.samplerate) != (len(x), x.shape[1], sr):
        raise ValueError('Output metadata mismatch')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve() or a.workers < 1:
        p.error('Use distinct folders and positive workers')
    files = sorted(f for f in a.input.iterdir() if f.is_file() and f.suffix.lower()=='.wav')
    if not files:
        p.error('No WAV files')
    a.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(render, [(f, a.output/f.name) for f in files]))
    print(f'Processed {len(files)} WAV files')
