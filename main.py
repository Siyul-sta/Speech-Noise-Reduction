"""Submission CLI: python main.py --input input --output output"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor


def render(task):
    path, output_dir, method = task
    if method == 'final':
        from src.final_dsp import enhance
    else:
        from src.engine_runner import enhance as legacy
        def enhance(x, sr):
            return legacy(x, sr, method=method)
    info = sf.info(path)
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    y = enhance(x, sr)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError(f'Invalid output for {path.name}')
    # Match the PCM16 quantization used for the published validation results.
    sf.write(output_dir / path.name, y, sr, subtype='PCM_16')
    result = sf.info(output_dir / path.name)
    if (result.frames, result.channels, result.samplerate) != (info.frames, info.channels, info.samplerate):
        raise AssertionError(f'Output metadata differs: {path.name}')


def process_directory(input_dir, output_dir, method='final', workers=1):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError('Input and output directories must differ')
    if workers < 1:
        raise ValueError('workers must be positive')
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise ValueError("No WAV files found in input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(path, output_dir, method) for path in files]
    if workers == 1:
        for task in tasks:
            render(task)
    else:
        with ProcessPoolExecutor(workers) as pool:
            list(pool.map(render, tasks))
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    parser.add_argument("--method", choices=("final", "a", "b"), default="final")
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(f"Processed {process_directory(args.input, args.output, args.method, args.workers)} WAV files")
