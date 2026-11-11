"""Competition CLI. Example: python main.py --input input --output output --method b"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from src.engine_runner import enhance


def process_directory(input_dir, output_dir, method):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise ValueError("No WAV files found in input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        info = sf.info(path)
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        y = enhance(x, sr, method=method)
        if y.shape != x.shape or not np.isfinite(y).all():
            raise ValueError(f"Invalid output for {path.name}")
        subtype = info.subtype if info.subtype in (
            "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"
        ) else "PCM_16"
        sf.write(output_dir / path.name, np.clip(y, -1.0, 1.0), sr, subtype=subtype)
        result = sf.info(output_dir / path.name)
        if (result.frames != info.frames or result.channels != info.channels
                or result.samplerate != info.samplerate):
            raise AssertionError(f"Output metadata differs: {path.name}")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    parser.add_argument("--method", choices=("a", "b", "v4"), default="b")
    args = parser.parse_args()
    print(f"Processed {process_directory(args.input, args.output, args.method)} WAV files")
