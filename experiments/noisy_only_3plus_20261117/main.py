"""Run an experimental Noisy-only candidate. Does not change root main.py."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import final_dsp as dsp
from candidate import make_estimator, MODES, ORIGINAL


def process(args):
    # All candidates re-use root submission code to preserve PCM16/file handling.
    import runpy
    process_directory = runpy.run_path(str(ROOT / "main.py"))["process_directory"]
    previous = dsp.noise_estimate
    dsp.noise_estimate = make_estimator(args.mode)
    try:
        return process_directory(args.input, args.output, workers=1)
    finally:
        dsp.noise_estimate = previous


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=MODES, default="spp25")
    args = p.parse_args()
    print("Processed", process(args), "files")
