"""Run the weak SPP blend candidate through the standard WAV entry point."""
from experiment import ORIGINAL, dsp, tracker

if __name__ == '__main__':
    def estimate(power, initial, full, mode):
        return tracker(power, initial, full, mode, 'blend_0.25')

    dsp.noise_estimate = estimate
    import argparse
    import runpy
    from pathlib import Path
    process_directory = runpy.run_path(str(Path(__file__).resolve().parents[2] / "main.py"))["process_directory"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        count = process_directory(args.input, args.output, workers=1)
        print(f"Processed {count} WAV files")
    finally:
        dsp.noise_estimate = ORIGINAL
