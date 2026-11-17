"""Train/Validation scoring. Clean is read ONLY here, after Noisy-only inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from candidate import MODES, enhance


def evaluate_one(job):
    row, root, modes = job
    clean, sr = sf.read(root / row["clean_path"], dtype="float64")
    noisy, nsr = sf.read(root / row["noisy_path"], dtype="float64")
    if sr != 16000 or nsr != sr or clean.shape != noisy.shape or clean.ndim != 1:
        raise ValueError("Expected matched mono 16 kHz WAV pairs")
    limit = int(row.get("valid_samples") or len(clean))
    if not 0 < limit <= len(clean):
        raise ValueError("Invalid valid_samples")
    # No reference audio is passed to the enhancement routine.
    output = {"segment_id": row["segment_id"]}
    for mode in modes:
        restored = enhance(noisy, sr, mode)
        bytestream = io.BytesIO()
        sf.write(bytestream, restored, sr, format="WAV", subtype="PCM_16")
        bytestream.seek(0)
        decoded, _ = sf.read(bytestream, dtype="float64")
        c, y = clean[:limit], decoded[:limit]
        if decoded.shape != clean.shape or not np.isfinite(y).all():
            raise ValueError("Output size/finite-value violation")
        output[mode + "_pesq"] = float(pesq(sr, c, y, "wb"))
        output[mode + "_stoi"] = float(stoi(c, y, sr, extended=False))
        output[mode + "_snr"] = float(10*np.log10(np.sum(c*c) /
                                              max(np.sum((c-y)**2), 1e-18)))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0, help="Optional Train screening limit; 0 means all")
    args = p.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_files > 0:
        rows = rows[:args.max_files]
    if not rows:
        raise ValueError("Empty manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.output.open("w", newline="") as handle:
        writer = None
        for count, result in enumerate(pool.map(evaluate_one,
                            [(r, args.root, args.modes) for r in rows]), start=1):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)
            if count % 10 == 0:
                handle.flush()
        handle.flush()

    import pandas as pd
    df = pd.read_csv(args.output)
    for mode in args.modes:
        print(mode, "count=", len(df), "PESQ=", round(df[mode+"_pesq"].mean(), 6),
              "STOI=", round(df[mode+"_stoi"].mean(), 6),
              "SNR=", round(df[mode+"_snr"].mean(), 6))
    print("Important: don't select by single-file gains; check mean and regressions.")
    if "spp25" in args.modes:
        for mode in args.modes:
            if mode != "spp25":
                delta = df[mode+"_pesq"] - df["spp25_pesq"]
                print("vs_spp25", mode, "delta=", round(delta.mean(),6),
                      "improved=", int((delta>0).sum()), "regressed=", int((delta<0).sum()))


if __name__ == "__main__":
    main()
