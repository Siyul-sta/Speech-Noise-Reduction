"""Reference-based validation only: clean audio must NEVER enter inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from src.engine_runner import enhance


def score(clean, estimate, sr):
    if sr != 16000:
        raise ValueError("Challenge PESQ-WB requires 16 kHz")
    if clean.shape != estimate.shape:
        raise ValueError("Clean and estimate shapes do not match")
    clean, estimate = np.asarray(clean), np.asarray(estimate)
    snr = 10 * np.log10(np.sum(clean ** 2) /
                        max(np.sum((clean - estimate) ** 2), 1e-18))
    # PESQ accepts mono input. Validation consists of mono WAV files.
    if clean.ndim != 1:
        raise ValueError("PESQ evaluator expects mono validation data")
    return float(pesq(sr, clean, estimate, "wb")), float(snr)


def evaluate(root, out, limit=0):
    root = Path(root)
    samples = pd.read_csv(root / "validation" / "samples.csv")
    if limit:
        samples = samples.head(limit)
    rows = []
    for i, r in enumerate(samples.itertuples(index=False), 1):
        clean, sr = sf.read(root / r.clean_path)
        noisy, sr_noisy = sf.read(root / r.noisy_path)
        if sr != sr_noisy:
            raise ValueError(f"Sample rate mismatch: {r.segment_id}")
        outcomes = {}
        for method in ("a", "b"):
            enhanced = enhance(noisy, sr, method=method)
            p, s = score(clean, enhanced, sr)
            outcomes[method] = (p, s)
            rows.append({"segment_id": r.segment_id, "method": method,
                         "pesq_wb": p, "snr_db": s,
                         "noise_category": getattr(r, "noise_category", ""),
                         "target_snr_db": getattr(r, "target_snr_db", np.nan)})
        # Oracle uses CLEAN only for offline analysis, NEVER hidden-test routing.
        winner = max(outcomes, key=lambda m: outcomes[m][0])
        rows.append({"segment_id": r.segment_id, "method": "oracle_analysis_only",
                     "pesq_wb": outcomes[winner][0],
                     "snr_db": outcomes[winner][1],
                     "noise_category": getattr(r, "noise_category", ""),
                     "target_snr_db": getattr(r, "target_snr_db", np.nan)})
        if i % 20 == 0:
            print(f"Evaluated {i}/{len(samples)}", flush=True)
    results = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(results.groupby("method")[["pesq_wb", "snr_db"]].mean().to_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="noise_reduction_challenge")
    parser.add_argument("--output", default="results/dual_engine_validation.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.data, args.output, args.limit)
