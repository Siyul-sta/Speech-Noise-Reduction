"""Classical DSP PESQ-WB grid search; clean speech is used for scoring ONLY.

python scripts/optimize_pesq.py --data noise_reduction_challenge.zip
python scripts/optimize_pesq.py --data noise_reduction_challenge.zip --metric snr --limit 12 --configs baseline,less_hrnr
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engines.tsnr_hrnr_v3 import Config, enhance

BASE = Config()
EXPERIMENTS = {
    "baseline": {},
    "less_hrnr": {"harmonic_weight": 0.5},
    "no_hrnr": {"harmonic_weight": 0.0},
    "medium_hrnr": {"harmonic_weight": 1.0},
    "hrnr_0p2": {"harmonic_weight": 0.2},
    "noise_0p7": {"noise_scale": 0.7},
    "noise_0p85": {"noise_scale": 0.85},
    "noise_1p15": {"noise_scale": 1.15},
    "noise_1p3": {"noise_scale": 1.3},
    "dry_0p05": {"dry_mix": 0.05},
    "dry_0p1": {"dry_mix": 0.1},
    "dry_0p2": {"dry_mix": 0.2},
    "floor_0p05": {"floor": 0.05},
    "floor_0p15": {"floor": 0.15},
    "floor_0p2": {"floor": 0.2},
    "freq_0p5": {"smooth_freq": 0.5},
    "freq_1p0": {"smooth_freq": 1.0},
    "time_0p5": {"smooth_time": 0.5},
    "smooth_both": {"smooth_freq": 0.7, "smooth_time": 0.5},
    "fft_512": {"n_fft": 512, "hop": 128},
    "fft_768": {"n_fft": 768, "hop": 192},
}


def speaker(segment_id):
    return str(segment_id).split("_", 1)[0]


def assign_subsets(ids):
    """Speaker-disjoint, deterministic holdout (20%, at least one speaker)."""
    speakers = sorted({speaker(x) for x in ids},
                      key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < 2:
        raise ValueError("Need >=2 speakers for tune/holdout split")
    count = max(1, round(len(speakers) * 0.2))
    held = set(speakers[-count:])
    return ["holdout" if speaker(x) in held else "tune" for x in ids]


class ValidationData:
    def __init__(self, data):
        self.root = Path(data)
        self.archive = (zipfile.ZipFile(self.root) if self.root.is_file()
                        and self.root.suffix.lower() == ".zip" else None)
        if self.archive:
            candidates = [x for x in self.archive.namelist()
                          if x.endswith("validation/samples.csv")]
            if len(candidates) != 1:
                raise ValueError("Expected one validation/samples.csv in ZIP")
            self.prefix = candidates[0].split("validation/samples.csv")[0]
            self.samples = pd.read_csv(io.BytesIO(self.archive.read(candidates[0])))
        else:
            self.samples = pd.read_csv(self.root / "validation" / "samples.csv")

    def audio(self, relative):
        relative = str(relative).replace("\\", "/")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid audio path")
        if self.archive:
            x, sr = sf.read(io.BytesIO(self.archive.read(self.prefix + relative)))
        else:
            x, sr = sf.read(self.root / relative)
        return np.asarray(x, dtype=np.float64), sr

    def close(self):
        if self.archive:
            self.archive.close()


def score(clean, estimate, sr, metric, pesq_fn):
    if sr != 16000 or clean.ndim != 1 or clean.shape != estimate.shape:
        raise ValueError("Expected aligned 16kHz mono audio")
    if not np.isfinite(estimate).all():
        raise ValueError("Nonfinite samples from enhancer")
    err = np.sum((clean - estimate) ** 2)
    snr = float(10.0 * np.log10(np.sum(clean ** 2) / max(err, 1e-18)))
    quality = (float(pesq_fn(sr, clean, estimate, "wb"))
               if metric == "pesq" else float("nan"))
    return snr, quality


def run(data, output_dir, metric="pesq", limit=0, configs=None):
    if metric not in ("pesq", "snr"):
        raise ValueError("Unknown metric")
    pesq_fn = None
    if metric == "pesq":
        try:
            from pesq import pesq
        except ImportError as err:
            raise RuntimeError("Install PESQ: pip install pesq==0.0.4") from err
        pesq_fn = pesq
    names = list(EXPERIMENTS) if configs is None else list(configs)
    if not names or any(x not in EXPERIMENTS for x in names):
        raise ValueError("Unknown/empty --configs")
    ds = ValidationData(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.samples["subset"] = assign_subsets(ds.samples.segment_id)
    if limit:
        ds.samples = ds.samples.head(limit)
    records = []
    try:
        for i, row in enumerate(ds.samples.itertuples(index=False), 1):
            clean, sr = ds.audio(row.clean_path)
            noisy, nsr = ds.audio(row.noisy_path)
            if sr != nsr or clean.shape != noisy.shape:
                raise ValueError(f"Unaligned clean/noisy: {row.segment_id}")
            info = {"segment_id": row.segment_id, "subset": row.subset,
                    "noise_category": getattr(row, "noise_category", ""),
                    "target_snr_db": getattr(row, "target_snr_db", np.nan)}
            snr, quality = score(clean, noisy, sr, metric, pesq_fn)
            records.append({**info, "config": "noisy", "snr_db": snr,
                            "pesq_wb": quality})
            for name in names:
                # CLEAN is intentionally not passed to any enhancement routine.
                y = enhance(noisy, sr, replace(BASE, **EXPERIMENTS[name]))
                snr, quality = score(clean, y, sr, metric, pesq_fn)
                records.append({**info, "config": name, "snr_db": snr,
                                "pesq_wb": quality})
            if i % 20 == 0:
                print(f"Processed {i}/{len(ds.samples)}", flush=True)
        frame = pd.DataFrame(records)
        frame.to_csv(output_dir / "per_file.csv", index=False)
        summary = frame.groupby(["subset", "config"], as_index=False).agg(
            n=("segment_id", "nunique"), mean_snr_db=("snr_db", "mean"),
            mean_pesq_wb=("pesq_wb", "mean"))
        summary.to_csv(output_dir / "summary_by_subset.csv", index=False)
        column = "mean_pesq_wb" if metric == "pesq" else "mean_snr_db"
        tune = summary[(summary.subset == "tune") & (summary.config != "noisy")]
        if tune.empty:
            raise ValueError("No tune samples; increase --limit")
        choice = tune.sort_values(column, ascending=False).iloc[0]
        chosen_name = str(choice["config"])
        held = summary[(summary.subset == "holdout")
                       & (summary.config == chosen_name)]
        report = {"selection_metric": column, "selected_config": chosen_name,
                  "parameters": asdict(replace(BASE, **EXPERIMENTS[chosen_name])),
                  "tune_score": float(choice[column]),
                  "holdout_score": None if held.empty else float(held.iloc[0][column]),
                  "n_files": int(frame.segment_id.nunique()),
                  "note": "Selection used tune speakers only. Hidden-test PESQ is unknown."}
        (output_dir / "selection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(summary.sort_values(["subset", column], ascending=[True, False])
              .to_string(index=False))
        print("Selected on tune:", chosen_name,
              "; holdout:", report["holdout_score"])
        return frame, summary, report
    finally:
        ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="noise_reduction_challenge.zip")
    p.add_argument("--output-dir", default="results/pesq_search")
    p.add_argument("--metric", choices=("pesq", "snr"), default="pesq")
    p.add_argument("--limit", type=int, default=0, help="0=all 164 validation WAVs")
    p.add_argument("--configs", default="", help="comma-separated; blank=all")
    a = p.parse_args()
    run(a.data, a.output_dir, a.metric, a.limit,
        None if not a.configs else a.configs.split(","))
