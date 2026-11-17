# Noisy-only PESQ 3.0+ research branch (2026-11-17)

**Goal, not a measured score:** PESQ-WB >= 3.0 on the public Validation set. Current reproducible target control: SPP 25% mixture **2.857110** (164 files). The original submission pipeline is **2.848777**. The PESQ difference needed is **+0.142890**.

## Rules and separation
- The **enhancement code reads only Noisy WAV**. No Clean WAV, reference noise, oracle PSD, stored evaluation labels, or train-time noise file is required at inference.
- **Only** `experiment.py` reads Clean, solely to compute PESQ-WB, STOI and SNR **after** inference.
- These are exploratory variants. None is claimed to outperform 2.857110 without measurement. Do not replace the official root `main.py` with an unverified variant.

## Algorithms

- `baseline`: reproduces the baseline Noisy-only local noise PSD
- `spp25`: reproduces the previously measured 25% SPP geometric blend (positive control)
- `quiet_focus`: speech-presence weighted confidence; more SPP in noise-only bins
- `speech_guard`: suppresses upward SPP noise-PSD errors in speech-dominant bins
- `down_guard`: only permits SPP to reduce overestimated baseline PSD
- `nonstationary`: favors SPP in rapidly varying noise PSD regions, reduces it when speech is present

All variants keep the same second-submission reconstruction stages and original WAV serialization. Importantly, none makes use of a reference-noise file.

## Execution

Run from repository root, with Python dependencies in root `requirements.txt` installed:

```bash
python experiments/noisy_only_3plus_20261117/main.py --input input --output output --mode spp25

python experiments/noisy_only_3plus_20261117/experiment.py --root noise_reduction_challenge --manifest noise_reduction_challenge/train/samples.csv --output results/three_plus_train.csv --workers 4
python experiments/noisy_only_3plus_20261117/experiment.py --root noise_reduction_challenge --manifest noise_reduction_challenge/validation/samples.csv --output results/three_plus_validation.csv --workers 4
```

If Train has no `samples.csv`, use the manifest supplied with the dataset or create a CSV with `segment_id,noisy_path,clean_path`, with paths relative to the dataset root. Always select a single candidate on Train before judging all 164 Validation examples; retain per-file results. Do not calibrate a unique gain from each file's Clean.

## Status

The GitHub branch contains source and a reproducible evaluator. **No public-Validation PESQ for the new variants has been produced here.** A separate 8-Train/9-Validation standalone spectral reconstruction screening (different engine) did not improve Train STOI and was rejected.

The reference-assisted 3.175/3.624 diagnostic experiments are excluded from the candidate set and cannot serve as submission scores.
