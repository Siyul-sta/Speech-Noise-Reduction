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

python experiments/noisy_only_3plus_20261117/build_manifest.py --root noise_reduction_challenge --stage train --output results/three_plus_train_manifest.csv
python experiments/noisy_only_3plus_20261117/experiment.py --root noise_reduction_challenge --manifest results/three_plus_train_manifest.csv --output results/three_plus_train_screen.csv --max-files 96 --workers 4
# After selecting ONE candidate on Train (replace quiet_focus with that mode):
python experiments/noisy_only_3plus_20261117/build_manifest.py --root noise_reduction_challenge --stage validation --output results/three_plus_validation_manifest.csv
python experiments/noisy_only_3plus_20261117/experiment.py --root noise_reduction_challenge --manifest results/three_plus_validation_manifest.csv --output results/three_plus_validation.csv --modes spp25 quiet_focus --workers 4
```

The manifest builder discovers pairs from train/noisy and train/clean. Select one candidate on the 96-file spread-out Train screen, then compare it with spp25 on all 164 Validation files, retaining per-file results. This helps avoid repeated Validation-tuned parameter selection. Do not calibrate a unique gain from each file's Clean.

## Status

The GitHub branch contains source and a reproducible evaluator. **No public-Validation PESQ for the new variants has been produced here.** A separate 8-Train/9-Validation standalone spectral reconstruction screening (different engine) did not improve Train STOI and was rejected.

The reference-assisted 3.175/3.624 diagnostic experiments are excluded from the candidate set and cannot serve as submission scores.

## Limited exploratory PSD diagnostic (NOT the official final pipeline)

On a separate lightweight Noisy-only PSD estimator, the mean across per-file median absolute PSD errors in **noise-dominant** STFT bins was:

| Noise PSD | Train (8 pairs) | Validation (9 pairs) |
|---|---:|---:|
| local PSD | 4.2206 dB | 3.8710 dB |
| fixed SPP25 | 3.9732 dB | 3.5621 dB |
| quiet_focus | **3.7909 dB** | **3.3728 dB** |

For diagnostics ONLY, level-aligned Clean was used to derive the true noise and label noise-dominant bins. Neither the candidate nor the submission code consumes Clean. Because this used a lightweight proxy estimator, it does **not** establish improvement of the full submission's PESQ-WB. The independent standalone restoration candidate failed Train STOI screening and was not retained. Official `pesq` package was not available in this execution sandbox; the next required step is full-pipeline evaluation on the Train/Validation audio with that package installed.
