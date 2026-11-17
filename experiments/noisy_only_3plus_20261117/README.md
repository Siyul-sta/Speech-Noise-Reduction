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

The GitHub branch contains source and a reproducible evaluator. A separate independent 9-file public-Validation subsample has now been scored with a locally built PESQ 0.0.5 engine. The full 164-file score has **not** been computed, and the 9-file numbers must not be described as official full-Validation results. A separate 8-Train/9-Validation standalone spectral reconstruction screening (different engine) did not improve Train STOI and was rejected.

The reference-assisted 3.175/3.624 diagnostic experiments are excluded from the candidate set and cannot serve as submission scores.

## Limited exploratory PSD diagnostic (NOT the official final pipeline)

On a separate lightweight Noisy-only PSD estimator, the mean across per-file median absolute PSD errors in **noise-dominant** STFT bins was:

| Noise PSD | Train (8 pairs) | Validation (9 pairs) |
|---|---:|---:|
| local PSD | 4.2206 dB | 3.8710 dB |
| fixed SPP25 | 3.9732 dB | 3.5621 dB |
| quiet_focus | **3.7909 dB** | **3.3728 dB** |

For diagnostics ONLY, level-aligned Clean was used to derive the true noise and label noise-dominant bins. Neither the candidate nor the submission code consumes Clean. Because this used a lightweight proxy estimator, it does **not** establish improvement of the full submission's PESQ-WB. The independent standalone restoration candidate failed Train STOI screening and was not retained. Official `pesq` package was not available in this execution sandbox; the next required step is full-pipeline evaluation on the Train/Validation audio with that package installed.

## Measured PESQ-WB on 9 Validation WAV pairs (incomplete sample)

The sample covers five Validation speakers, selected earlier for inspection from Google Drive. 16 kHz mono WAV, Noisy-only inference, PCM16 write/read before `pesq(sr, clean, enhanced, 'wb')`. Clean entered **only the evaluator**.

| Method | Mean PESQ-WB | Difference vs same-9 baseline | Better / worse files |
|---|---:|---:|---:|
| baseline | 2.979086 | 0 | — |
| spp25 | 2.976089 | -0.002997 | 4 / 5 |
| quiet_focus | 2.976911 | -0.002175 | 4 / 5 |
| speech_guard | 2.977453 | -0.001633 | 6 / 3 |
| down_guard | **2.981761** | **+0.002675** | 7 / 2 |
| nonstationary | 2.979433 | +0.000347 | 6 / 3 |

**Do not compare these subset means against the 164-file 2.857110 result**. Likewise, 2.981761 is not a 164-file Validation score. The earlier lightweight PSD diagnostic ranks quiet_focus highest on its own error metric but that does not transfer to the PESQ outcome. The branch's `candidate.py` collision with the existing r6 module has been fixed by renaming to `noise_candidate.py`. No changes or merges into `main` were made.

## Full public Validation PESQ-WB — 164 / 164 files (2026-11-17)

All 164 paired Google Drive Clean/Noisy Validation WAVs were evaluated using 16-kHz mono PCM16, the archived Noisy-only second-submission enhancement pipeline, PCM16 output serialization and `pesq(16000, clean, enhanced, "wb")` **after** inference. The historical full-Validation controls were reproduced exactly to six decimals (baseline 2.848777; SPP25 2.857110).

| Rank | Model | Mean PESQ-WB (164) | Delta vs baseline | Improved / regressed vs baseline |
|---:|---|---:|---:|---:|
| 1 | SPP25 | **2.857110** | +0.008332 | 99 / 65 |
| 2 | quiet_focus | 2.853519 | +0.004741 | 99 / 65 |
| 3 | speech_guard | 2.852237 | +0.003460 | 100 / 64 |
| 4 | nonstationary | 2.851782 | +0.003005 | 102 / 62 |
| 5 | down_guard | 2.850549 | +0.001772 | 124 / 40 |
| 6 | baseline | 2.848777 | — | — |

Original Noisy: **1.394672** mean PESQ-WB. No proposed alternative exceeded SPP25 or achieved the 3.0 target. The prior 9-file down_guard lead did not generalize (9-file 2.981761 versus full-set 2.850549). Although down_guard improves 124/164 files over baseline, its gains are too small relative to the SPP25 gains to lead on mean PESQ.

**Decision: do NOT merge this research branch to main.** The selected full-Validation control remains SPP25 at **2.857110**. The original `main` branch was not changed by the validation run.
