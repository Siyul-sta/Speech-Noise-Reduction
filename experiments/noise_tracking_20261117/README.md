# Weak SPP noise tracking blend

This experiment preserves the second-submission pipeline and geometrically blends its noise PSD with the existing bidirectional speech presence probability tracker. The SPP weight is 0.25. Inference uses noisy audio only.

| Data | Files | Baseline PESQ-WB | Candidate PESQ-WB |
|---|---:|---:|---:|
| Train screening | 32 | 3.116883 | 3.124423 |
| Other Train speakers | 48 | 2.904102 | 2.906764 |
| Validation | 164 | 2.848777 | 2.857110 |

Validation improved by 0.008332 across the mean, with 99 improvements and 65 regressions. The largest regression was -0.059300 and largest improvement +0.116923. SNR changed from 2.278749 to 2.276238 dB. All scores use PCM16 round-trip waveforms. The baseline reproduces the submission score. This is a small development-set improvement; hidden-test generalization is unmeasured.

Train selected the 0.25 blend from four candidates. Validation evaluated only that selected candidate. Existing public Validation has been examined repeatedly in earlier experiments.

## Noise audit

In Train screening, the mean across per-file/per-resolution median PSD errors showed the SPP tracker reduced absolute error in noise-dominant bins from 6.319 to 4.931 dB. In speech-dominant bins, absolute error increased from 7.702 to 9.649 dB. Median signed bias increased from +5.795 to +9.175 dB in speech-dominant bins. The weak blend moderates this tradeoff.

The audit uses level-aligned Clean to derive diagnostic noise and label bins, and does not select parameters per file. Details are in train_noise_audit.csv. Previous reference-assisted diagnostic scores are archived in previous_validation.csv and require Clean.

## Run from repository root

Install root requirements.txt first.

```bash
python experiments/noise_tracking_20261117/main.py --input input --output output
```

The candidate uses the standard WAV writer and input format preservation logic with one worker.

Reproduce evaluation by replacing DATA_ROOT with the challenge dataset root.

```bash
python experiments/noise_tracking_20261117/experiment.py --root DATA_ROOT --manifest DATA_ROOT/validation/samples.csv --output validation.csv --variants baseline blend_0.25
python experiments/noise_tracking_20261117/audit.py --root DATA_ROOT --manifest experiments/post_submission_20261117/screen_manifest.csv --output noise_audit.csv
```

The tracker reuses the SPP implementation in r6/enhance_gain.py and its documented Gerkmann and Hendriks (2012) basis. No new external implementation was imported. The official second-submission commit remains 418790027ff48c1e4149cc3db371f10f80b15bac.
