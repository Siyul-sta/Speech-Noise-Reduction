# 최신 DSP 실험 결과

최고 공개 Validation PESQ-WB **2.848777**인 코드와 전체 실험 기록을 [2026-11-17 실험 폴더](experiments/dsp_20261117/README.md)에 추가했습니다. 실행은 해당 폴더의 `depth_round/main.py --mode 12_1500`을 사용합니다. 공개 Validation 반복 사용 및 SNR·STOI 변화는 실험 README에서 확인할 수 있습니다.

---

# Speech-Noise-Reduction

2026 Signals and Systems · Classical DSP Speech Noise Reduction Challenge.

## Engines
- **A** — IMCRA + OM-LSA + modulation-domain Kalman + adaptive fusion.
- **B** — TSNR + HRNR + Wiener v3.

**Integration status:** The adapter, batch runner, reference-based evaluator, and both independently authored engines are on this feature branch. Both teammate engine files are now present on this feature branch for review. Integration has passed limited smoke checks on 5 validation samples, but full PESQ-WB evaluation and clean-environment installation have **not** been completed. Do not submit this branch as a validated final candidate.

## Setup
Requires Python 3.10+ and the exact engine files above.

```bash
python -m pip install -r requirements.txt
```

## Run inference (no clean reference required)
```bash
python main.py --input input --output output --method a
python main.py --input input --output output --method b
```

Each input WAV gets an output with the same filename, sampling rate, number of channels and samples.

## Evaluate on provided validation data
```bash
python -m src.evaluation --data noise_reduction_challenge --output results/dual_engine_validation.csv
```

Validation expects `validation/samples.csv` plus paired clean/noisy WAVs in the official dataset. CSV includes per-file PESQ-WB, SNR and a **reference-only oracle analysis**; oracle is not a permissible hidden-test inference strategy.

## Auto selection
Not enabled. Per-file clean PESQ is inaccessible on hidden tests. A future noisy-only deterministic router must demonstrate generalization on held-out validation data before being enabled.

## Provenance
Reference notebooks are preserved under `notebooks/`. Record authors, literature, AI usage, parameter settings and results in `docs/`. Never upload course-provided audio datasets without distribution permission.

## Submission
Select one validated default method, tag the exact submission commit and provide the tag + AI usage report on LearnUs.
