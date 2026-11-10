# Dual-engine experiment and submission plan

## Goal
Compare both teammate-written traditional DSP algorithms under the same PESQ-WB evaluation. Preserve all original notebook baselines.

- Engine A: IMCRA / OM-LSA / Modulation Kalman / Adaptive Fusion.
- Engine B: TSNR / HRNR / Wiener v3.

## Rules
1. Enhancement code must receive only **noisy** audio. Clean speech is used only by the separate evaluation script.
2. Process all `input/*.wav` into `output/<same-name>.wav`, preserving sampling rate, channels and frame count.
3. Compare mean PESQ-WB first; SNR and STOI are supplementary.
4. Do not choose A or B by using per-file clean-reference PESQ during hidden-test inference.
5. Do not implement unsupervised auto routing or gain fusion until a documented selection rule has been validated without access to the clean hidden-test references.
6. Tag the exact submission commit and document the associated validation score and references.

## Evaluation phases
- Phase 1: Run A/B separately on the same validation WAV pairs.
- Phase 2: Report mean and per-file PESQ, SNR; compute oracle score **offline only** to quantify the theoretical benefit of file-wise selection.
- Phase 3: If oracle advantage is meaningful, devise a deterministic rule based on noisy-only features; compare it against the best fixed method on held-out data.
- Phase 4: Consider gain fusion only after comparable STFT grids and consistent phase conventions are verified.

## Team integration
Engine authors submit independent PRs to `src/engines/`. Integration reviewer verifies tests, installation, run instructions, and final tags before merging.

## CLI
`python main.py --input input --output output --method a`
`python main.py --input input --output output --method b`
