# Independent teammate DSP engines

This directory contains the two submitted engine implementations (original algorithms, no tuning modifications made during integration):

- `speech_gain_engine.py` — IMCRA + OM-LSA + modulation-domain Kalman + adaptive fusion. The `compute_gain_matrix(magnitude, power)` entry point returns a (frequency bins, time frames) gain matrix.
- `tsnr_hrnr_v3.py` — TSNR + HRNR + Wiener, exposes `enhance(x, sr, Config())`.

The integration layer `src/engine_runner.py` handles common interfaces. Engine A uses SciPy STFT with a 512-point Hann window and hop 128. Engine B retains its own 1024-point STFT default.

## Review status
- Both original engine files compile and run on representative synthetic inputs.
- Five actual paired validation files were tested with equal sample lengths and finite outputs.
- **Engine A underperformed the noisy input on some of these five validation clips.** Do not choose it just because it has a more elaborate pipeline.
- Full mean PESQ-WB on all 164 validation files has **not** been measured, and no algorithm is approved as the final challenge submission.
- Please preserve author attribution and obtain approval for sharing course datasets; do not commit the WAV dataset.

See `docs/ENGINE_REVIEW.md` for the limited smoke-test evidence.
