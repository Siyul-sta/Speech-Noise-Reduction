# Engine integration smoke test — 2026-11-10

Two teammate-authored DSP engines were added to the draft PR **without modifying their signal processing implementation**. The common adapter was tested on five validation WAV pairs from the supplied class dataset (16kHz mono, 48000 samples each). Measurements below are **SNR only** (PESQ was not installed in this environment). The full Validation average is NOT yet established.

| Validation segment | Noisy SNR | Engine A SNR | Engine B SNR |
|---|---:|---:|---:|
| speaker015_utt01_seg0001 | 10.00 | 14.76 | 15.48 |
| speaker015_utt01_seg0002 | 15.00 | 6.70 | 16.82 |
| speaker015_utt01_seg0003 | 5.00 | 9.24 | 14.97 |
| speaker015_utt01_seg0004 | 15.00 | 16.62 | 18.32 |
| speaker015_utt01_seg0005 | 15.00 | 9.10 | 17.87 |

**Important:** Engine A worsened SNR for multiple clips, especially segment 0002. Do not assume it is optimal. Both algorithms produced finite, correctly sized outputs, but these are not PESQ-WB rankings. Input normalization and STFT interface compatibility require full validation.

## Before merge / submission
1. Run `python -m src.evaluation --data noise_reduction_challenge` in a clean environment with dependencies installed and confirm all 164 reference pairs produce PESQ-WB.
2. Compare mean PESQ-WB plus per-noise category metrics; do not optimize only SNR.
3. Run `python main.py --input input --output output --method a` and `--method b` on mono and stereo WAVs to verify filename, frame count, sample rate, and channel count.
4. Document academic references, AI assistance, and teammate authorship.
5. Keep the PR in draft until validation and review approval.

