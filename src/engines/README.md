# Independent DSP engines

Two original teammate submissions should be placed here **without altering their provenance**:

- `speech_gain_engine.py`: IMCRA + OM-LSA + modulation-domain Kalman + adaptive fusion (exports `compute_gain_matrix(magnitude, power)`).
- `tsnr_hrnr_v3.py`: TSNR + HRNR + Wiener (exports `enhance(x, sr, Config())`).

These source files are **not included in this integration PR**. Each author should upload their reviewed original through their own feature branch and PR. Never claim that the integration pipeline is runnable before both engine files are present.

The A adapter uses scipy STFT: Hann, FFT=512, hop=128. B uses its author's own STFT configuration. Both process channels independently and return exactly the source number of samples.

Do not upload clean/noisy datasets to public GitHub without authorization.
