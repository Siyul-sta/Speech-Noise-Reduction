"""Score emitted PCM16 WAVs using the TA functions and STOI."""
from experiment import ROOT, TA
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from pystoi import stoi


def score(row):
    clean, noisy = TA['read_pair'](ROOT / 'noise_reduction_challenge', row)
    output = ROOT / 'bandwidth_round/wav_final' / Path(row['noisy_path']).name
    enhanced, rate = sf.read(output)
    assert rate == 16000 and enhanced.shape == clean.shape
    assert sf.info(output).subtype == 'PCM_16' and np.isfinite(enhanced).all()
    value, error = TA['pesq_score'](clean, enhanced)
    if error:
        raise ValueError(error)
    return dict(id=row['segment_id'], noise=row['noise_category'], target=row['target_snr_db'], pesq=value, snr=TA['snr_db'](clean, enhanced), stoi=stoi(clean, enhanced, rate, extended=False))


if __name__ == '__main__':
    rows = pd.read_csv(ROOT / 'noise_reduction_challenge/validation/samples.csv')
    with ProcessPoolExecutor(6) as pool:
        result = pd.DataFrame(pool.map(score, rows.to_dict('records')))
    result.to_csv(ROOT / 'bandwidth_round/validation_disk.csv', index=False)
    print(result[['pesq', 'snr', 'stoi']].mean().to_string())
