"""Score saved WAVs only; this evaluator is never imported by inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from pystoi import stoi


def snr(clean, y):
    energy, error = np.sum(clean**2), np.sum((clean - y)**2)
    if energy <= 0:
        raise ValueError('Silent reference')
    return float(np.inf if error == 0 else 10 * np.log10(energy / error))


def score(task):
    data, output, row = task
    clean, rate = sf.read(data / row['clean_path'], dtype='float64')
    noisy, noisy_rate = sf.read(data / row['noisy_path'], dtype='float64')
    destination = output / Path(row['noisy_path']).name
    y, out_rate = sf.read(destination, dtype='float64')
    if rate != 16000 or noisy_rate != rate or out_rate != rate:
        raise ValueError('PESQ-WB evaluation requires 16 kHz')
    if clean.shape != noisy.shape or y.shape != noisy.shape or not np.isfinite(y).all():
        raise ValueError(f'Invalid output: {destination}')
    valid = int(row.get('valid_samples', len(clean)))
    if not 0 < valid <= len(clean):
        raise ValueError('Invalid valid_samples')
    c, n, y = clean[:valid], noisy[:valid], y[:valid]
    return dict(segment_id=row['segment_id'], noise=row['noise_category'],
                target=row['target_snr_db'], pesq=float(pesq(rate, c, y, 'wb')),
                snr=snr(c, y), stoi=float(stoi(c, y, rate, extended=False)),
                noisy_pesq=float(pesq(rate, c, n, 'wb')), noisy_snr=snr(c, n),
                frames=sf.info(destination).frames, channels=sf.info(destination).channels)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path, help='Folder containing validation/samples.csv')
    p.add_argument('--output', required=True, type=Path, help='Previously emitted WAV directory')
    p.add_argument('--report', type=Path, default=Path('results/final_submission'))
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    manifest = pd.read_csv(args.data / 'validation/samples.csv')
    if manifest.empty or manifest.segment_id.duplicated().any():
        raise ValueError('Empty manifest or duplicate IDs')
    tasks = [(args.data, args.output, row) for row in manifest.to_dict('records')]
    with ProcessPoolExecutor(args.workers) as pool:
        detail = pd.DataFrame(pool.map(score, tasks))
    if len(detail) != len(manifest) or not np.isfinite(detail[['pesq', 'snr', 'stoi']]).all().all():
        raise ValueError('Incomplete evaluation')
    args.report.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.report / 'validation_per_file.csv', index=False)
    summary = dict(n_total=len(detail), n_success=len(detail), n_failed=0,
                   **detail[['pesq', 'snr', 'stoi', 'noisy_pesq', 'noisy_snr']].mean().to_dict())
    (args.report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for column in ('noise', 'target'):
        detail.groupby(column)[['pesq', 'snr', 'stoi']].mean().to_csv(args.report / f'by_{column}.csv')
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       versions={name: importlib.metadata.version(name) for name in
                                 ('numpy', 'scipy', 'soundfile', 'pandas', 'pesq', 'pystoi')},
                       mode='PESQ-WB 16 kHz, PCM16 decoded outputs',
                       aggregation='Unweighted arithmetic mean across files',
                       snr='10*log10(sum(clean**2)/sum((clean-output)**2)), no level alignment')
    (args.report / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
