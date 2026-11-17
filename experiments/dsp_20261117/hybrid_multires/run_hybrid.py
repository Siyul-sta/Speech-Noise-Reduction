"""Standalone optimized candidate runner; --estimator v3 keeps original baseline."""
from pathlib import Path
import argparse
import json
from dataclasses import replace
import numpy as np
import soundfile as sf
from hybrid_v3 import enhance, Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='input', type=Path)
    p.add_argument('--output', default='output', type=Path)
    p.add_argument('--estimator', choices=['v3', 'imcra', 'spp', 'quantile'], default='quantile')
    p.add_argument('--blend', type=float, default=None)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('best_config.json'))
    a = p.parse_args()
    settings = json.loads(a.config.read_text()) if a.estimator != 'v3' else {'config': {}, 'blend': 0.0}
    cfg = replace(Config(), **settings['config'])
    blend = settings['blend'] if a.blend is None else a.blend
    if a.input.resolve() == a.output.resolve():
        p.error('Input and output directories must differ')
    paths = sorted(a.input.glob('*.wav'))
    if not paths:
        p.error('No input WAVs found')
    a.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        x, sr = sf.read(path, dtype='float64', always_2d=True)
        y = np.column_stack([enhance(x[:, k], sr, cfg, estimator=a.estimator, blend=blend) for k in range(x.shape[1])])
        if y.shape != x.shape or not np.isfinite(y).all():
            raise ValueError(f'Invalid output: {path.name}')
        sf.write(a.output / path.name, np.clip(y, -1, 1), sr, subtype=sf.info(path).subtype)
        info = sf.info(a.output / path.name)
        assert (info.frames, info.channels, info.samplerate) == (len(x), x.shape[1], sr)
    print(f'Processed {len(paths)} WAV files')

if __name__ == '__main__':
    main()
