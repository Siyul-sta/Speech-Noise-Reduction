"""Enhance 16-kHz mono challenge WAVs using noisy input only."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
from previous import Config,enhance
from candidate import process

ROOT=Path(__file__).resolve().parent
def render(src,dst,mode):
    x,sr=sf.read(src,dtype='float64')
    if sr!=16000 or x.ndim!=1 or not len(x):raise ValueError(f'Expected nonempty mono 16 kHz: {src}')
    cfg=Config(**json.loads((ROOT/'pesq_config.json').read_text())['config'])
    if len(x)<4096:raise ValueError('At least 4096 samples required; challenge WAVs have 48000.')
    first=enhance(x,sr,cfg,estimator='quantile',blend=1.)
    y=process(x,first,mode)
    if y.shape!=x.shape or not np.isfinite(y).all():raise ValueError(f'Invalid result: {src}')
    sf.write(dst,y,sr,subtype='PCM_16')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('input'))
    p.add_argument('--output',type=Path,default=Path('output'))
    p.add_argument('--mode',choices=['combine_global','balanced_mix','base','v5res_500_0.48_16','v5form_fixed','v5mid_balance','v5res_500_0.32_20','v5safe_blend_4','v5carrier_skirt_20_16_0.7','v5carrier_skirt_24_16_0.7','v5carrier_span_1024_19_0.7','v5fuse_flux','v5fuse_soft','hi_floor_001','hi_floor_tone_1.0','hi_floor_noise_001','hi_floor_noiselocal_001','v5short_profile_-0.5_soft','v5short_profile_-1_hard'],default='v5fuse_soft')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve():p.error('Separate input/output directories required')
    paths=sorted(args.input.glob('*.wav'))
    if not paths:p.error('No WAV files found')
    args.output.mkdir(parents=True,exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(render,paths,[args.output/q.name for q in paths],[args.mode]*len(paths)))
