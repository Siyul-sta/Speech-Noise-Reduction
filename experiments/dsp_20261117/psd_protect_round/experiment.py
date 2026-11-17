"""Reduce upward noise PSD modulation when noisy spectral speech evidence is strong."""
import importlib.util, argparse, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shape_experiment', ROOT / 'shape_round/experiment.py')
shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape)
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import ndimage
from bench import TA
import noisy_estimators


def estimate(x, mode):
    original = noisy_estimators.estimate

    def protected(power, initial, full, estimator_mode):
        baseline = original(power, initial, full, estimator_mode)
        amount = float(mode.split('_')[1])
        smoothed = ndimage.gaussian_filter(power, (1.5, 1))
        ratio = smoothed / np.maximum(baseline, 1e-18)
        evidence = np.clip((ratio - 2) / 8, 0, 1)
        anchor = np.quantile(baseline[:, full], 0.2, axis=1, keepdims=True)
        excess = np.maximum(np.log(np.maximum(baseline, 1e-18) / np.maximum(anchor, 1e-18)), 0)
        return baseline * np.exp(-amount * evidence * excess)

    if mode != 'baseline':
        noisy_estimators.estimate = protected
    try:
        return shape.estimate(x, 'noise_4')
    finally:
        noisy_estimators.estimate = original


def score(task):
    stage,row,modes=task;c,x=TA['read_pair'](ROOT/'noise_reduction_challenge',row);records=[]
    for mode in modes:
        y=estimate(x,mode);b=io.BytesIO();sf.write(b,y,16000,format='WAV',subtype='PCM_16');b.seek(0);decoded,_=sf.read(b)
        p,e=TA['pesq_score'](c,decoded)
        if e:raise ValueError(e)
        records.append(dict(stage=stage,id=row['segment_id'],noise=row['noise_category'],target=row['target_snr_db'],mode=mode,pesq=p,snr=TA['snr_db'](c,decoded)))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('modes',nargs='+');a=p.parse_args()
    if a.stage=='validation':d=pd.read_csv(ROOT/'noise_reduction_challenge/validation/samples.csv')
    else:d=pd.read_csv(ROOT/('r6/extra.csv' if a.stage=='extra' else 'r6/train_split.csv')).query('stage==@a.stage')
    with ProcessPoolExecutor(6) as pool:rows=sum(pool.map(score,[(a.stage,r,a.modes) for r in d.to_dict('records')]),[])
    df=pd.DataFrame(rows);df.to_csv(Path(__file__).parent/(a.stage+'_'+'-'.join(a.modes)+'.csv'),index=False)
    print(df.groupby('mode')[['pesq','snr']].mean().to_string())
