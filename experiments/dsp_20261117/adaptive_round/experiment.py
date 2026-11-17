"""Adjust final attenuation from noisy-only spectral confidence."""
import importlib.util,argparse,io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gain_experiment',ROOT/'gain_round/experiment.py')
gain=importlib.util.module_from_spec(spec);spec.loader.exec_module(gain)
import numpy as np,pandas as pd,soundfile as sf
from scipy import ndimage
from bench import TA
import pockets


def estimate(x, mode):
    def depth_rule(freq, speech_spec, noisy_spec, base_depth):
        base_text, slope_text = mode.split('_')
        base_amount = float(base_text)
        slope = float(slope_text)
        center = 1500
        speech = ndimage.gaussian_filter(abs(speech_spec)**2,(2,1.5))
        noisy = ndimage.gaussian_filter(abs(noisy_spec)**2,(2,1.5))
        background = np.quantile(noisy,.2,axis=1,keepdims=True)
        evidence = speech / (speech + background + 1e-18)
        weight = np.exp(-0.5 * (np.log2(np.maximum(freq, 1) / float(center)) / 0.6)**2)
        
        global_evidence = float(np.mean(evidence))
        amount = np.clip(base_amount + slope * (global_evidence - 0.5), 6, 18)
        return np.maximum(base_depth[:, None] + 4 * (1 - evidence) + amount * weight[:, None], 0)
    pockets.DEPTH_RULE=depth_rule
    try:return gain.estimate(x,'lsa_0.25')
    finally:pockets.DEPTH_RULE=None


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
