"""First-stage PSD replacement, with diagnostic clean access confined to oracle modes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'r6'))
import argparse,json,io
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal,ndimage
import core
from bench import TA,BASE
from previous import enhance
from candidate import process
ROOT=Path(__file__).resolve().parents[1]

def first_pass(x, mode, clean=None):
    def estimator(power,initial,full,actual,sr,nfft,hop):
        if mode.startswith(('local_', 'quiet_')):
            from noisy_estimators import estimate
            return estimate(power,initial,full,mode)
        if mode.startswith('oracle'):
            residual=actual-clean
            noise=abs(signal.stft(residual,fs=sr,nperseg=nfft,noverlap=nfft-hop)[2])**2
            return np.maximum(ndimage.gaussian_filter(noise,(1,2)),1e-18)
        # Speech-presence probability slows updates during excess spectral energy.
        previous=initial.copy()
        tracked=np.empty_like(power)
        smoothed=ndimage.gaussian_filter(power,(1,1))
        strength=float(mode.rsplit('_',1)[1])
        for t in range(power.shape[1]):
            ratio=smoothed[:,t]/np.maximum(previous,1e-18)
            speech=np.clip((ratio-1.5)/4,0,1)
            rate=.08*(1-speech)+.001*speech
            if mode.startswith('spp'):
                speech=1/(1+32.6*np.exp(-np.minimum(ratio,100)*31.6/32.6))
                rate=.02*(1-speech)
            previous=(1-rate)*previous+rate*smoothed[:,t]
            tracked[:,t]=previous
        from adaptive import relative_noise
        baseline=relative_noise(power,initial,full)
        return np.exp((1-strength)*np.log(np.maximum(baseline,1e-18))+strength*np.log(np.maximum(tracked,1e-18)))
    core.NOISE_ESTIMATOR=None if mode in ('baseline','baseline_plain','baseline_unshaped') else estimator
    try:
        return enhance(x,16000,replace(BASE,structure='fit_eq_erb'),estimator='quantile',blend=1)
    finally:
        core.NOISE_ESTIMATOR=None

def run(task):
    row,modes,stage=task
    clean,x=TA['read_pair'](ROOT/'noise_reduction_challenge',row)
    rows=[]
    for mode in modes:
        pilot=first_pass(x,mode,clean if mode.startswith('oracle') else None)
        output=pilot if mode.endswith('_plain') else process(x,pilot,'hi_floor_tone_1.0')
        if mode.endswith('_unshaped'):
            from hi_floor import restore
            output=restore(x,process(x,pilot,'v5fuse_unshaped'),'hi_floor_tone_1.0')
        b=io.BytesIO();sf.write(b,output,16000,format='WAV',subtype='PCM_16');b.seek(0);y,_=sf.read(b)
        p,e=TA['pesq_score'](clean,y)
        if e:raise ValueError(e)
        rows.append(dict(stage=stage,id=row['segment_id'],noise=row['noise_category'],target=row['target_snr_db'],mode=mode,pesq=p,snr=TA['snr_db'](clean,y)))
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('modes',nargs='+');a=p.parse_args()
    if a.stage=='validation':d=pd.read_csv(ROOT/'noise_reduction_challenge/validation/samples.csv')
    else:d=pd.read_csv(ROOT/('r6/extra.csv' if a.stage=='extra' else 'r6/train_split.csv')).query('stage==@a.stage')
    with ProcessPoolExecutor(6) as pool:rows=sum(pool.map(run,[(r,a.modes,a.stage) for r in d.to_dict('records')]),[])
    df=pd.DataFrame(rows);df.to_csv(Path(__file__).parent/(a.stage+'_'+'-'.join(a.modes)+'.csv'),index=False)
    print(df.groupby('mode')[['pesq','snr']].mean().to_string())
