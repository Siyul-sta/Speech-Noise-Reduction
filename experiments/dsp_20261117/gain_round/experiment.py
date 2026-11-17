"""Compare first-stage gain rules with the selected noisy-only PSD fixed."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'psd_round'))
# Import PSD runner under its own name to avoid this file's module name.
import importlib.util
spec=importlib.util.spec_from_file_location('psd_experiment',ROOT/'psd_round/experiment.py')
psd=importlib.util.module_from_spec(spec);spec.loader.exec_module(psd)
import argparse,io
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,soundfile as sf
from scipy.special import exp1,i0e,i1e
import core
from bench import TA
from candidate import process

def estimate(x,mode):
    def gain(xi,gamma,floor):
        kind,amount=mode.split('_');amount=float(amount)
        xi=np.maximum(xi,1e-10);gamma=np.maximum(gamma,1e-10)
        w=xi/(1+xi);v=np.maximum(w*gamma,1e-10)
        if kind in ('lsa','lsafinal'):g=w*np.exp(.5*exp1(v))
        elif kind in ('stsa','stsafinal'):g=np.sqrt(np.pi)/2*np.sqrt(v)/gamma*((1+v)*i0e(v/2)+v*i1e(v/2))
        elif kind=='sqrt':g=np.sqrt(w)
        else:raise ValueError(mode)
        return np.clip((1-amount)*w+amount*np.clip(g,floor,1),floor,1)
    gain.final_only='final' in mode
    core.GAIN_RULE=None if mode=='baseline' else gain
    try:pilot=psd.first_pass(x,'local_15_0.5')
    finally:core.GAIN_RULE=None
    return process(x,pilot,'hi_floor_tone_1.0')

def score(task):
    stage,row,modes=task;c,x=TA['read_pair'](ROOT/'noise_reduction_challenge',row);rows=[]
    for mode in modes:
        y=estimate(x,mode);b=io.BytesIO();sf.write(b,y,16000,format='WAV',subtype='PCM_16');b.seek(0);y,_=sf.read(b)
        p,e=TA['pesq_score'](c,y)
        if e:raise ValueError(e)
        rows.append(dict(stage=stage,id=row['segment_id'],noise=row['noise_category'],target=row['target_snr_db'],mode=mode,pesq=p,snr=TA['snr_db'](c,y)))
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('modes',nargs='+');a=p.parse_args()
    if a.stage=='validation':d=pd.read_csv(ROOT/'noise_reduction_challenge/validation/samples.csv')
    else:d=pd.read_csv(ROOT/('r6/extra.csv' if a.stage=='extra' else 'r6/train_split.csv')).query('stage==@a.stage')
    with ProcessPoolExecutor(6) as pool:rows=sum(pool.map(score,[(a.stage,r,a.modes) for r in d.to_dict('records')]),[])
    d=pd.DataFrame(rows);d.to_csv(Path(__file__).parent/(a.stage+'_'+'-'.join(a.modes)+'.csv'),index=False)
    print(d.groupby('mode')[['pesq','snr']].mean().to_string())
