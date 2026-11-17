from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,json,hashlib
import numpy as np,pandas as pd,soundfile as sf
from bench import TA,BASE
from previous import enhance as old
from candidate import process
ROOT=Path(__file__).resolve().parent
def run(task):
 data,row,modes,stage=task
 c,x=TA['read_pair'](data,row)
 key=hashlib.sha256(x.tobytes()).hexdigest()
 cache=Path('/tmp/dsp_r4_cache');cache.mkdir(exist_ok=True)
 f=cache/(key+'.npy')
 if f.exists(): y=np.load(f)
 else:
  y=old(x,16000,replace(BASE,structure='fit_eq_erb'),estimator='quantile',blend=1)
  np.save(f,y)
 records=[]
 for mode in modes:
  z=process(x,y,mode,c if mode.startswith('oracle') else None)
  assert z.shape==x.shape and np.isfinite(z).all()
  from io import BytesIO
  b=BytesIO();sf.write(b,z,16000,format='WAV',subtype='PCM_16');b.seek(0);v,_=sf.read(b)
  p,e=TA['pesq_score'](c,v);assert not e
  records.append([stage,row['segment_id'],row['noise_category'],row['target_snr_db'],mode,p,TA['snr_db'](c,v)])
 return records
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('modes',nargs='+');p.add_argument('--data',type=Path,default=ROOT.parent/'noise_reduction_challenge');a=p.parse_args()
 d=pd.read_csv(a.data/'validation/samples.csv') if a.stage=='validation' else (pd.read_csv(ROOT/'extra.csv') if a.stage=='extra' else pd.read_csv(ROOT/'train_split.csv').query('stage==@a.stage'))
 with ProcessPoolExecutor(7) as pool: rows=sum(list(pool.map(run,[(a.data,r,a.modes,a.stage) for r in d.to_dict('records')])),[])
 df=pd.DataFrame(rows,columns=['stage','id','noise','target','mode','pesq','snr']);out=ROOT/'results';out.mkdir(exist_ok=True)
 df.to_csv(out/(a.stage+'_'+'-'.join(a.modes)+'.csv'),index=False)
 print(df.groupby('mode')[['pesq','snr']].mean().sort_values('pesq',ascending=False))
