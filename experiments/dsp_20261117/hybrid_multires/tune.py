from pathlib import Path
import json,argparse,io
import numpy as np,pandas as pd,soundfile as sf
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from hybrid_v3 import enhance,Config as HConfig
P=argparse.ArgumentParser();P.add_argument('--data',type=Path,required=True);P.add_argument('--notebook',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args(); ROOT=a.data.resolve(); OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
nb=json.loads(a.notebook.read_text());
for i in (6,8,32):exec(compile(''.join(nb['cells'][i]['source']),f'TA cell {i}','exec'))
BASE=HConfig(); C={}
def add(n,blend=1.,**kw):C[n]=(blend,kw)
add('quantile_base')
for v in [.7,1.3,1.7,2.2]:add(f'scale_{v}',noise_scale=v)
for v in [.08,.25]:add(f'q_{v}',init_quantile=v)
for v in [1.5,6.]:add(f'dof_{v}',init_dof=v)
for v in [0.,1.]:add(f'shrink_{v}',init_shrink_beta=v)
add('adaptive_dof',init_adaptive_dof=True)
for v in [.5,.75]:add(f'blend_{v}',blend=v)
for f,t in [(0,.8),(.6,0),(.6,.8),(1.,1.5)]:add(f'smooth_{f}_{t}',smooth_freq=f,smooth_time=t)
for v in [.03,.2]:add(f'floor_{v}',floor=v)
add('dry_0.05',dry_mix=.05)
def pcm(x):
 b=io.BytesIO();sf.write(b,np.clip(x,-1,1),16000,format='WAV',subtype='PCM_16');b.seek(0);return sf.read(b,dtype='float64')[0]
def work(task):
 row,names=task;c,x=read_pair(ROOT,row,CFG);v=int(row.get('valid_samples',len(c)));c=c[:v];ns=snr_db(c,x[:v]);npesq,_=pesq_score(c,x[:v]);out=[]
 for name,blend,kw in names:
  y=pcm(enhance(x,16000,replace(BASE,**kw),estimator='quantile',blend=blend));s=snr_db(c,y[:v]);p,e=pesq_score(c,y[:v]);out.append(dict(segment_id=row['segment_id'],noise_category=row['noise_category'],method=name,snr_db=s,delta_snr_db=s-ns,pesq=p,delta_pesq=p-npesq,pesq_error=e))
 return out
def run(rows,names,label):
 results=[]; tasks=[(row,[(n,*C[n]) for n in names]) for row in rows.to_dict('records')]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,rs in enumerate(pool.map(work,tasks,chunksize=1),1):
   results.extend(rs)
   if i%12==0:print(label,i,len(rows),flush=True)
 d=pd.DataFrame(results);d.to_csv(OUT/f'{label}_per_file.csv',index=False);s=summarize_scores(d).sort_values('pesq_mean',ascending=False);s.to_csv(OUT/f'{label}_summary.csv',index=False);print(label,s[['method','pesq_mean','snr_mean_db','pesq_failed']].to_string(index=False),flush=True);return s
if __name__=='__main__':
 m=load_manifest(ROOT,'validation');assert len(m)==164
 m=m.sort_values(['noise_category','target_snr_db','segment_id']).reset_index(drop=True);rng=np.random.default_rng(20261116);idx=[rng.integers(lo,hi) for lo,hi in zip(np.linspace(0,164,49,dtype=int)[:-1],np.linspace(0,164,49,dtype=int)[1:])];screen=m.iloc[idx];screen[['segment_id']].to_csv(OUT/'screen_ids.csv',index=False)
 s=run(screen,list(C),'stage1');ps=[n for n in s.method if n.startswith(('scale_','q_','dof_','shrink_','adaptive_','blend_'))][:3];sm=[n for n in s.method if n.startswith('smooth_')][:2]
 for p in ps:
  for t in sm:
   b,k=C[p];_,j=C[t];add(p+'__'+t,blend=b,**(k|j))
 top=s.method.head(3).tolist();n2=list(dict.fromkeys(['quantile_base']+top+[n for n in C if '__' in n]));s2=run(screen,n2,'stage2');finalists=list(dict.fromkeys(['quantile_base']+s2.method.head(3).tolist()));
 (OUT/'candidates.json').write_text(json.dumps({'base':asdict(BASE),'all':C,'finalists':finalists,'seed':20261116,'screen_n':48,'remaining_n':116},indent=2));final=run(m,finalists,'full');d=pd.read_csv(OUT/'full_per_file.csv');rem=d[~d.segment_id.isin(screen.segment_id)];summarize_scores(rem).sort_values('pesq_mean',ascending=False).to_csv(OUT/'remaining_summary.csv',index=False);print('DONE',flush=True)
