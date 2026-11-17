from pathlib import Path
from loop import periods
from scipy import signal
from runner import TA,ROOT
import hashlib,numpy as np,pandas as pd
def main():
 d=pd.read_csv(ROOT/'train_split.csv').query("stage=='dev'")
 rows=[]
 for _,r in d.iterrows():
  c,x=TA['read_pair'](ROOT.parent/'noise_reduction_challenge',r); n=x-c
  key=hashlib.sha256(x.tobytes()).hexdigest()
  y=np.load('/tmp/dsp_r4_cache/'+key+'.npy');p,score=periods(x,y)
  lags=np.arange(96,len(x)*7//10);acf=signal.correlate(n,n,method='fft')[len(x)-1:];energy=np.r_[0,np.cumsum(n*n)]
  ac=acf[lags]/np.sqrt((energy[-1]-energy[lags])*energy[-1-lags]+1e-18)
  k=ac.argmax()
  rows.append([r.segment_id,r.noise_category,p,score,lags[k],ac[k],ac[p-96] if p else 0])
 out=pd.DataFrame(rows,columns=['id','noise','detected_lag','score','oracle_lag','oracle_corr','corr_at_detected'])
 out.to_csv(ROOT/'results/diagnostic_lags.csv',index=False)
 print(out.query('oracle_corr>.94').to_string(index=False))
if __name__=='__main__':main()
