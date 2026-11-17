import sys,csv
from pathlib import Path
import numpy as np,soundfile as sf
from scipy import signal
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src import final_dsp as d
import argparse
p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--manifest',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(args.root)
records=[]
for row in list(csv.DictReader(open(args.manifest))):
 c,sr=sf.read(root/row['clean_path']);x,_=sf.read(root/row['noisy_path'])
 scale=x.dot(c)/c.dot(c)
 old=d.core.enhance_single
 def inject(actual,sr,cfg,**kw):
  original=d.core.NOISE_ESTIMATOR
  def audit(p,i,f,s,sr,n,h):
   b=d.noise_estimate(p,i,f,'local_15_0.5')
   q=d.core.track_noise_spp(p,i)
   z=signal.stft(scale*c,nperseg=n,noverlap=n-h)[2]
   noise=abs(signal.stft(s-scale*c,nperseg=n,noverlap=n-h)[2])**2
   for region,mask in [('speech',abs(z)**2>noise),('noise',abs(z)**2<=noise)]:
    for method,v in [('baseline',b),('spp',q)]:
     ratio=10*np.log10(np.maximum(v,1e-18)/np.maximum(noise,1e-18))
     records.append({'id':row['segment_id'],'region':region,'method':method,'median_bias_db':np.median(ratio[mask]),'median_absolute_error_db':np.median(abs(ratio[mask]))})
   return b
  d.core.NOISE_ESTIMATOR=audit
  try:return old(actual,sr,cfg,**kw)
  finally:d.core.NOISE_ESTIMATOR=original
 d.core.enhance_single=inject
 try:d.enhance(x)
 finally:d.core.enhance_single=old
with open(args.output,'w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
