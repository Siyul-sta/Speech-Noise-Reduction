import numpy as np
from scipy import signal,ndimage
EPS=1e-18
def process(x,y,mode,c=None):
 if mode.startswith("hi_floor_"):
  from hi_floor import restore
  base=process(x,y,"v5fuse_flux")
  return restore(x,base,mode)
 if mode=='base':return y
 if mode.startswith("wide6"):
  from wideband import refine
  return refine(x,process(x,y,"balanced_mix"),mode)
 if mode.startswith("onset_guard"):
  from onset_guard import protect
  return protect(x,process(x,y,"v5fuse_soft"),mode)
 if mode.startswith("v5"):
  from v5 import improve
  return improve(x,y,mode)
 if mode.startswith("psola"):
  from voiced import refine
  return refine(x,y,mode)
 if mode.startswith("white"):
  from whiten import process as whiten
  return whiten(x,y,mode)
 if mode.startswith("shape"):
  from aid import shape
  return shape(x,y,mode)
 if mode.startswith("dyn"):
  from dynamic import apply
  return apply(x,y,mode)
 if mode.startswith("combine_"):
  from loop import recover
  from dynamic import apply
  v=recover(x,y,"loop_safe")
  return apply(x,v,"dyn_both" if mode=="combine_both" else "dyn_global")
 if mode.startswith("loop"):
  from loop import recover
  return recover(x,y,mode)
 if mode.startswith("project"):
  from loop import project
  return project(x,y,mode)
 if mode.startswith("multi_"):
  from multiband import apply
  return apply(x,y,mode)
 if mode.startswith("extend"):
  from extension import extend
  return extend(x,y,mode)
 if mode.startswith("track_"):
  from ridge_track import filter_tracks
  return filter_tracks(x,y,mode)
 if mode.startswith("inverse"):
  from inverse import solve
  return solve(x,y,mode)
 if mode.startswith("balanced"):
  from previous import Config,enhance
  from pathlib import Path
  from dynamic import apply
  import json
  cfg=Config(**json.loads(Path(__file__).with_name("selected_config.json").read_text())["config"])
  safe=enhance(x,16000,cfg,estimator="quantile",blend=1.)
  if mode=="balanced_mix":return .5*safe+.5*apply(x,y,"dyn_global")
  if mode=="balanced_dynamic":return apply(x,safe,"dyn_global")
  return process(x,safe,"shelf_4000_-18")
 if mode.startswith("period"):
  from periodic import cancel
  from previous import enhance,Config
  from dataclasses import replace
  from pathlib import Path
  import json
  cfg=replace(Config(),**json.loads(Path(__file__).with_name("pesq_config.json").read_text())["config"])
  pre=cancel(x,y,.75 if mode=="period" else 1)
  if np.array_equal(pre,x):return y
  return enhance(pre,16000,cfg,estimator="quantile",blend=1.)
 kw=dict(fs=16000,nperseg=1024,noverlap=896)
 st=lambda data:signal.stft(data,**kw)[2]
 inv=lambda spec:signal.istft(spec,**kw)[1][:len(x)]
 z,a=st(x),st(y)
 if mode.startswith('oracle'):
  s=st(c)
  if mode.startswith('oracle_mix'):
   true_s=abs(s)**2;true_n=abs(z-s)**2
   speech=ndimage.gaussian_filter(abs(a)**2,(1.2,1.2))
   noise=ndimage.gaussian_filter(abs(z-a)**2,(1.2,1.2))
   if mode in ('oracle_mix_s','oracle_mix_both'):speech=ndimage.gaussian_filter(true_s,(1.2,1.2))
   if mode in ('oracle_mix_n','oracle_mix_both'):noise=ndimage.gaussian_filter(true_n,(1.2,1.2))
   return inv(z*speech/(speech+noise+EPS))
  axis=1 if mode.endswith('freq') else 0
  g=np.clip(np.sum((s*a.conj()).real,axis=axis,keepdims=True)/(np.sum(abs(a)**2,axis=axis,keepdims=True)+EPS),0,6)
  return inv(a*g)
 from methods import refine
 return inv(refine(z,a,mode))
