import json
from pathlib import Path
from previous import Config,enhance
from formant import run as shape
from dynamic import apply
import numpy as np
from scipy import signal,ndimage
def process(x,y,mode):
 cfg=Config(**json.loads(Path(__file__).with_name('selected_config.json').read_text())['config'])
 if mode.startswith('v5front'):
  filtered=shape(x,x,'v5form_fixed')
  out=enhance(filtered,16000,cfg,estimator='quantile',blend=1)
  return .65*out+.35*shape(x,y,'v5form_fixed')
 safe=enhance(x,16000,cfg,estimator='quantile',blend=1)
 if mode.startswith('v5safe_blend'):
  from colour import color
  db=mode.split('_')[-1]
  return color(x,.5*(y+safe),"v5colour_500_"+db)
 if mode.startswith('v5safe_eq'):
  from colour import color
  shelf=signal.sosfiltfilt(signal.butter(2,3200,fs=16000,output='sos'),safe)
  shelf=.18*safe+.82*shelf
  return color(x,shelf,"v5colour_500_"+mode.split("_")[-1])
 if mode=='v5safe_only':return shape(x,safe,'v5form_fixed')
 best=shape(x,y,'v5form_fixed')
 if mode=='v5safe_mix':return .45*safe+.55*best
 if mode=='v5safe_light':return .65*safe+.35*best
 # Recover low-frequency energy from the higher-fidelity path. Preserve mid emphasis.
 f,_,a=signal.stft(best,fs=16000,nperseg=768,noverlap=640)
 z=signal.stft(safe,fs=16000,nperseg=768,noverlap=640)[2]
 k=180 if mode=='v5safe_sub' else 300
 w=1/(1+(f/k)**6)
 return signal.istft(a+(z-a)*w[:,None],fs=16000,nperseg=768,noverlap=640)[1][:len(x)]
