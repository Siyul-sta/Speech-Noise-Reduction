"""Dual-path speech recovery with a low-frequency shelf and level matching."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from previous import Config,enhance

def recover(x,pilot,refined,mode):
    _, _, weight, low, high, _ = mode.split('_')
    weight,low,high = map(float,(weight,low,high))
    cfg = Config(**json.loads(Path(__file__).with_name('selected_config.json').read_text())['config'])
    safe = enhance(x,16000,cfg,estimator='quantile',blend=1)
    y = weight*safe + (1-weight)*(.7*pilot+.3*refined)
    opts=dict(fs=16000,nperseg=1024,noverlap=768)
    f,_,a=signal.stft(y,**opts)
    db = low/(1+(f/320)**5) + high*(f/2000)**3/(1+(f/2000)**3)
    if mode.startswith('v5carrier_ring'):
        # Apply modest local gain around the speech band, preserving bass elsewhere.
        db = low*np.exp(-.5*(np.log2(np.maximum(f,1)/500)/.26)**2) + high*(f/2200)**4/(1+(f/2200)**4)
    out = signal.istft(a*10**(db[:,None]/20),**opts)[1][:len(x)]
    return out*np.clip(x@out/max(out@out,1.e-18),.5,2)
