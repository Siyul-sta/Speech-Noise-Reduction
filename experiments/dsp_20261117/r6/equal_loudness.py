"""Per-file spectral tilt correction using reliable frames of the noisy-only pilot."""
import numpy as np
from scipy import signal,ndimage

def adapt(x,y,mode):
    opts=dict(fs=16000,nperseg=1024,noverlap=768)
    f,_,a=signal.stft(y,**opts)
    power=abs(a)**2
    mid=(f>=390)&(f<=650)
    active=np.mean(power[mid],axis=0)>np.quantile(np.mean(power[mid],axis=0),.5)
    if not np.any(active):
        return np.zeros_like(x)
    profile=np.mean(power[:,active],axis=1)
    ref=np.mean(profile[mid])+1.e-18
    bass=10*np.log10((np.mean(profile[(f>=120)&(f<=300)])+1.e-18)/ref)
    high=10*np.log10((np.mean(profile[(f>=800)&(f<=2200)])+1.e-18)/ref)
    strength=float(mode.split('_')[2])
    low=np.clip(24+strength*bass,16,34)
    upper=np.clip(16+strength*(high+10),8,24)
    if mode.split('_')[3]=='soft':
        low-=4
    octaves=np.log2(np.maximum(f,1)/500)
    gain=10**((np.exp(-.5*(octaves/.32)**2)-1)*np.where(f<=500,low,upper)/20)
    out=signal.istft(a*gain[:,None],**opts)[1][:len(x)]
    return out*np.clip(x@out/max(out@out,1.e-18),.1,8.)
