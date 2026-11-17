"""Steer the speech-band emphasis using high-confidence harmonic peaks."""
import numpy as np
from scipy import ndimage,signal

def shape(x,y,mode):
    opts=dict(fs=16000,nperseg=1024,noverlap=768)
    f,_,a=signal.stft(y,**opts)
    in_band=(f>=390)&(f<=625)
    mag=abs(a[in_band])
    peak=f[in_band][np.argmax(mag,axis=0)]
    ratio=np.max(mag,axis=0)/(np.mean(mag,axis=0)+1.e-14)
    conf=np.clip((ratio-1.8)/2,0,1)
    shift=np.log2(peak/500)*conf
    shift=ndimage.gaussian_filter1d(shift,2)
    strength=-.55 if 'away' in mode else .55
    center=500*2**(strength*shift)
    logf=np.log2(np.maximum(f[:,None],1)/center)
    response=np.exp(-.5*(logf/.32)**2)
    bass=20 if 'soft' in mode else 24
    gain=10**((response-1)*np.where(logf<=0,bass,16)/20)
    out=signal.istft(a*gain,**opts)[1][:len(x)]
    return out*np.clip(x@out/max(out@out,1.e-18),.1,8.)
