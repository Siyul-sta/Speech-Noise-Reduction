"""Deterministic within-utterance non-local means in auditory bands."""
import numpy as np
from scipy import signal,ndimage
from coherence import repair
from pockets import restore

def enhance(x,pilot,mode):
    restored=repair(x,pilot,'v5carrier_internal_1024_19_0.7')
    y=smooth(x,restored,mode)
    return restore(x,y,'v5r6_slope_20_16' if 'soft' in mode else 'v5r6_slope_24_16')

def smooth(x,restored,mode):
    kw=dict(fs=16000,nperseg=512,noverlap=384)
    f,_,a=signal.stft(restored,**kw)
    z=signal.stft(x,**kw)[2]
    edges=np.geomspace(75,8000,21)
    bands=np.clip(np.searchsorted(edges,f),0,len(edges)-1)
    ranges=np.unique(bands)
    energy=abs(a)**2
    features=np.stack([np.log(np.mean(energy[bands==b],axis=0)+1.e-15) for b in ranges])
    # Fixed distances compare neighbouring frames as well as instantaneous spectra.
    feat=np.vstack((features,ndimage.gaussian_filter1d(features,2,axis=1)))
    diff=feat.T[:,None,:]-feat.T[None,:,:]
    distance=np.mean(np.minimum(abs(diff),5)**2,axis=2)
    t=np.arange(a.shape[1])
    weights=np.exp(-distance/1.2)*(abs(t[:,None]-t[None,:])>=6)
    weights/=np.maximum(np.sum(weights,axis=1,keepdims=True),1.e-12)
    local=ndimage.gaussian_filter(energy,(1.5,1))
    pooled=local@weights.T
    floor=np.quantile(ndimage.gaussian_filter(abs(z)**2,(1,1)),.18,axis=1)[:,None]
    if 'noise' in mode:
        # Repeated speech envelopes aid attenuation where the local pilot is weak.
        ratio=(pooled+local)/(pooled+local+floor+1.e-15)
        gain=.7+.3*ratio
    else:
        gain=np.clip(((pooled+1.e-15)/(local+1.e-15))**.12,.75,1.26)
    y=signal.istft(a*gain,**kw)[1][:len(x)]
    return y
