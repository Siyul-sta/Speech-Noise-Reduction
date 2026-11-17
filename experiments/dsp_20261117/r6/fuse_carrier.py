"""Fuse carrier estimates from different time-frequency resolutions."""
from coherence import repair
from pockets import restore
from equal_loudness import adapt
import numpy as np
from scipy import signal,ndimage

def fuse(x,pilot,mode):
    short = repair(x,pilot,'v5carrier_internal_1024_19_0.7')
    long = repair(x,pilot,'v5carrier_internal_4096_19_0.7')
    kw=dict(fs=16000,nperseg=1024,noverlap=768)
    f,_,a=signal.stft(short,**kw)
    b=signal.stft(long,**kw)[2]
    energy=ndimage.gaussian_filter(abs(a)**2,(2,1))
    flux=np.abs(np.diff(np.log(energy+1.e-16),axis=1,prepend=np.log(energy[:,:1]+1.e-16)))
    weight = .2+.65*np.clip(flux/1.2,0,1)
    if 'band' in mode:
        weight = f[:,None]**2/(f[:,None]**2+650**2)
    y=signal.istft(b+(a-b)*weight,**kw)[1][:len(x)]
    if 'unshaped' in mode:
        return y
    if "center" in mode:
        from harmonic_center import shape
        return shape(x,y,mode)
    if 'smooth' in mode:
        from nonlocal_filter import smooth
        y = smooth(x,y,'v5nlocal_avg')
    if 'auto' in mode:
        return adapt(x,y,'v5carrier_adapt_0.5_hard_.7')
    return restore(x,y,'v5r6_slope_20_16' if 'soft' in mode else 'v5r6_slope_24_16')
