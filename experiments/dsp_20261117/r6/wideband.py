"""Use a coherent noise estimate to repair a broadband speech pilot."""
import numpy as np
from scipy import ndimage, signal
from coherence import repair

def refine(x,pilot,mode):
    short=repair(x,pilot,'v5carrier_internal_1024_19_0.7')
    kw=dict(fs=16000,nperseg=1024,noverlap=768)
    f,_,a=signal.stft(pilot,**kw)
    b=signal.stft(short,**kw)[2]
    correction=b-a
    if 'low' in mode:
        weight=1/(1+(f[:,None]/330)**4)
    elif 'phase' in mode:
        weight=np.clip((b*a.conj()).real/(abs(a)*abs(b)+1.e-18),0,1)
        # Update phase towards the repaired branch while retaining the pilot magnitude.
        b=abs(a)*np.exp(1j*np.angle(b))
        correction=b-a
    else:
        weight=np.ones_like(f[:,None])*.5
    y=signal.istft(a+weight*correction,**kw)[1][:len(x)]
    if 'level' in mode:
        r=signal.sosfiltfilt(signal.butter(3,700,fs=16000,output='sos'),x-y)
        low=signal.sosfiltfilt(signal.butter(3,700,fs=16000,output='sos'),y)
        cross=ndimage.uniform_filter1d(r*low,4001)
        power=ndimage.uniform_filter1d(low*low,4001)
        y += low*np.clip(cross/(power+1.e-18),-.18,.18)
    return y
