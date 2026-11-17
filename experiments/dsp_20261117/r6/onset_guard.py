"""Reference-free speech-onset protection on a multi-resolution estimate."""
import numpy as np
from scipy import signal,ndimage

def protect(x, pilot, mode):
    soft=signal.stft(pilot,fs=16000,nperseg=1024,noverlap=768)[2]
    f,t,z=signal.stft(x,fs=16000,nperseg=1024,noverlap=768)
    residual=z-soft
    speech=ndimage.gaussian_filter(abs(soft)**2,(1,1))
    noise=ndimage.gaussian_filter(abs(residual)**2,(1,1))
    snr=speech/(speech+noise+1.e-18)
    # Robust across-frequency speech evidence identifies abrupt voiced/onset growth.
    evidence=np.median(snr[(f>=250)&(f<=5000)],axis=0)
    attack=int(mode.split('_')[-1])
    rise=np.maximum(np.diff(evidence,prepend=evidence[0]),0)
    onset=ndimage.gaussian_filter1d(rise,1)
    activity=ndimage.gaussian_filter1d(evidence,2)
    floor=np.quantile(activity,.2)
    gate=np.clip((activity-floor)/.20,0,1)
    gate=np.maximum(gate,np.clip(onset*attack,0,.65))
    # Restore only high-confidence bins, capped to the complex mixture.
    conf=np.clip((snr-.40)/.5,0,1)
    amount=gate[None,:]*conf
    out=soft+amount*(z-soft)
    return signal.istft(out,fs=16000,nperseg=1024,noverlap=768)[1][:len(x)]
