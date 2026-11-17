import numpy as np
from scipy import signal,ndimage
def extend(x,y,mode):
    from dynamic import apply
    base=apply(x,y,'dyn_global')
    kw=dict(fs=16000,nperseg=512,noverlap=384)
    hz,_,a=signal.stft(base,**kw)
    b=signal.stft(y,**kw)[2]
    z=signal.stft(x,**kw)[2]
    # Recover high-band temporal envelopes from the nearest lower octave.
    # The first-pass estimate supplies the spectral tilt; the lower band supplies timing.
    mag=abs(b)
    lower=np.interp(hz*.5,hz,np.arange(len(hz))).astype(int)
    env=np.sqrt(ndimage.gaussian_filter(mag**2,(3,1)))
    shape=env[lower]/(np.quantile(env[lower],.85,axis=1)[:,None]+1e-12)
    high=1/(1+(2800/np.maximum(hz,1))**6)
    if mode=='extend_noise':
        carrier=z/(np.maximum(abs(z),1e-12))
    else:
        src=b[lower]
        carrier=(src/np.maximum(abs(src),1e-12))**2
    target=np.quantile(abs(a),.85,axis=1)[:,None]*np.minimum(shape,3)
    if mode=='extend_env':
        out=(1-.55*high[:,None])*a+.55*high[:,None]*target*a/np.maximum(abs(a),1e-12)
    else:out=a+high[:,None]*target*carrier*.45
    return signal.istft(out,**kw)[1][:len(x)]
