"""Local multibin Wiener estimator, using neighboring frequency bins as observations."""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def filterbank(x,first,sr,mode):
    kwargs=dict(fs=sr,nperseg=512,noverlap=384)
    z=signal.stft(x,**kwargs)[2]
    a=signal.stft(first,**kwargs)[2]
    f,t=z.shape
    def patches(spec):
        return np.lib.stride_tricks.sliding_window_view(np.pad(spec,((2,2),(0,0))),5,axis=0)
    obs=patches(z)
    speech=patches(a)
    mag=abs(z)**2
    act=gaussian_filter1d(np.log1p(mag/(np.quantile(mag,.20,axis=1)[:,None]+1e-18)),3,axis=0)
    weights=np.exp(-2*(act+act.mean(0)))+1e-7
    weights/=weights.sum(1)[:,None]
    noise=np.einsum('fti,ftj,ft->fij',obs,obs.conj(),weights,optimize=True)
    ridge=np.maximum(noise.trace(axis1=1,axis2=2).real/5,1e-18)
    noise+=np.eye(5)*ridge[:,None,None]*.08
    out=np.empty_like(z)
    for start in range(0,t,6):
        region=slice(max(0,start-8),min(t,start+14))
        target=slice(start,min(t,start+6))
        v=speech[:,region]
        cov=v.swapaxes(1,2)@v.conj()/v.shape[1]
        w=np.linalg.solve(cov+noise,cov[:,:,2,None])[...,0]
        out[:,target]=np.einsum('fi,fti->ft',w.conj(),obs[:,target])
    if mode=="space_phase":
        out=abs(a)*out/np.maximum(abs(out),1e-15)
    elif mode=="space_gate":
        out=a*np.sqrt(np.clip(abs(out)/np.maximum(abs(z),1e-15),.02,1))
    return signal.istft(.55*a+.45*out,**kwargs)[1][:len(x)]
