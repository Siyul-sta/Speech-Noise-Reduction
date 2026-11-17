import numpy as np
from scipy import signal,ndimage
def filter_signal(x,y,mode):
    kw=dict(fs=16000,nperseg=512,noverlap=384)
    z=signal.stft(x,**kw)[2];a=signal.stft(y,**kw)[2];freq,t=z.shape;e=1e-17
    def stack(s):
        p=np.pad(s,((0,0),(4,4)),mode='edge')
        return np.stack([p[:,i:i+t] for i in (0,2,4,6,8)],axis=2)
    v=stack(z);u=stack(a)
    quiet=np.quantile(abs(z)**2,.2,axis=1)[:,None]+e
    weight=1/(1+ndimage.gaussian_filter(abs(a)**2,(1,2))/quiet)**2
    weight/=np.maximum(weight.sum(1)[:,None],e)
    rn=np.einsum('fti,ftj,ft->fij',v,v.conj(),weight,optimize=True)
    diag=np.maximum(rn.trace(axis1=1,axis2=2).real/5,e)
    inv=np.linalg.inv(rn+.12*diag[:,None,None]*np.eye(5))
    spec=np.empty_like(z)
    for i in range(0,t,4):
        near=slice(max(0,i-7),min(i+11,t));block=slice(i,min(i+4,t))
        cov=np.mean(u[:,near]*u[:,near,2,None].conj(),axis=1)
        steering=cov/np.maximum(cov[:,2,None].real,e)
        steering[:,2]=1
        w=(inv@steering[:,:,None])[:,:,0]
        w/=np.maximum(np.sum(w*steering.conj(),axis=1,keepdims=True).real,e)
        spec[:,block]=np.sum(w[:,None,:].conj()*v[:,block],axis=2)
    # Transfer the inferred multiframe noise attenuation to the pilot speech.
    q=ndimage.gaussian_filter(abs(spec)**2,(1,1))
    p=ndimage.gaussian_filter(abs(z)**2,(1,1))
    gain=np.clip(np.sqrt(q/(p+e)),.15,1.3)
    if mode=='v5mvdr_phase':out=a*gain*np.exp(.3j*np.angle(spec*z.conj()))
    elif mode=='v5mvdr_soft':out=a*np.sqrt(gain)
    else:out=a*gain
    return signal.istft(out,**kw)[1][:len(x)]
