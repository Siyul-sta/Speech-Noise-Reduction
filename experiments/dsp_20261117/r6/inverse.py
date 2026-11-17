"""Multiresolution weighted least-squares separation with a common time-domain estimate."""
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import cg,LinearOperator
from dynamic import apply

def solve(x,y,mode):
    n=len(x);transforms=[]
    for size,hop in ((384,128),(1024,256),(2048,512)):
        w=np.sqrt(np.maximum(np.hanning(size+1)[:-1],0))
        starts=np.arange(-size+hop,n,hop)
        def analysis(v,w=w,starts=starts,size=size,hop=hop):
            padded=np.pad(v,(size-hop,size))
            return np.fft.rfft(np.lib.stride_tricks.sliding_window_view(padded,size)[::hop][:len(starts)]*w,axis=1,norm='ortho')
        def synth(v,w=w,starts=starts,size=size,hop=hop):
            frames=np.fft.irfft(v,n=size,axis=1,norm='ortho')*w
            out=np.zeros(n+2*size)
            for j in range(size//hop):
                block=frames[j::size//hop].ravel()
                out[j*hop:j*hop+len(block)]+=block
            return out[size-hop:size-hop+n]
        z=analysis(x);a=analysis(y)
        p=abs(z)**2
        floor=np.quantile(ndimage.gaussian_filter(p,(2,1)),.15,axis=0)+1e-15
        sp=ndimage.gaussian_filter(abs(a)**2,(1,1))+.015*floor
        ns=ndimage.gaussian_filter(abs(z-a)**2,(2,1))+.15*floor
        # Normalize each bin's total precision, then blend overlapping spectral constraints.
        mask=sp/(sp+ns)
        precision=np.clip((floor/(sp+ns))**.4,.2,3)
        transforms.append((analysis,synth,precision,mask))
    alpha=.6
    b=alpha*y
    for analysis,synth,prec,mask in transforms:
        b=b+synth(analysis(x)*prec*mask)
    def matvec(v):
        out=alpha*v
        for analysis,synth,prec,_ in transforms:out=out+synth(analysis(v)*prec)
        return out
    sol,info=cg(LinearOperator((n,n),dtype=float,matvec=matvec),b,x0=y,maxiter=9,rtol=.001)
    if mode=='inverse_phase':
        analysis,synth,_,_=transforms[1]
        a=analysis(y);z=analysis(sol)
        sol=synth(abs(a)*z/np.maximum(abs(z),1e-15))*.5
    f=.25 if mode=='inverse_light' else .6
    return apply(x,(1-f)*y+f*sol,'dyn_global')
