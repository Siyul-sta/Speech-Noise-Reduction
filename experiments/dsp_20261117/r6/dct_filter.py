import numpy as np
from scipy import fft,ndimage
from numpy.lib.stride_tricks import sliding_window_view

def estimate(x,first,n,kind):
    hop=n//8
    win=np.sin(np.pi*(np.arange(n)+.5)/n)
    def frames(a):
        padded=np.pad(a,(n,n))
        return fft.dct(sliding_window_view(padded,n)[::hop]*win,norm='ortho',axis=1)
    z=frames(x)
    a=frames(first)
    power=z*z
    floor=np.quantile(ndimage.gaussian_filter(power,(1.5,1)),.15,axis=0)+1e-15
    s=ndimage.gaussian_filter(a*a,(1.2,1))
    post=np.sqrt(s/(s+floor))
    if kind=='mask':
        gain=np.minimum(post,1)
        y=z*gain
    else:
        # Preserve sign-consistent first-pass components; suppress residuals isolated
        # in either the cosine or Fourier domain.
        ratio=np.clip(abs(a)/(abs(z)+1e-12),0,1)
        support=ndimage.gaussian_filter(np.log(.01+ratio),(2,2))
        y=a*np.exp(.5*support)
    out=np.zeros(len(x)+2*n)
    for offset,frame in enumerate(fft.idct(y,norm='ortho',axis=1)):
        i=offset*hop
        out[i:i+n]+=win*frame
    return out[n:n+len(x)]*(2*hop/n)
