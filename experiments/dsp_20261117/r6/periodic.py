import numpy as np
from scipy import signal,ndimage
def cancel(x,y,strength):
    n=len(x)
    # Search autocorrelation in overlapping blocks after suppressing first-pass speech.
    residue=x-y
    blocks=np.lib.stride_tricks.sliding_window_view(residue,4096)[::1536]*np.hanning(4096)
    spec=np.fft.rfft(blocks,8192)
    power=abs(spec)**2
    # Emphasize spectral components shared across blocks.
    stable=np.quantile(power,.20,axis=0)/(np.quantile(power,.80,axis=0)+1e-15)
    corr=np.fft.irfft(power*stable,8192)[:,:1024]
    corr/=np.maximum(corr[:,:1],1e-15)
    curve=np.median(corr,axis=0)
    peaks=signal.find_peaks(curve)[0]
    peaks=peaks[(peaks>=12)&(peaks<=800)]
    if not len(peaks):return x
    k=peaks[np.argmax(curve[peaks]-.00008*peaks)]
    if curve[k]<.27:return x
    p=k+.5*(curve[k-1]-curve[k+1])/min(curve[k-1]-2*curve[k]+curve[k+1],-1e-12)
    idx=np.arange(n)
    # Median of the same phases across two-sided neighboring cycles rejects sparse speech.
    offsets=np.r_[-24:-3,4:25]
    vals=np.array([np.interp(idx+o*p,idx,x,left=np.nan,right=np.nan) for o in offsets])
    template=np.nanmedian(vals,axis=0)
    weight=np.clip((curve[k]-.27)/.55,0,1)
    return x-strength*weight*template
