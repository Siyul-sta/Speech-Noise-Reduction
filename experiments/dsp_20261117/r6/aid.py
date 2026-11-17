import numpy as np
from scipy import signal,ndimage
def shape(x,y,mode):
 kw=dict(fs=16000,nperseg=1024,noverlap=896)
 hz,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2]; eps=1e-18
 # Estimate how much recoverable speech energy remains in each band.
 p=ndimage.gaussian_filter(abs(z)**2,(1,2))
 lo,hi=np.quantile(p,[.12,.90],axis=1)
 rho=np.quantile(abs(a)**2,.85,axis=1)/(lo+eps)
 reliable=rho/(rho+4)
 slope=(1/(1+(3000/np.maximum(hz,1))**4))
 if mode=='shape_adapt':slope*=1-.7*reliable
 if mode=='shape_more':slope=(1/(1+(2400/np.maximum(hz,1))**4))
 g=10**(-30*slope/20)
 # Estimate speech lost below 900 Hz by regression in high-SNR frames.
 # Re-estimate it only in bands where the residual is coherent with the enhanced signal.
 en=ndimage.gaussian_filter(abs(a)**2,(2,2))
 speech=en/(en+4*lo[:,None]+eps)
 coherence=ndimage.gaussian_filter((z*a.conj()).real,(2,3))/(en+eps)
 bass=(1/(1+(hz/750)**4))[:,None]
 anchored=a*(1+bass*speech*np.clip(coherence-1,0,.9))
 if mode=='shape_dry':
  anchored+=.25*bass*speech*(z-anchored)
 out=anchored*g[:,None]
 return signal.istft(out,**kw)[1][:len(x)]
