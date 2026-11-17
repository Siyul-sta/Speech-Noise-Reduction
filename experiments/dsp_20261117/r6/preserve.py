import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2]
 e=1e-18;p=abs(a)**2
 level=ndimage.gaussian_filter1d(np.sum(p,axis=0),3)
 weight=np.clip(np.sqrt(level/(np.quantile(level,.85)+e)),0,1)
 band=np.exp(-.5*(np.log2(np.maximum(f,1)/550)/.65)**2)[:,None]
 strength=13*(1-(.65 if mode=='v5protect_soft' else 1)*weight)
 if mode=='v5protect_const':strength=13.
 high=a*10**(strength*band/20)
 # Preserve the first pass's syllable energy; isolated noisy spectral peaks cannot set the level.
 after=np.sum(abs(high)**2,axis=0)
 norm=np.sqrt(level/np.maximum(ndimage.gaussian_filter1d(after,3),e))
 out=high*np.clip(norm,.1,1.5)
 if mode=='v5protect_raw':
  rel=p/(p+np.quantile(abs(z)**2,.25,axis=1)[:,None]+e)
  out+=.65*weight*rel**2*(z-a)
 if mode=='v5protect_low':
  low=1/(1+(f[:,None]/350)**4)
  out=out*(1-low)+low*(.6*a+.4*z)
 return signal.istft(out,**kw)[1][:len(x)]
