import numpy as np
from scipy import signal,ndimage
def apply(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 hz,_,a=signal.stft(y,**kw)
 hzweight=(1/(1+(2600/np.maximum(hz,1))**4))
 out=a*(10**(-19*hzweight/20))[:,None]
 power=abs(out)**2
 level=ndimage.gaussian_filter1d(np.sum(power,axis=0),2)
 ref=np.quantile(level,.8)+1e-18
 bands=ndimage.gaussian_filter(power,(9,2))
 norm=np.quantile(bands,.8,axis=1)[:,None]+1e-18
 rel=bands/norm
 if mode.startswith("dynx"):
  exponent,cut,db=map(float,mode.split("_")[1:])
  high=1/(1+(cut/np.maximum(hz,1))**4)
  out=a*10**(-db*high[:,None]/20)
  g=np.maximum(level/ref,.0001)[None,:]**exponent
 elif mode=='dyn_global':g=(np.maximum(level/ref,.001)**.10)[None,:]
 elif mode=='dyn_compress':g=(np.maximum(level/ref,.001)**-.08)[None,:]
 elif mode=='dyn_band':g=np.maximum(rel,.001)**.10
 elif mode=='dyn_lift':g=np.maximum(rel,.001)**-.07
 else:g=np.sqrt(np.maximum(rel,.001)**.15*np.maximum(level/ref,.001)**.1)
 out=out*np.clip(g,.2 if mode.startswith("dynx") else .5,2. if mode.startswith("dynx") else 1.7)
 w=signal.istft(out,**kw)[1][:len(x)]
 alpha=np.dot(x,w)/max(np.dot(w,w),1e-18)
 return np.clip(alpha,.65,1.7)*w
