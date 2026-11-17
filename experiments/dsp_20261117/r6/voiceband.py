import numpy as np
from scipy import signal,ndimage
def process(x,y,mode):
 kw=dict(fs=16000,nperseg=1024,noverlap=896)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];e=1e-18
 band=np.exp(-.5*(np.log2(np.maximum(f,1)/500)/.48)**2)[:,None]
 p=ndimage.gaussian_filter(abs(a)**2,(1.2,1.5));q=np.quantile(p,.85,axis=1)[:,None]
 floor=np.quantile(ndimage.gaussian_filter(abs(z)**2,(1,1)),.15,axis=1)[:,None]+e
 g=10**((band-1)*16/20)
 if mode=='v5vb_global':
  level=np.sum(p*band,axis=0)
  gate=np.clip((level/(np.quantile(level,.8)+e)+.001)**.10,.55,1.7)
 elif mode=='v5vb_phase':
  # Preserve the magnitude, but restore raw input phase in high-confidence speech bins.
  phase=np.angle(z*a.conj());a*=np.exp(.5j*phase*p/(p+floor))
  gate=1.
 elif mode=='v5vb_static':
  sn=np.quantile(p,.85,axis=1)[:,None]/floor
  gate=.4+.6*np.sqrt(sn/(sn+2))
 elif mode=='v5vb_peak':
  local=ndimage.gaussian_filter(p,(4,3))
  gate=np.clip(((p+e)/(local+e))**.09,.6,1.45)
 else:
  gate=np.clip((p/(q+e)+.001)**.09,.55,1.55)
 out=signal.istft(a*g*gate,**kw)[1][:len(x)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),e),.1,8)
