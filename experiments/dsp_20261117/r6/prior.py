import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];e=1e-18
 p=ndimage.gaussian_filter(abs(a)**2,(1,2))
 floor=np.quantile(ndimage.gaussian_filter(abs(z)**2,(1,2)),.18,axis=1)[:,None]+e
 env=ndimage.gaussian_filter(abs(z-a)**2,(2,2))
 n=np.minimum(env,3*floor)+.3*floor
 band=np.exp(-.5*(np.log2(np.maximum(f,1)/550)/.75)**2)[:,None]
 penalty=10**((1-band)*(1.7 if mode=='v5prior_hard' else 1.2))
 if mode=='v5prior_static':
  p=np.quantile(p,.80,axis=1)[:,None];n=floor
 elif mode=='v5prior_mix':
  p=np.sqrt(p*np.quantile(p,.75,axis=1)[:,None])
 g=np.sqrt((p+n)/(p+penalty*n+e))
 if mode=='v5prior_floor':g=np.maximum(g,.36)
 out=signal.istft(a*g,**kw)[1][:len(x)]
 return out*np.clip(np.dot(x,out)/max(np.dot(out,out),e),.4,4)
