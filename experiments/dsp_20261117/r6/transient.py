import numpy as np
from scipy import signal,ndimage
def process(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 hz,_,a=signal.stft(y,**kw)
 z=signal.stft(x,**kw)[2];eps=1e-18
 # Adaptive nonlinear auditory companding: limit extreme peaks relative to their local envelope.
 rms=np.sqrt(ndimage.gaussian_filter1d(y*y,160)+eps)
 if mode.startswith('v5clip'):
  k=float(mode.split('_')[1])
  out=k*rms*np.tanh(y/(k*rms))
  return out*np.clip(np.dot(x,out)/max(np.dot(out,out),eps),.7,1.8)
 if mode.startswith('v5detail'):
  # Recover input details in reliable bands at rapid speech changes.
  p=abs(z)**2; noise=np.quantile(ndimage.gaussian_filter(p,(1,1)),.2,axis=1)[:,None]
  env=ndimage.gaussian_filter(abs(a)**2,(2,1))
  rel=env/(env+noise+eps)
  slope=np.maximum(0,np.diff(np.log(env+eps),axis=1,prepend=np.log(env[:,:1]+eps)))
  flux=ndimage.gaussian_filter(slope,(2,2))
  mix=.42*rel**2*np.clip(flux/.4,0,1)
  if mode=='v5detail_bass':mix*=1/(1+(hz[:,None]/550)**4)
  out=a+mix*(z-a)
 else:
  # Compare short-term energy to its own slower, bandwise envelope.
  p=abs(a)**2;fast=ndimage.gaussian_filter(p,(1,1))
  slow=ndimage.gaussian_filter(p,(4,8))
  ratio=np.clip((fast+eps)/(slow+eps),.025,16)
  exponent=float(mode.split('_')[1]);out=a*ratio**exponent
 return signal.istft(out,**kw)[1][:len(x)]
