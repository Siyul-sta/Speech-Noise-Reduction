import numpy as np
from scipy import signal,ndimage
def process(x,y,mode):
 size=640;hop=160
 f,_,a=signal.stft(y,fs=16000,nperseg=size,noverlap=size-hop)
 def tap(v):
  p=np.pad(v,(size//2,size//2+hop-1))
  frames=np.lib.stride_tricks.sliding_window_view(p,size)[::hop][:a.shape[1]]
  weights=signal.windows.dpss(size,2.5,4)
  return np.mean([abs(np.fft.rfft(frames*w,axis=1)).T**2 for w in weights],axis=0)
 p=tap(x);s=tap(y);e=1e-18
 # Calibrate the stationary noise floor from a multi-taper spectrum with lower variance.
 n=np.quantile(p,.18,axis=1)[:,None]
 mod=s/(s+.85*n+e)
 quality=np.quantile(s,.9,axis=1)/(n[:,0]+e)
 prior=np.exp(-.5*(np.log2(np.maximum(f,1)/500)/.40)**2)
 conf=quality/(quality+1.5)
 emphasis=prior*np.sqrt(conf)/max(np.max(prior*np.sqrt(conf)),e)
 g=10**((emphasis-1)*17/20)[:,None]
 if mode=='v5taper_select':out=a*g
 elif mode=='v5taper_gate':out=a*g*(.6+.4*np.sqrt(mod))
 else:
  z=signal.stft(x,fs=16000,nperseg=size,noverlap=size-hop)[2]
  out=(a+.4*mod**2*(z-a))*g
 y=signal.istft(out,fs=16000,nperseg=size,noverlap=size-hop)[1][:len(x)]
 return y*np.clip(np.dot(x,y)/max(np.dot(y,y),e),.1,8.)
