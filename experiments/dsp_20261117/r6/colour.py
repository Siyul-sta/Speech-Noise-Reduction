import numpy as np
from scipy import signal,ndimage
def color(x,y,mode):
 kw=dict(fs=16000,nperseg=1024,noverlap=768)
 hz,_,a=signal.stft(y,**kw)
 _,center,db=mode.split('_');c=float(center);db=float(db)
 weight=np.exp(-.5*(np.log2(np.maximum(hz,1)/c)/.65)**2)
 out=signal.istft(a*10**(db*weight[:,None]/20),**kw)[1][:len(y)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),1e-18),.6,1.8)
