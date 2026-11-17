import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=1024,noverlap=768)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];e=1e-18
 band=np.exp(-.5*(np.log2(np.maximum(f,1)/510)/.7)**2)[:,None]
 p=ndimage.gaussian_filter(abs(a)**2,(1,2))
 n=np.quantile(ndimage.gaussian_filter(abs(z)**2,(1,2)),.15,axis=1)[:,None]+e
 rel=p/(p+3*n)
 if mode=='v5mid_input':a=z
 elif mode=='v5mid_refill':a=a+.65*rel**2*(z-a)
 elif mode=='v5mid_bass':a+=.8*rel/(1+(f[:,None]/380)**4)*(z-a)
 else:a+=.5*rel*band*(z-a)
 if mode=='v5mid_adapt':band=band*(.35+.65*ndimage.gaussian_filter(rel,(3,3)))
 gain=10**(10*band/20)
 if mode=='v5mid_balance':gain=1+(gain-1)*.5
 out=signal.istft(a*gain,**kw)[1][:len(x)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),e),.05,3.)
