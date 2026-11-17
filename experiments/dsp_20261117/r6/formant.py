import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];e=1e-15
 p=ndimage.gaussian_filter(abs(a)**2,(2,2))
 n=np.quantile(ndimage.gaussian_filter(abs(z)**2,(2,2)),.12,axis=1)[:,None]+e
 band=(f>=320)&(f<=900)
 weight=np.sqrt(p[band])*(f[band,None]/500)**.7
 if mode=='v5form_reliable': weight*=p[band]/(p[band]+n[band])
 center=np.sum(weight*f[band,None],axis=0)/np.maximum(weight.sum(0),e)
 center=ndimage.gaussian_filter1d(np.clip(center,390,740),4)
 if mode=='v5form_fixed':center=np.full_like(center,550)
 width=.85 if mode=='v5form_wide' else .65
 response=np.exp(-.5*(np.log2(np.maximum(f[:,None],1)/center)/width)**2)
 gain=10**((9 if mode=='v5form_soft' else 13)*response/20)
 if mode=='v5form_restore':
  a+=.35*(p/(p+2*n))**2*(z-a)
 out=signal.istft(a*gain,**kw)[1][:len(x)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),e),.1,2.)
