import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=1024,noverlap=768)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];e=1e-18
 p=ndimage.gaussian_filter(abs(z)**2,(1,1))
 lo,hi=np.quantile(p,[.15,.85],axis=1)
 r=np.quantile(abs(a)**2,.85,axis=1)/(lo+e)
 u=np.log1p(r)*np.log1p(hi/(lo+e))
 u=ndimage.gaussian_filter1d(u,3)
 centers=np.arange(360,841,20)
 grid=np.log2(np.maximum(f,1)[None,:]/centers[:,None])
 atoms=np.exp(-.5*(grid/.36)**2)
 quality=(atoms@u)/atoms.sum(1)
 prior=np.exp(-.5*(np.log2(centers/500)/.5)**2)
 prob=np.maximum(quality,1e-5)**3*prior
 center=np.sum(centers*prob)/max(np.sum(prob),e)
 g=np.exp(-.5*(np.log2(np.maximum(f,1)/center)/.34)**2)
 if mode=='v5clear_multi':
  g=(prob@atoms)/max(prob.sum(),e);g/=max(g.max(),e)
 if mode=='v5clear_local':
  baseline=np.exp(-.5*(np.log2(np.maximum(f,1)/510)/.48)**2)
  g=baseline*np.clip((u+1)/(np.median(u[(f>=350)&(f<850)])+1),.55,1.5)
  g=ndimage.gaussian_filter1d(g,1);g/=max(g.max(),e)
 db=18
 out=signal.istft(a*10**((g[:,None]-1)*db/20),**kw)[1][:len(x)]
 return out*np.clip(np.dot(x,out)/max(np.dot(out,out),e),.1,8)
