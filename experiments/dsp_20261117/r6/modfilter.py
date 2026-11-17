import numpy as np
from scipy import signal,ndimage
def run(x,y,mode):
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 hz,_,a=signal.stft(y,**kw)
 p=abs(a)**2
 bark=13*np.arctan(.00076*hz)+3.5*np.arctan((hz/7500)**2)
 centers=np.arange(0,bark[-1]+1)
 bank=np.maximum(0,1-abs(centers[:,None]-bark)/1.4)
 env=np.log(bank@p+1e-15)
 short=ndimage.gaussian_filter1d(env,1.5,axis=1)
 mid=ndimage.gaussian_filter1d(env,7,axis=1)
 long=ndimage.gaussian_filter1d(env,24,axis=1)
 if mode=='v5rasta':diff=short-long
 elif mode=='v5syllable':diff=mid-long
 elif mode=='v5antimod':diff=long-mid
 elif mode=='v5attack':diff=short-mid
 elif mode=='v5sustain':diff=mid-short
 gain=np.exp(np.clip((bank.T@diff)/bank.sum(0)[:,None]*.12,-.6,.45))
 out=signal.istft(a*gain,**kw)[1][:len(x)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),1e-15),.7,1.6)
