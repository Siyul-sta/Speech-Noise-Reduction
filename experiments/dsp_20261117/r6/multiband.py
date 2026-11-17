import numpy as np
from scipy import signal,ndimage
def apply(x,y,mode):
 hz,_,a=signal.stft(y,fs=16000,nperseg=1024,noverlap=896)
 z=signal.stft(x,fs=16000,nperseg=1024,noverlap=896)[2]
 # Complementary crossover: preserve low-frequency waveform when first-pass speech dominates.
 p=ndimage.gaussian_filter(abs(z)**2,(1,2));noise=np.quantile(p,.18,axis=1)[:,None]+1e-18
 s=ndimage.gaussian_filter(abs(a)**2,(1,2));w=s/(s+noise)
 level=ndimage.gaussian_filter1d(np.sum(s,axis=0),2); ref=np.quantile(level,.8)+1e-18
 gate=np.clip((level/ref+.001)**.10,.5,1.6)
 k,db,b=map(float,mode.split('_')[1:])
 shelf=1/(1+(k/np.maximum(hz,1))**4)
 high=a*10**(-db*shelf[:,None]/20)*gate
 bass=1/(1+(hz/max(b,1))**4)
 out=high*(1-bass[:,None])+(a+.75*w*(z-a))*bass[:,None]
 y=signal.istft(out,fs=16000,nperseg=1024,noverlap=896)[1][:len(x)]
 return y*np.clip(np.dot(x,y)/max(np.dot(y,y),1e-15),.65,1.7)
