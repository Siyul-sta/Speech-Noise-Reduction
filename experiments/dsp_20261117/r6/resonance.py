import numpy as np
from scipy import signal
def run(x,y,mode):
 _,center,width,db=mode.split('_'); center,width,db=map(float,(center,width,db))
 f,_,a=signal.stft(y,fs=16000,nperseg=1024,noverlap=768)
 response=np.exp(-.5*(np.log2(np.maximum(f,1)/center)/width)**2)
 gain=10**((response-1)*db/20)
 w=signal.istft(a*gain[:,None],fs=16000,nperseg=1024,noverlap=768)[1][:len(x)]
 scale=np.clip(np.dot(x,w)/max(np.dot(w,w),1e-18),.1,8.)
 return w*scale
