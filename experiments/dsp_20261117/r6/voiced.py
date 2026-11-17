import numpy as np
from scipy import signal,ndimage
def refine(x,y,mode):
 n=len(x);hop=128;L=640
 frames=np.lib.stride_tricks.sliding_window_view(np.pad(y,(L//2,L//2)),L)[::hop]
 frames=frames-frames.mean(1,keepdims=True)
 ac=np.fft.irfft(abs(np.fft.rfft(frames,2*L))**2)[:,:265]
 energy=np.c_[np.zeros(len(frames)),np.cumsum(frames**2,axis=1)]
 delays=np.arange(40,246)
 norm=np.sqrt(np.maximum(energy[:,L-delays]*(energy[:,L,None]-energy[:,delays]),1e-18))
 r=ac[:,delays]/norm
 # Viterbi over log pitch, allowing jumps at low voicing strength.
 score=r[0].copy(); parent=[]
 cost=np.minimum(1.6*np.abs(np.log(delays[:,None]/delays[None,:])),.45)
 for frame in r[1:]:
  v=score[:,None]-cost; path=v.argmax(axis=0);parent.append(path)
  score=v[path,np.arange(len(delays))]+frame-.00018*delays
 path=[score.argmax()]
 for pp in reversed(parent):path.append(pp[path[-1]])
 j=np.array(path[::-1]);k=delays[j]
 c=r[np.arange(len(r)),j]
 period=np.interp(np.arange(n),np.arange(len(k))*hop,k)
 conf=np.interp(np.arange(n),np.arange(len(c))*hop,np.clip((c-.45)/.42,0,1))
 # Phase is integrated over time to preserve cycles across glides.
 phase=np.cumsum(1/period); idx=np.arange(n)
 if mode=="track_only":return period,conf
 def avg(s):
  weighted=s.copy();total=np.ones(n)
  for shift in [-3,-2,-1,1,2,3]:
   pos=np.interp(phase+shift,phase,idx)
   v=np.interp(pos,idx,s)
   err=ndimage.uniform_filter1d((v-y)**2,161)
   energy=ndimage.uniform_filter1d(y*y,161)
   w=np.exp(-np.clip(err/np.maximum(energy,1e-12),0,6))*(1-abs(shift)/5)
   weighted+=w*v;total+=w
  return weighted/total
 if mode=='psola':return y+conf*.65*(avg(y)-y)
 clean=avg(x); den=avg(y)
 if mode=='psola_raw':return y+conf*.5*(clean-y)
 if mode=='psola_detail':return y+conf*.5*(clean-den)
 return y+conf*(den-y)
