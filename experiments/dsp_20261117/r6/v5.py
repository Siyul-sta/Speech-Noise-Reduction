import numpy as np
from scipy import signal,ndimage,linalg
from loop import recover
from dynamic import apply
def improve(x,y,mode):
 y=apply(x,recover(x,y,"loop_safe"),'dyn_global')
 if mode=='v5base':return y
 if mode.startswith("v5nlocal"):
  from nonlocal_filter import enhance
  return enhance(x,y,mode)
 if mode.startswith("v5short"):
  from coherence import repair
  from equal_loudness import adapt
  from pockets import restore
  adjusted = repair(x,y,'v5carrier_internal_1024_19_0.7')
  if mode == 'v5short_soft':
   return restore(x,adjusted,'v5r6_slope_20_16')
  return adapt(x,adjusted,mode)
 if mode.startswith("v5fuse"):
  from fuse_carrier import fuse
  return fuse(x,y,mode)
 if mode.startswith("v5carrier"):
  from coherence import repair
  return repair(x,y,mode)
 if mode.startswith("v5r6_"):
  from pockets import restore
  return restore(x,y,mode)
 if mode.startswith("v5proj"):
  from projection import run
  return run(x,y,mode)
 if mode.startswith("v5trade"):
  from tradeoff import run
  return run(x,y,mode)
 if mode.startswith("v5taper"):
  from taper import process
  return process(x,y,mode)
 if mode.startswith("v5clear"):
  from clarity import run
  return run(x,y,mode)
 if mode.startswith("v5vb"):
  from voiceband import process
  return process(x,y,mode)
 if mode.startswith("v5harm"):
  from harmonic import process
  return process(x,y,mode)
 if mode.startswith("v5res_"):
  from resonance import run
  return run(x,y,mode)
 if mode.startswith(("v5safe","v5front")):
  from parallel_band import process
  return process(x,y,mode)
 if mode.startswith("v5prior"):
  from prior import run
  return run(x,y,mode)
 if mode.startswith("v5protect"):
  from preserve import run
  return run(x,y,mode)
 if mode.startswith("v5mid"):
  from midpath import run
  return run(x,y,mode)
 if mode.startswith("v5form"):
  from formant import run
  return run(x,y,mode)
 if mode in ("v5rasta","v5syllable","v5antimod","v5attack","v5sustain"):
  from modfilter import run
  return run(x,y,mode)
 if mode.startswith("v5colour"):
  from colour import color
  return color(x,y,mode)
 if mode.startswith("v5mvdr"):
  from mvdr import filter_signal
  return filter_signal(x,y,mode)
 if mode.startswith("v5scale"):
  from scale import run
  return run(x,y,mode)
 if mode.startswith(("v5clip","v5detail","v5mod")):
  from transient import process
  return process(x,y,mode)
 kw=dict(fs=16000,nperseg=640,noverlap=480)
 hz,t,a=signal.stft(y,**kw); p=abs(a)**2;e=1e-18
 if mode.startswith('v5lpc'):
  # Regularized autocorrelation LPC from preemphasized spectra for each frame.
  pre=abs(1-.92*np.exp(-2j*np.pi*hz/16000))**2
  ac=np.fft.irfft(ndimage.gaussian_filter1d(p,1,axis=1)*pre[:,None],axis=0)
  order=16
  coeff=[]
  for r in ac.T:
   first=r[:order].copy();first[0]=max(first[0]*1.012,e)
   coeff.append(np.r_[1,-linalg.solve_toeplitz(first,r[1:order+1])])
  c=np.array(coeff);k=np.arange(order+1)
  num=np.abs(np.fft.rfft(c*.65**k,n=640,axis=1)).T
  den=np.abs(np.fft.rfft(c*.86**k,n=640,axis=1)).T
  gain=np.clip(num/(den+e),.4,2.3)
  if mode=='v5lpc_inv':gain=1/gain
  if mode=='v5lpc_soft':gain=np.sqrt(gain)
 else:
  z=signal.stft(x,**kw)[2];pw=ndimage.gaussian_filter(abs(z)**2,(1,1))
  floor=np.quantile(pw,.2,axis=1)[:,None]+e
  hz2=hz[:,None]
  low=1/(1+(hz2/400)**4)
  # Detect periodicity from the preliminary speech estimate, suppress sub-F0 rumble.
  ac=np.fft.irfft(p,axis=0);lags=np.arange(42,229)
  val=ac[lags]/np.maximum(ac[0],e)
  k=np.argmax(val,axis=0)
  f0=ndimage.median_filter(16000/lags[k],5)
  voiced=np.clip((val[k,np.arange(len(k))]-.30)/.45,0,1)
  cut=f0*(.68 if mode=='v5hp_soft' else .88)*voiced+280*(1-voiced)
  gain=1/np.sqrt(1+(cut/np.maximum(hz2,1))**6)
  if mode=='v5hband':
   dist=np.abs((hz2+.5*f0)%f0-.5*f0)
   comb=.25+.75*np.exp(-.5*(dist/35)**2)
   gain*=1-low*voiced*(1-comb)
  if mode=='v5bass':
   # Add low-band support from the original only in confidently voiced harmonics.
   rel=p/(p+floor)
   a=a+.65*low*voiced*rel*(z-a)
 out=signal.istft(a*gain,**kw)[1][:len(x)]
 return out*np.clip(np.dot(out,x)/max(np.dot(out,out),e),.75,1.5)
