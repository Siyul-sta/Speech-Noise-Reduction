import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d,median_filter,uniform_filter1d
E=1e-18
def refine(z,a,mode):
 p=abs(z)**2
 q=abs(a)**2
 smooth=gaussian_filter(p,(1,2))
 lo,hi=np.quantile(smooth,[.12,.9],axis=1)
 reliability=np.clip(np.log((hi+E)/(lo+E))/5,.01,1)
 env=np.log((smooth+E)/(lo[:,None]+E))
 # Normalize each band's time variation to compare envelope activity across octaves.
 med=np.median(env,axis=1,keepdims=True)
 standardized=(env-med)/np.maximum(np.std(env,axis=1,keepdims=True),.4)
 if mode.startswith('mod'):
  weights=reliability**3
  template=gaussian_filter1d(standardized*weights[:,None],12,axis=0)/np.maximum(gaussian_filter1d(weights,12)[:,None],.001)
  shared=template*np.maximum(np.mean(template*standardized,axis=1,keepdims=True),0)
  g=np.exp(np.clip(shared,-2,.5)*(.7 if mode=='mod' else .35)*(1-reliability[:,None]))
  return a*g
 if mode.startswith('rank'):
  # Infer a stationary noise amplitude distribution from low-modulation bins.
  quant=np.quantile(smooth,[.08,.28,.60,.85],axis=1)
  background=quant[0]+np.minimum(quant[1]-quant[0],quant[0])
  uncertainty=np.minimum(quant[2]-quant[0],2*background)
  excess=(smooth-background[:,None])/np.maximum(uncertainty[:,None],E)
  g=1/(1+np.exp(-np.clip(excess-1,-20,20)))
  return a*((.45+.55*g) if mode=='rank' else np.sqrt(g))
 if mode.startswith('resynth'):
  # Extrapolate missing spectral envelope from reliable adjacent frequency bands.
  mag=np.sqrt(gaussian_filter(q,(1,1)))
  w=(reliability[:,None]*np.clip(env/3,.02,1))**2
  target=gaussian_filter1d(mag*w,6,axis=0)/np.maximum(gaussian_filter1d(w,6,axis=0),.00001)
  delta=np.maximum(target-mag,0)*(1-w)*np.clip(env/3,0,1)
  return a+(.5 if mode=='resynth' else 1)*delta*z/np.maximum(abs(z),1e-12)
 if mode.startswith("island"):
  from scipy.ndimage import binary_propagation
  # Link faint speech bins to nearby strong speech bins and to cross-band onsets.
  peak=np.quantile(gaussian_filter(q,(1,1)),.9,axis=1)[:,None]+E
  level=gaussian_filter(q,(1,1))/peak
  floor=np.quantile(p,.18,axis=1)[:,None]+E
  evidence=gaussian_filter(q,(1,1))/floor
  seed=(level>.25)&(evidence>2)
  reachable=(level>.008)&(evidence>.09)
  support=binary_propagation(seed,structure=np.ones((3,3)),mask=reachable)
  prob=gaussian_filter(support.astype(float),(1.4,1.5))
  gate=(.15+.85*prob)**(.5 if mode=="island_soft" else 1)
  return a*gate
 if mode.startswith("roll"):
  hz=np.linspace(0,8000,len(p))
  freq,order=map(float,mode.split("_")[1:])
  return a/np.sqrt(1+(hz[:,None]/freq)**(2*order))
 if mode.startswith("clarity"):
  hz=np.linspace(0,8000,len(p)); high=(1/(1+(2400/np.maximum(hz,1))**4))[:,None]
  # Consonants are brief and spectrally broad. Raise the bandwidth during coherent bursts.
  treble=hz>2400
  burst=gaussian_filter1d(np.mean(np.maximum(standardized[treble],0),axis=0),1)
  keep=np.clip((burst-.4)/1.8,0,1)
  floor=.08 if mode=="clarity" else .22
  return a*(1-high*(1-floor)*(1-.7*keep))
 if mode.startswith("shelf"):
  hz=np.linspace(0,8000,len(p))
  freq,db=map(float,mode.split("_")[1:])
  high=1/(1+(freq/np.maximum(hz,1))**4)
  return a*(10**(db*high/20))[:,None]
 if mode.startswith("bandwidth"):
  hz=np.linspace(0,8000,len(p))
  # In attenuated high-energy-noise regions, estimate recoverable bandwidth from contrast.
  quality=np.quantile(gaussian_filter(q,(2,2)),.85,axis=1)/(lo+E)
  qual=gaussian_filter1d(np.log1p(quality),20)
  w=1-np.exp(-qual)
  cut=1/(1+(hz/2400)**4)
  return a*(cut+(1-cut)*w**(2 if mode=="bandwidth" else 4))[:,None]
 raise ValueError(mode)
