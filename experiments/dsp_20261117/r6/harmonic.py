import numpy as np
from scipy import signal
from formant import run
from voiced import refine
def process(x,y,mode):
 shaped=run(x,y,'v5form_fixed')
 period,conf=refine(x,y,'track_only')
 phase=np.cumsum(2*np.pi/period); f0=16000/period
 total=np.zeros_like(x);sos=signal.butter(3,24,fs=16000,output='sos')
 innovation=x-shaped if mode in ('v5harm_input','v5harm_strong') else y-shaped
 for h in range(1,5):
  carrier=np.exp(1j*h*phase)
  env=signal.sosfiltfilt(sos,2*innovation*carrier.conj())
  support=1/(1+(h*f0/350)**6)
  total+=np.real(env*carrier)*support
 amount=.9 if mode=='v5harm_strong' else .5
 return shaped+amount*conf*total
