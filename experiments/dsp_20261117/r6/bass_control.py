"""Time-varying low-band recovery constrained by the speech-band envelope."""
import numpy as np
from scipy import signal,ndimage
from pockets import restore

def combine(x, pilot, mode):
    strong = restore(x, pilot, 'v5r6_slope_24_16')
    bass = signal.sosfiltfilt(signal.butter(3, 320, fs=16000,output='sos'), pilot-strong)
    env = lambda v: np.sqrt(np.maximum(ndimage.uniform_filter1d(v*v, 801), 1.e-15))
    reference = env(strong)
    ratio = reference / env(bass)
    kind = mode.split('_')[2]
    if kind == 'bound':
        gain = np.minimum(.6,.35*ratio)
    elif kind == 'weak':
        gain = np.minimum(.45,.14*ratio)
    elif kind == 'mask':
        fraction = reference / (np.quantile(reference,.8)+1.e-12)
        gain = np.minimum(.65,.5*ratio)*np.clip(fraction,0,1)
    else:
        gain = .45/(1+ratio**2)
    return strong + bass*gain
