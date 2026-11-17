import numpy as np
from scipy import signal,ndimage
from pockets import restore

def reconstruct(x, pilot, mixture, cleaned, coherence, gain, kw, mode):
    # Restore the speech envelope from the cleaned mixture using a noise floor
    # calculated separately within each frequency channel.
    smoothed = ndimage.gaussian_filter(abs(cleaned)**2,(1,1))
    floor = np.quantile(smoothed,.2,axis=1)[:,None]
    posterior = np.maximum(smoothed-floor,0)/(smoothed+1.e-18)
    kind = mode.split('_')[2]
    if kind == 'direct':
        a = cleaned * np.sqrt(gain)
    elif kind == 'sqrt':
        a = cleaned * np.sqrt(posterior)
    elif kind == 'hybrid':
        a = cleaned * (gain * posterior)**.25
    elif kind == 'high':
        f = np.linspace(0,8000,len(pilot))[:,None]
        weight = coherence + (1-coherence)*f**2/(f**2+900**2)
        a = pilot + weight*(cleaned*np.sqrt(gain)-pilot)
    else:
        restoration = cleaned*np.sqrt(gain)-pilot
        a = pilot + (.3 + .7*coherence)*restoration
    y = signal.istft(a,**kw)[1][:len(x)]
    return restore(x,y,'v5r6_slope_24_16')
