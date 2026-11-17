"""Continuous two-sided speech-band filter with independent transition widths."""
import numpy as np
from scipy import signal
def equalize(x, y, mode):
    _, _, center, low_width, high_width, low_depth, high_depth, _ = mode.split('_')
    center, low_width, high_width, low_depth, high_depth = map(float, (center,low_width,high_width,low_depth,high_depth))
    options = dict(fs=16000, nperseg=1024, noverlap=768)
    f, _, a = signal.stft(y, **options)
    octaves = np.log2(np.maximum(f,1)/center)
    width = np.where(octaves<0, low_width, high_width)
    depth = np.where(octaves<0, low_depth, high_depth)
    gain = 10**((np.exp(-.5*(octaves/width)**2)-1)*depth/20)
    out = signal.istft(a*gain[:,None], **options)[1][:len(x)]
    return out*np.clip(x@out/max(out@out,1.e-18),.1,8.)
