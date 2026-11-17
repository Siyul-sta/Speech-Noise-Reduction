import numpy as np,json
from pathlib import Path
from scipy import signal,ndimage
from previous import enhance,Config
def process(x,y,mode):
 kw=dict(fs=16000,nperseg=1536,noverlap=1280)
 z=signal.stft(x,**kw)[2]
 p=ndimage.gaussian_filter(abs(z)**2,(.7,2))
 floor=np.quantile(p,.15,axis=1)+1e-15
 log=np.log(floor)
 smooth=ndimage.gaussian_filter1d(log,12)
 if mode in ('white_line','white_line2'):ref=smooth
 else:ref=np.median(log)
 amt=.40 if mode in ('white','white_line') else .65
 g=np.exp(np.clip(amt*(ref-log),-3,2))
 # Use one zero-phase transfer function on the padded waveform, and undo it after denoising.
 padded=np.pad(x,(1536,1536),mode='reflect')
 grid=np.linspace(0,1,len(np.fft.rfft(padded)))
 G=np.interp(grid,np.linspace(0,1,len(g)),g)
 v=np.fft.irfft(np.fft.rfft(padded)*G,len(padded))[1536:-1536]
 cfg=Config(**json.loads(Path(__file__).with_name('pesq_config.json').read_text())['config'])
 w=enhance(v,16000,cfg,estimator='quantile',blend=1.)
 exp=.8 if mode in ('white2','white_line2') else 1.
 w=np.fft.irfft(np.fft.rfft(np.pad(w,(1536,1536),mode='reflect'))/G**exp,len(padded))[1536:-1536]
 return .45*y+.55*w
