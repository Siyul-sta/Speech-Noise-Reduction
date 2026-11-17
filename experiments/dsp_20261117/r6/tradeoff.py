import numpy as np,json
from pathlib import Path
from previous import Config,enhance
from scipy import signal,ndimage
def run(x,y,mode):
 cfg=Config(**json.loads(Path(__file__).with_name('selected_config.json').read_text())['config'])
 safe=enhance(x,16000,cfg,estimator='quantile',blend=1.)
 kw=dict(fs=16000,nperseg=768,noverlap=640)
 f,_,a=signal.stft(y,**kw);z=signal.stft(x,**kw)[2];b=signal.stft(safe,**kw)[2]
 e=1e-18
 en=ndimage.gaussian_filter(abs(b)**2,(2,2))
 # Protect high-confidence bass components. Focus mild emphasis on mid bands.
 noise=np.quantile(ndimage.gaussian_filter(abs(z)**2,(1,2)),.16,axis=1)[:,None]+e
 sn=en/(en+noise)
 low=1/(1+(f[:,None]/400)**4)
 band=np.exp(-.5*(np.log2(np.maximum(f,1)/500)/.65)**2)[:,None]
 if mode=='v5trade_full':mix=.85*low*sn
 elif mode=='v5trade_band':mix=.80*low
 else:mix=.7*low*sn+.25*sn
 out=a+(b-a)*np.clip(mix,0,1)
 gain=10**(4*band/20)
 # Preserve the absolute low-frequency amplitude after matching speech-band level.
 pilot=signal.istft(out*gain,**kw)[1][:len(x)]
 alpha=np.clip(np.dot(x,pilot)/max(np.dot(pilot,pilot),e),.1,3)
 spec=out*gain*alpha
 if mode=='v5trade_bass':spec+=(b-spec)*low*sn
 return signal.istft(spec,**kw)[1][:len(x)]
