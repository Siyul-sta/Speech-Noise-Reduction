import numpy as np
from scipy import signal,ndimage
from dynamic import apply
def filter_tracks(x,y,mode):
    args=dict(fs=16000,nperseg=2048,noverlap=1792)
    _,_,z=signal.stft(x,**args)
    a=signal.stft(y,**args)[2]
    db=20*np.log10(np.maximum(abs(z),1e-12))
    prominence=db-ndimage.median_filter(db,(15,1))
    # Prefer tones present in the discarded residual, even during changes of speech pitch.
    ratio=abs(z-a)/(abs(z)+1e-12)
    evidence=np.clip(prominence/8,0,4)*np.clip(ratio,.1,1)
    bins,time=z.shape; idx=np.arange(bins)
    notch=np.ones_like(db)
    for _ in range(5):
        score=evidence[:,0]; paths=[]
        for t in range(1,time):
            padded=np.pad(score,(2,2),constant_values=-np.inf)
            alternatives=np.stack([padded[k:k+bins]-.32*abs(2-k) for k in range(5)])
            best=alternatives.argmax(axis=0)
            score=alternatives[best,idx]+evidence[:,t];paths.append(best-2)
        path=[score.argmax()]
        for ptr in reversed(paths):path.append(path[-1]+ptr[path[-1]])
        path=np.asarray(path[::-1])
        prom=prominence[path,np.arange(time)]
        confidence=np.mean(prom>8)
        if confidence<.65:break
        distance=idx[:,None]-path
        attenuation=.8*np.exp(-.5*(distance/1.35)**2)*np.clip((prom-3)/14,0,1)
        notch*=1-attenuation
        evidence[np.abs(distance)<6]=0
    if mode=='track_post':
        w=signal.istft(a*notch,**args)[1][:len(x)]
    else:
        from previous import Config,enhance
        from pathlib import Path
        import json
        cfg=Config(**json.loads(Path(__file__).with_name('pesq_config.json').read_text())['config'])
        v=signal.istft(z*notch,**args)[1][:len(x)]
        w=enhance(v,16000,cfg,estimator='quantile',blend=1)
        w=.5*(w+y)
    return apply(x,w,'dyn_global')
