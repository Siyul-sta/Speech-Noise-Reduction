import numpy as np
from scipy import signal,ndimage,fft
def periods(x,y):
 n=len(x)
 resid=x-y
 nfft=fft.next_fast_len(2*n)
 f=fft.rfft(resid,nfft); ps=abs(f)**2
 # Sharpen broadband recurrence peaks; pure tones alone are ambiguous at long lags.
 bg=ndimage.gaussian_filter1d(ps,48)+np.max(ps)*1e-10
 ac=fft.irfft(ps/bg**.75,nfft)[:n]
 lags=np.arange(96,min(n*7//10,n-1024))
 curve=ac[lags]/max(ac[0],1e-18)/(1-lags/n)
 peaks=signal.find_peaks(curve)[0]
 if not len(peaks):return 0,0.
 k=peaks[np.argmax(curve[peaks]*np.sqrt(1-lags[peaks]/n))]
 # Prefer short fundamental lag if it explains the same repetition.
 short=peaks[(curve[peaks]>.9*curve[k])&(peaks<=k)]
 k=short[0] if len(short) else k
 return int(lags[k]),curve[k]
def recover(x,y,mode):
 period,rho=periods(x,y)
 if period==0 or rho<.10:return y
 n=len(x)
 # Analyze delayed copies with exactly the same FFT grid, including non-hop-multiple delays.
 kw=dict(fs=16000,nperseg=512,noverlap=384)
 st=lambda v:signal.stft(v,**kw)[2]
 a=st(y); z=st(x)
 frames=np.arange(z.shape[1])*128
 num=np.zeros(z.shape,complex);den=np.zeros(z.shape);sq=np.zeros_like(den)
 count=np.zeros(z.shape[1]);eps=1e-17
 # An aligned copy with weak inferred speech provides a cleaner noise observation.
 for m in range(-min(10,(n-1)//period),min(10,(n-1)//period)+1):
  step=m*period
  shifted=np.roll(x,step); v=st(shifted)
  prior=st(np.roll(y,step))
  valid=(frames-step>=256)&(frames-step<n-256)
  s=ndimage.gaussian_filter(abs(prior)**2,(1.,1.5))
  floor=np.quantile(abs(z)**2,.25,axis=1)[:,None]+eps
  w=valid/(s+.10*floor+eps)
  num+=w*v;den+=w;sq+=w*abs(v)**2;count+=valid
 coherent=abs(num)**2/(sq*den+eps)
 noise=num/np.maximum(den,eps)
 gate=np.clip((coherent-.25)/.65,0,1)*np.clip((count-1),0,1)
 weight=np.clip((rho-.10)/.35,0,1)
 if mode=='loop_safe':weight*=.70
 cleaned=signal.istft(z-weight*gate*noise,**kw)[1][:n]
 if mode=='loop_direct':
  return y+weight*np.clip(rho,0,1)*(cleaned-y)
 if mode=='loop_fuse':
  # Trust subtraction in coherent bins; retain the first-pass result elsewhere.
  return signal.istft(a+weight*gate*(z-noise-a),**kw)[1][:n]
 from previous import enhance,Config
 import json
 from pathlib import Path
 cfg=Config(**json.loads(Path(__file__).with_name('pesq_config.json').read_text())['config'])
 out=enhance(cleaned,16000,cfg,estimator='quantile',blend=1)
 return .15*y+.85*out

def project(x,y,mode):
    period,rho=periods(x,y)
    if not period or rho<.075:return y
    # Broadband repetition evidence must agree across independent frequency bands.
    n=len(x)
    kw=dict(fs=16000,nperseg=512,noverlap=384)
    st=lambda v:signal.stft(v,**kw)[2]
    z=st(x);a=st(y)
    e=1e-18
    prior=ndimage.gaussian_filter(abs(a)**2,(1.2,1.4))
    floor=np.quantile(abs(z)**2,.22,axis=1)[:,None]+e
    total=np.zeros_like(prior);correction=np.zeros_like(a)
    centers=np.arange(a.shape[1])*128
    align=[]
    for sign in (-1,1):
        shift=period*sign;valid=(centers-shift>256)&(centers-shift<n-256)
        other=st(np.roll(x,shift));initial=st(np.roll(y,shift))
        # Require a sharp delayed-noise correlation in at least part of the spectrum.
        residual=z-a; r=other-initial
        prod=residual[:,valid]*r[:,valid].conj()
        if prod.size:
            coh=np.maximum(0,prod.real.sum(1))/(np.sqrt((abs(residual[:,valid])**2).sum(1)*(abs(r[:,valid])**2).sum(1))+e)
            align.append(np.mean(coh>.8))
        else:continue
        var=ndimage.gaussian_filter(abs(initial)**2,(1.2,1.4))
        frac=(prior+.01*floor)/(prior+var+.03*floor)
        delta=(z-other)-(a-initial)
        w=valid[None,:]*coh[:,None]**2
        correction+=w*frac*delta;total+=w
    if not align or max(align)<.035:return y
    strength=.9 if mode=='project' else .5
    out=a+strength*correction/np.maximum(total,.65)
    return signal.istft(out,**kw)[1][:n]
