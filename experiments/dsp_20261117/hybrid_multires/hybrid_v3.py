"""PSD-only hybrid: original v3 restoration, optional new noise estimation."""
from dataclasses import dataclass, replace
import numpy as np
from scipy import signal, ndimage
from enhance_gain import GainConfig, init_noise_psd, gain_candidate_b, track_noise_spp
@dataclass(frozen=True)
class Config:
 n_fft:int=1024; hop:int=256; noise_fraction:float=.2; noise_scale:float=1.; dd_alpha:float=.94; floor:float=.10; dd_mix:float=.40; harmonic_weight:float=2.; harmonic_drive:float=.8; smooth_time:float=0.; smooth_freq:float=0.; adapt:float=0.; adapt_gate:float=1.6; dry_mix:float=0.
 multi_mix:float=0.; multi_fft:int=2048; multi_freq:float=1.5; multi_time:float=2.; multi_scale:float=1.3; multi_split_hz:float=0.;
 psd_low_scale:float=1.; psd_high_scale:float=1.; speech_protect:float=0.; gain_power:float=1.; log_smooth:bool=False
 init_quantile:float=.15; init_dof:float=3.; init_shrink_beta:float=.5; init_adaptive_dof:bool=False
def enhance_single(x,sr,cfg=Config(), *, estimator="imcra", blend=1.0):
 if estimator not in ("v3", "imcra", "spp", "quantile") or not 0 <= blend <= 1: raise ValueError("invalid estimator or blend")
 if sr != 16000: raise ValueError("16 kHz required")
 x=np.asarray(x,dtype=np.float64)
 if x.ndim != 1 or not np.isfinite(x).all(): raise ValueError("expected finite mono waveform")
 if len(x)<64 or np.max(np.abs(x))<1e-12:return x.copy()
 nfft=min(cfg.n_fft,len(x));hop=min(cfg.hop,nfft//2)
 _,times,z=signal.stft(x,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary='zeros',padded=True)
 power=np.abs(z)**2; centers=np.rint(times*sr).astype(int)
 full=np.flatnonzero((centers>=nfft//2)&(centers+nfft//2<=len(x)))
 if not len(full): full=np.arange(z.shape[1])
 energies=power[:,full].mean(axis=0);k=max(1,int(np.ceil(cfg.noise_fraction*len(full))));selected=full[np.argsort(energies,kind='stable')[:k]]
 noise=np.maximum(ndimage.uniform_filter1d(power[:,selected].mean(axis=1),size=3,mode='nearest'),1e-18)*cfg.noise_scale
 # Use exact v3 STFT power, not a filtered waveform.
 if estimator != "v3" and blend > 0:
  gc=GainConfig(n_fft=nfft,hop=hop,imcra_V=max(1,round(15*128/hop)),init_quantile=cfg.init_quantile,init_dof=cfg.init_dof,init_shrink_beta=cfg.init_shrink_beta,init_adaptive_dof=cfg.init_adaptive_dof)
  init=init_noise_psd(power,gc)
  freq=np.fft.rfftfreq(nfft,1/sr)
  init=init*(cfg.psd_low_scale**(1/(1+(freq/700)**2)))*(cfg.psd_high_scale**((freq/2500)**2/(1+(freq/2500)**2)))
  if estimator=="imcra":
   _,post,_=gain_candidate_b(power,gc,init)
   tracked=np.column_stack((init,post[:,:-1]))
  elif estimator=="spp": tracked=track_noise_spp(power,init,gc)
  else: tracked=np.broadcast_to(init[:,None],power.shape)
  noise_matrix=np.exp((1-blend)*np.log(np.maximum(noise[:,None],1e-18))+blend*np.log(np.maximum(tracked*cfg.noise_scale,1e-18)))
 else: noise_matrix=None
 gain_dd=np.empty_like(power);gamma=np.empty_like(power);prev=np.zeros_like(noise);noise_frame=noise.copy()
 for t in range(z.shape[1]):
  if noise_matrix is not None: noise_frame=noise_matrix[:,t]
  g=power[:,t]/np.maximum(noise_frame,1e-18); gamma[:,t]=g;inst=np.maximum(g-1,0)
  xi=inst if t==0 else cfg.dd_alpha*prev/np.maximum(noise_frame,1e-18)+(1-cfg.dd_alpha)*inst
  w=np.clip(xi/(1+xi),cfg.floor,1.0);gain_dd[:,t]=w;prev=w*w*power[:,t]
  if cfg.adapt>0 and noise_matrix is None:
   is_noise=g<cfg.adapt_gate;noise_frame=np.where(is_noise,(1-cfg.adapt)*noise_frame+cfg.adapt*power[:,t],noise_frame);noise_frame=np.maximum(noise*.15,np.minimum(noise_frame,noise*5.0))
 xi_tsnr=gamma*gain_dd**2
 if cfg.harmonic_weight>0:
  g0=np.clip(xi_tsnr/(1+xi_tsnr),cfg.floor,1.0);_,prelim=signal.istft(g0*z,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary=True);prelim=prelim[:len(x)]
  rms=max(float(np.sqrt(np.mean(prelim**2))),1e-9);harmonic=np.tanh(cfg.harmonic_drive*prelim/rms)*(rms/cfg.harmonic_drive)
  _,_,zh=signal.stft(harmonic,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary='zeros',padded=True);regenerated=np.minimum(np.abs(zh)**2,power)
  xi_tsnr=xi_tsnr+cfg.harmonic_weight*regenerated/(power/np.maximum(gamma,1e-15)+1e-18)
 gain_tsnr=np.clip(xi_tsnr/(1+xi_tsnr),cfg.floor,1.0);gain=cfg.dd_mix*gain_dd+(1-cfg.dd_mix)*gain_tsnr
 if cfg.smooth_freq>0 or cfg.smooth_time>0:
  raw=gain
  if cfg.log_smooth: gain=np.exp(ndimage.gaussian_filter(np.log(np.maximum(gain,1e-12)),sigma=(cfg.smooth_freq,cfg.smooth_time),mode='nearest'))
  else: gain=ndimage.gaussian_filter(gain,sigma=(cfg.smooth_freq,cfg.smooth_time),mode='nearest')
  if cfg.speech_protect: gain=gain+cfg.speech_protect*np.maximum(raw-gain,0)*np.clip((gamma-3)/8,0,1)
 if cfg.gain_power!=1: gain=np.clip(gain**cfg.gain_power,cfg.floor,1.)
 if cfg.dry_mix:gain=(1-cfg.dry_mix)*gain+cfg.dry_mix
 _,y=signal.istft(gain*z,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary=True);y=y[:len(x)]
 if len(y)<len(x):y=np.pad(y,(0,len(x)-len(y)))
 return y


def enhance(x,sr,cfg=Config(), *, estimator="imcra", blend=1.0):
 y=enhance_single(x,sr,cfg,estimator=estimator,blend=blend)
 if not cfg.multi_mix: return y
 alt=replace(cfg,n_fft=cfg.multi_fft,hop=cfg.hop,smooth_freq=cfg.multi_freq,smooth_time=cfg.multi_time,noise_scale=cfg.multi_scale,multi_mix=0.)
 z=enhance_single(x,sr,alt,estimator=estimator,blend=blend)
 delta=z-y
 if cfg.multi_split_hz>0:
  sos=signal.butter(4,cfg.multi_split_hz,fs=sr,btype='lowpass',output='sos');delta=signal.sosfiltfilt(sos,delta)
 return y+cfg.multi_mix*delta
