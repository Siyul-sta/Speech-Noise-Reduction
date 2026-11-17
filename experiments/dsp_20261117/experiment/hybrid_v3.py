"""PSD-only hybrid: original v3 restoration, optional new noise estimation."""
from dataclasses import dataclass, replace
import numpy as np
from scipy import signal, ndimage
from enhance_gain import GainConfig, init_noise_psd, gain_candidate_b, track_noise_spp
@dataclass(frozen=True)
class Config:
 structure:str="base"
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
  if cfg.structure in ("quiet","track","line","joint","quiet_context"):
   from adaptive import estimate
   tracked=estimate(power,init,full,"quiet" if cfg.structure=="quiet_context" else cfg.structure)
  if cfg.structure in ("cep","cep_lift","cep_pitch","pitch","pitch_dry","pitch_deep","cross","cross_low","rms_log","guard","guard_soft","guard_geom","relative","relative_geom","log_track","log_frame","log_time","log_wide","log_spectral","log_deep","log_blend","steady","deep_global","deep_lift"):
   from adaptive import relative_noise
   tracked=relative_noise(power,init,full)
  noise_matrix=np.exp((1-blend)*np.log(np.maximum(noise[:,None],1e-18))+blend*np.log(np.maximum(tracked*cfg.noise_scale,1e-18)))
 else: noise_matrix=None
 gain_dd=np.empty_like(power);gamma=np.empty_like(power);prev=np.zeros_like(noise);noise_frame=noise.copy()
 for t in range(z.shape[1]):
  if noise_matrix is not None: noise_frame=noise_matrix[:,t % noise_matrix.shape[1]]
  g=power[:,t]/np.maximum(noise_frame,1e-18); gamma[:,t]=g;inst=np.maximum(g-1,0)
  xi=inst if t==0 else cfg.dd_alpha*prev/np.maximum(noise_frame,1e-18)+(1-cfg.dd_alpha)*inst
  w=np.clip(xi/(1+xi),cfg.floor,1.0);gain_dd[:,t]=w;prev=w*w*power[:,t]
  if cfg.adapt>0 and noise_matrix is None:
   is_noise=g<cfg.adapt_gate;noise_frame=np.where(is_noise,(1-cfg.adapt)*noise_frame+cfg.adapt*power[:,t],noise_frame);noise_frame=np.maximum(noise*.15,np.minimum(noise_frame,noise*5.0))
 if cfg.structure in ("bidirectional","consensus","future"):
  back=np.empty_like(gain_dd)
  previous=np.zeros(len(power))
  for t in range(power.shape[1]-1,-1,-1):
   inst=np.maximum(gamma[:,t]-1,0)
   xi=inst if t==power.shape[1]-1 else cfg.dd_alpha*previous/noise_matrix[:,t]+(1-cfg.dd_alpha)*inst
   back[:,t]=np.clip(xi/(1+xi),cfg.floor,1)
   previous=power[:,t]*back[:,t]**2
  if cfg.structure=="bidirectional": gain_dd=(gain_dd+back)/2
  elif cfg.structure=="future": gain_dd=np.maximum(gain_dd,back)
  else: gain_dd=np.sqrt(gain_dd*back)
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
 if cfg.structure in ("context","quiet_context"):
  from adaptive import context_gain
  local=context_gain(power,noise_matrix)
  gain=np.sqrt(gain*local)
 if cfg.structure in ("ridge","odds","odds_ridge"):
  from adaptive import refine
  gain=refine(gain,power,noise_matrix,raw,cfg.structure)
 if cfg.structure in ("bark","envelope","barkgeom"):
  from adaptive import auditory_gain
  gain=auditory_gain(gain,power,sr,cfg.structure)
 if cfg.structure in ("bark_level","geom_level","notch_geom","geom_fusion","relative_geom"):
  from adaptive import auditory_gain,compensate
  gain=auditory_gain(gain,power,sr,"bark" if cfg.structure=="bark_level" else "barkgeom")
  gain=compensate(gain,power,noise_matrix)
 if cfg.structure in ("band_log","band_cap"):
  from adaptive import band_mask,compensate
  gain=compensate(band_mask(gain,power,sr,cfg.structure),power,noise_matrix)
 if cfg.structure in ("log_track","log_frame","log_time","log_wide","log_spectral","log_deep","log_blend","steady","deep_global","deep_lift"):
  from adaptive import band_mask,compensate,rebalance
  target=band_mask(gain,power,sr,"band_log")
  if cfg.structure=="log_time":
   target=np.exp(ndimage.gaussian_filter1d(np.log(target),1.8,axis=1))
  if cfg.structure=="log_wide":
   target=band_mask(target,power,sr,"band_log")
  if cfg.structure in ("deep_global","deep_lift","steady"):
   if cfg.structure!="steady": target=band_mask(target,power,sr,"band_log")
   if cfg.structure!="deep_global":
    lg=np.log(np.maximum(target,.001))
    variation=np.abs(np.diff(lg,prepend=lg[:,:1],axis=1))
    stability=np.exp(-3*ndimage.gaussian_filter(variation,(2,1)))
    target=np.exp(lg+stability*(ndimage.gaussian_filter1d(lg,2,axis=1)-lg))
   gain=compensate(target,power,noise_matrix)
  elif cfg.structure in ("log_spectral","log_deep","log_blend","steady","deep_global","deep_lift"):
   from adaptive import spectral_balance
   if cfg.structure=="log_deep":
    target=band_mask(target,power,sr,"band_log")
   if cfg.structure=="log_blend":
    target=.75*target+.25*gain
   gain=spectral_balance(target,power,noise_matrix,sr)
  else:
   gain=compensate(target,power,noise_matrix) if cfg.structure=="log_track" else rebalance(target,power,noise_matrix)
 if cfg.structure in ("guard","guard_soft","guard_geom"):
  from adaptive import band_mask,compensate,auditory_gain
  target=compensate(band_mask(gain,power,sr,"band_log"),power,noise_matrix)
  safe=compensate(auditory_gain(gain,power,sr,"barkgeom"),power,noise_matrix)
  speech_energy=ndimage.gaussian_filter1d(np.sum(power*gain**2,axis=0),2)
  weight=np.clip(speech_energy/np.maximum(np.quantile(speech_energy,.85),1e-18),0,1)
  if cfg.structure=="guard_geom": weight=.50
  elif cfg.structure=="guard_soft": weight=.6*weight
  gain=target*(1-weight)+safe*weight
 if cfg.structure in ("cross","cross_low","rms_log"):
  from adaptive import band_mask,compensate,auditory_gain
  perceptual=band_mask(gain,power,sr,"band_log")
  if cfg.structure=="rms_log":
   estimated_energy=max(np.sum(power-noise_matrix),0)
   factor=np.sqrt(estimated_energy/max(np.sum(power*perceptual**2),1e-18))
   gain=perceptual*np.clip(factor,1.,3.)
  else:
   safe=compensate(auditory_gain(gain,power,sr,"barkgeom"),power,noise_matrix)
   split=600 if cfg.structure=="cross_low" else 1300
   high=1/(1+(split/np.maximum(freq,1))**3)
   gain=safe*(1-high[:,None])+compensate(perceptual,power,noise_matrix)*high[:,None]
 if cfg.structure in ("pitch","pitch_dry","pitch_deep"):
  from adaptive import band_mask,compensate,pitch_protection
  voiced=pitch_protection(z,gain,noise_matrix,sr,nfft)
  target=band_mask(gain,power,sr,"band_log")
  if cfg.structure=="pitch_deep": target=band_mask(target,power,sr,"band_log")
  safe=1. if cfg.structure=="pitch_dry" else gain
  target=target+voiced*(safe-target)*.65
  gain=compensate(target,power,noise_matrix)
 if cfg.structure in ("cep","cep_lift","cep_pitch"):
  from adaptive import band_mask,compensate,cepstral_mask,pitch_protection
  target=band_mask(gain,power,sr,"band_log")
  target=cepstral_mask(target,sr,"lifter" if cfg.structure=="cep_lift" else "smooth")
  if cfg.structure=="cep_pitch":
   voiced=pitch_protection(z,gain,noise_matrix,sr,nfft)
   target+=.65*voiced*(gain-target)
  gain=compensate(target,power,noise_matrix)
 if cfg.gain_power!=1: gain=np.clip(gain**cfg.gain_power,cfg.floor,1.)
 if cfg.dry_mix:gain=(1-cfg.dry_mix)*gain+cfg.dry_mix
 _,y=signal.istft(gain*z,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary=True);y=y[:len(x)]
 if len(y)<len(x):y=np.pad(y,(0,len(x)-len(y)))
 return y


def enhance(x,sr,cfg=Config(), *, estimator="imcra", blend=1.0):
 if cfg.structure in ("tri_dynamic","tri_high","tri384","tri_strong","tri_auditory"):
  from adaptive import notch_lines,transient_mix
  base=replace(cfg,structure="cross_low")
  original=enhance(x,sr,base,estimator=estimator,blend=blend)
  short=384 if cfg.structure=="tri384" else 512
  branch=replace(base,n_fft=short,hop=128,dd_alpha=cfg.dd_alpha**.5,smooth_time=3,multi_mix=0.)
  if cfg.structure=="tri_auditory": branch=replace(branch,structure="log_track")
  alt=enhance_single(notch_lines(x,sr),sr,branch,estimator=estimator,blend=blend)
  if cfg.structure in ("tri_dynamic","tri_high"):
   return transient_mix(x,alt,original,sr,"high" if cfg.structure=="tri_high" else "flux")
  if cfg.structure=="tri_strong": return .3*original+.7*alt
  return .5*(original+alt)
 if cfg.structure in ("spin3","spin_strict","spin_pitch","spin_log","spin_band"):
  mode={"spin_strict":"strict","spin_pitch":"pitch_deep","spin_log":"log_track","spin_band":"guard_geom"}.get(cfg.structure,"cross_low")
  base=replace(cfg,structure=mode)
  outputs=[]
  for offset in (0, cfg.hop//3, 2*cfg.hop//3):
   shifted=np.pad(x,(offset,0))
   outputs.append(enhance(shifted,sr,base,estimator=estimator,blend=blend)[offset:])
  return np.mean(outputs,axis=0)
 if cfg.structure in ("phase_refine","residual"):
  from adaptive import reconstruct
  base=enhance(x,sr,replace(cfg,structure="cross_low"),estimator=estimator,blend=blend)
  return reconstruct(x,base,sr,"shrink" if cfg.structure=="residual" else "phase")
 if cfg.structure in ("cycle","reverse_mix","dense","tri","tri_short","tri_long"):
  base=replace(cfg,structure="cross_low")
  original=enhance(x,sr,base,estimator=estimator,blend=blend)
  if cfg.structure=="cycle":
   delay=cfg.hop//2
   alt=enhance(np.pad(x,(delay,0)),sr,base,estimator=estimator,blend=blend)[delay:]
  elif cfg.structure=="reverse_mix":
   alt=enhance(x[::-1],sr,base,estimator=estimator,blend=blend)[::-1]
  else:
   from adaptive import notch_lines
   clean_input=notch_lines(x,sr)
   short={"dense":1024,"tri":768,"tri_short":512,"tri_long":1536}[cfg.structure]
   branch=replace(base,n_fft=short,hop=128,dd_alpha=cfg.dd_alpha**.5,smooth_time=3,multi_mix=0.)
   alt=enhance_single(clean_input,sr,branch,estimator=estimator,blend=blend)
  return .5*(original+alt)
 if cfg.structure in ("drift","drift_pitch","stack","stack_cross","phase_pitch"):
  from adaptive import drifting_notch
  x=drifting_notch(x,sr,"phase" if cfg.structure=="phase_pitch" else "normal")
  cfg=replace(cfg,structure={"drift":"cross_low","stack":"persist_pitch","stack_cross":"persist"}.get(cfg.structure,"pitch_deep"))
 if cfg.structure in ("persist","strict","persist_pitch","persist_guard"):
  from adaptive import persistent_notch
  x=persistent_notch(x,sr,"strict" if cfg.structure=="strict" else "normal")
  cfg=replace(cfg,structure={"persist_pitch":"pitch_deep","persist_guard":"guard_geom"}.get(cfg.structure,"cross_low"))
 if cfg.structure in ("cep","cep_lift","cep_pitch","pitch","pitch_dry","pitch_deep","cross","cross_low","rms_log","guard","guard_soft","guard_geom","notch","notch_geom","log_track","log_frame","log_time","log_wide","log_spectral","log_deep","log_blend"):
  from adaptive import notch_lines
  x=notch_lines(x,sr)
 y=enhance_single(x,sr,cfg,estimator=estimator,blend=blend)
 if not cfg.multi_mix: return y
 alt=replace(cfg,n_fft=cfg.multi_fft,hop=cfg.hop,smooth_freq=cfg.multi_freq,smooth_time=cfg.multi_time,noise_scale=cfg.multi_scale,multi_mix=0.)
 z=enhance_single(x,sr,alt,estimator=estimator,blend=blend)
 if cfg.structure in ("fusion","joint","geom_fusion"):
  from adaptive import combine
  return combine(x,y,z,sr,cfg.hop)
 delta=z-y
 if cfg.multi_split_hz>0:
  sos=signal.butter(4,cfg.multi_split_hz,fs=sr,btype='lowpass',output='sos');delta=signal.sosfiltfilt(sos,delta)
 return y+cfg.multi_mix*delta
