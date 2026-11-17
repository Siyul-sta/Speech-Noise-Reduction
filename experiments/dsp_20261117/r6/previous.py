"""Noisy-only experimental single-channel speech power and multiframe estimators."""
import numpy as np
from scipy import signal, ndimage
from dataclasses import replace
from core import Config, enhance as baseline
from adaptive import pitch_protection

def enhance(x,sr,cfg,*,estimator,blend):
    mode=cfg.structure
    orig=baseline(x,sr,replace(cfg,structure="cross_low"),estimator=estimator,blend=blend)
    if mode.startswith("space_"):
        from subspace import filterbank
        return filterbank(x,orig,sr,mode)
    if mode.startswith("fit_"):
        # Restore the part of mixture amplitude coherent with the final speech estimate.
        est=enhance(x,sr,replace(cfg,structure=mode[4:]),estimator=estimator,blend=blend)
        scale=np.dot(x,est)/max(np.dot(est,est),1e-18)
        return est*np.clip(scale,.7,3.)
    if mode=="cross_low" or len(x)<1024: return orig
    if mode.startswith("dct"):
        from dct_filter import estimate
        alt=estimate(x,orig,512,"mask" if mode=="dct_mask" else "consensus")
        return .5*(alt+orig)
    size=2048 if mode.startswith("eq_") else 768
    args=dict(fs=sr,nperseg=size,noverlap=size-128)
    hz,t,z=signal.stft(x,**args)
    a=signal.stft(orig,**args)[2]
    power=np.abs(z)**2
    eps=1e-18
    residual=np.abs(z-a)**2
    # Per-bin robust floor, with time dependence estimated from adjacent frequencies.
    floor=np.quantile(ndimage.gaussian_filter(power,(.7,2)),.12,axis=1)[:,None]
    rel=ndimage.uniform_filter1d(np.minimum(power/(floor+eps),8),11,axis=0)
    level=ndimage.percentile_filter(rel,20,size=(1,37))
    noise=(floor+eps)*np.clip(level/(np.median(level,axis=1)[:,None]+eps),.4,3)
    gain=np.clip(abs(a)/(abs(z)+1e-12),.005,1)
    harm=pitch_protection(z,gain,noise,sr,size)
    whitened=np.log1p(power/noise)
    flux=np.maximum(np.diff(whitened,axis=1,prepend=whitened[:,:1]),0)
    onset=np.clip(ndimage.gaussian_filter(flux,(4,1)),0,1)*np.minimum(hz[:,None]/1500,1)
    speech=ndimage.gaussian_filter(abs(a)**2,(.8,1))+(harm+.5*onset)*np.maximum(power-noise,0)
    mask=speech/(speech+noise)
    if mode in ("power","soft","protect"):
        alt=z*(np.sqrt(mask) if mode=="soft" else mask)
        if mode=="protect": alt=a+(harm+onset).clip(0,1)*(alt-a)
    elif mode in ("cov","cov_soft","cov_power","lag6","lag12","lag24","cov_cal","lag6_cal","combo","strong","subband"):
        # Interframe complex prediction of coherent noise. Fit only when first-pass
        # speech evidence is low, pooling frames across the same utterance.
        lag={"lag6":6,"lag12":12,"lag24":24,"lag6_cal":6}.get(mode,1)
        pad=np.pad(z,((0,0),(2*lag,2*lag)),mode="edge")
        v=np.stack([pad[:,i*lag:i*lag+z.shape[1]] for i in (0,1,3,4)],axis=-1)
        w=1/(1+speech/noise)**2
        cov=np.einsum('fti,ftj,ft->fij',v.conj(),v,w,optimize=True)
        corr=np.einsum('fti,ft,ft->fi',v.conj(),z,w,optimize=True)
        diag=np.maximum(cov.trace(axis1=1,axis2=2).real/4,eps)
        coeff=np.linalg.solve(cov+.1*diag[:,None,None]*np.eye(4),corr[...,None])[...,0]
        pred=np.einsum('fti,fi->ft',v,coeff)
        innovation=z-pred*(1-.75*harm)
        alt=innovation*(np.sqrt(mask) if mode=="cov_power" else gain)
    elif mode in ("contrast","activity","band"):
        # Estimate the long-term recoverable speech bandwidth from modulation,
        # and framewise speech support from robust frequency neighborhoods.
        env=ndimage.gaussian_filter(abs(a)**2,(2,2))
        quiet=np.quantile(env,.18,axis=1)[:,None]+eps
        loud=np.quantile(env,.85,axis=1)[:,None]+eps
        contrast=np.clip(np.log(loud/quiet)/5,0,1)
        active=np.clip((np.log((env+eps)/quiet)-.4)/2.5,0,1)
        if mode=="contrast": alt=a*contrast
        elif mode=="band": alt=a*np.sqrt(contrast*active)
        else: alt=a*active
    elif mode.startswith("eq_") or mode in ("occupancy","floor_ratio","modulation","hybrid_eq"):
        env=ndimage.gaussian_filter(power,(1,2))
        lo,hi=np.quantile(env,[.12,.90],axis=1)
        speech_peak=np.quantile(ndimage.gaussian_filter(abs(a)**2,(1,2)),.90,axis=1)
        modulation=np.maximum(hi-lo,eps)/(lo+eps)
        quality=speech_peak/(lo+eps)
        weight=np.sqrt(quality/(quality+3))
        if mode=="modulation": weight=modulation/(modulation+6)
        if mode=="hybrid_eq": weight=np.sqrt(weight*modulation/(modulation+6))
        if mode=="occupancy": weight=np.sqrt(np.mean(env>3*lo[:,None],axis=1))
        if mode.startswith("eq_"):
            weight=np.sqrt(weight*modulation/(modulation+6))
            exponent={"eq_long":1,"eq_deep":2,"eq_rebuild":2,"eq_cut":4,"eq_restore":2,"eq_envelope":3,"eq_rescue":3,"eq_shape":4,"eq_erb":3,"eq_anchor":2,"eq_bass":2,"eq_voice":2}[mode]
            g=np.maximum(weight**exponent,.02)[:,None]
            if mode=="eq_rebuild":
                filtered=signal.istft(z*g,**args)[1][:len(x)]
                return baseline(filtered,sr,replace(cfg,structure="cross_low"),estimator=estimator,blend=blend)
            if mode in ("eq_envelope","eq_rescue","eq_erb","eq_shape"):
                # Preserve regional spectral color while suppressing low-contrast bins.
                sm=ndimage.gaussian_filter1d(g[:,0],16)
                if mode=="eq_shape": g=np.clip(g/np.maximum(sm[:,None],.1)**.8,.02,1.15)
                elif mode=="eq_envelope": g=np.clip(g/np.maximum(sm[:,None],.1)**.45,.02,1.1)
                elif mode=="eq_rescue":
                    reliable=np.clip((abs(a)**2/(lo[:,None]+eps)-3)/15,0,1)
                    g=g+.65*(1-g)*ndimage.gaussian_filter(reliable,(1,1))
                else:
                    lower=1/(1+(hz/650)**4)
                    g=g+(1-g)*lower[:,None]*.65
            alt=a*g
            if mode in ("eq_anchor","eq_bass","eq_voice"):
                # Preserve reliably voiced low-frequency contributions in loud frames.
                low_band=1/(1+(hz[:,None]/(450 if mode=="eq_voice" else 800))**4)
                clean_ratio=ndimage.gaussian_filter(abs(a)**2,(1,1))/(lo[:,None]+eps)
                reliable=clean_ratio/(clean_ratio+4)
                gate=low_band*reliable
                if mode=="eq_bass": gate=low_band
                alt+=gate*(a+.7*reliable*(z-a)-alt)
            if mode=="eq_restore":
                alt+=.3*g**4*(z-a)
        else: alt=a*weight[:,None]
    else: raise ValueError(mode)
    if mode in ("combo","strong","subband"):
        env=ndimage.gaussian_filter(abs(a)**2,(2,2))
        spread=np.log((np.quantile(env,.85,axis=1)+eps)/(np.quantile(env,.18,axis=1)+eps))
        weight=np.clip(spread/5,.12,1)[:,None]
        alt=(.5*alt+.5*a)*weight**(1.5 if mode=="strong" else 1)
        if mode=="subband": alt=alt+(.35*a)*(1-weight)
    y=signal.istft(alt,**args)[1][:len(x)]
    fraction=.25 if mode in ("cov_soft","protect") else .5
    if mode in ("occupancy","floor_ratio","modulation","hybrid_eq"): fraction=.9
    if mode.startswith("eq_"): fraction=.98
    out=orig+fraction*(y-orig)
    if mode.startswith("eq_") or mode in ("cov_cal","lag6_cal","combo","strong","subband","occupancy","floor_ratio","modulation","hybrid_eq"):
        # Match first-pass RMS after decorrelation; only noisy estimates used.
        out*=np.clip(np.linalg.norm(orig)/max(np.linalg.norm(out),eps),.8,2.)
    return out
