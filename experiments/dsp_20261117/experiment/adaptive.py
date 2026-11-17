"""Blind PSD reconstruction from low-speech frames, with optional local tracking."""
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter


def estimate(power, initial, full, mode):
    eps = 1e-18
    core = power[:, full]
    # Whiten first so high-energy low-frequency bins do not dominate frame ranking.
    white = np.log1p(core / np.maximum(initial[:, None], eps))
    activity = uniform_filter1d(white, 21, axis=0, mode="nearest")
    activity += white.mean(axis=0)[None, :]
    count = max(1, len(full) // 5)
    indices = np.argpartition(activity, count-1, axis=1)[:, :count]
    quiet = np.take_along_axis(core, indices, axis=1).mean(axis=1)
    # Keep native spectral peaks. Detect stationary lines against adjacent bins.
    persistent = np.quantile(core, .25, axis=1)
    contrast = persistent / np.maximum(median_filter(persistent, 13), eps)
    line = np.clip(np.log2(np.maximum(contrast, 1)) / 3, 0, 1)
    if mode == "line":
        psd = initial * (1-line) + np.maximum(initial, quiet) * line
        return psd[:, None]
    psd = np.maximum(quiet, eps)
    if mode == "quiet":
        return psd[:, None]
    # Relative band energy tracked by a lower order statistic over ~0.5 s.
    from scipy.ndimage import percentile_filter
    ratio = power / psd[:, None]
    bands = np.arange(0, len(psd), 12)
    trajectory = np.add.reduceat(np.minimum(ratio, 8), bands, axis=0)
    trajectory /= np.diff(np.r_[bands,len(psd)])[:,None]
    local = percentile_filter(trajectory, 20, size=(1,31), mode="nearest")
    change = local / np.maximum(np.median(local[:,full],axis=1)[:,None], .05)
    correction = np.repeat(np.clip(change,.55,3),12,axis=0)[:len(psd)]
    return psd[:,None] * gaussian_filter1d(correction, 3, axis=0)**.65


def combine(x, short, long, sr, hop):
    from scipy import signal
    params = dict(fs=sr,nperseg=1024,noverlap=1024-hop)
    _,_,z = signal.stft(x,**params)
    _,_,a = signal.stft(short,**params)
    _,_,b = signal.stft(long,**params)
    mag = np.abs(z)
    floor = np.quantile(mag,.25,axis=1)[:,None]
    flux = np.maximum(np.diff(np.log1p(mag / np.maximum(floor,1e-12)),prepend=0,axis=1),0)
    onsets = gaussian_filter1d(flux,2,axis=0)
    weight = .15+.65*np.exp(-3*onsets)
    _,y=signal.istft(a+weight*(b-a),**params)
    return y[:len(x)]


def context_gain(power, noise):
    """Noncausal speech support: smooth whitened local power BEFORE gain estimation."""
    from scipy.ndimage import gaussian_filter
    ratio = power / np.maximum(noise, 1e-18)
    # Short center window preserves attacks, wider support rejects isolated bins.
    short = gaussian_filter(np.minimum(ratio, 100), (1.3,1.3), mode="nearest")
    broad = gaussian_filter(np.minimum(ratio, 40), (3,3), mode="nearest")
    evidence = np.sqrt(short*broad)
    prior = np.maximum(short-.85,0)
    wiener = prior / (1+prior)
    probability = 1/(1+np.exp(np.clip(3*(1.55-evidence),-40,40)))
    return .012+.988*probability*wiener


def refine(gain, power, noise, raw, mode):
    """Compress residual masks in speech gaps and restore strong local ridges."""
    from scipy.ndimage import gaussian_filter, maximum_filter1d
    ratio = power / np.maximum(noise,1e-18)
    support = gaussian_filter(np.minimum(ratio,16), (2,1))
    reliable = np.clip((support-2)/8,0,1)
    # Limit ridge restoration to strong bins, retaining baseline in uncertain regions.
    voice = reliable*np.maximum(raw-gain,0)
    if mode=="ridge":
        return np.minimum(1,gain+.85*voice)
    # Map a soft probability through odds to suppress weak residuals while
    # preserving bins with substantial speech occupancy.
    odds = np.maximum(gain,1e-5)/np.maximum(1-gain,1e-5)
    sharpened = 1/(1+(1/odds)**1.65)
    if mode=="odds":
        return sharpened
    return np.clip(sharpened+.8*voice,.005,1)


def auditory_gain(gain, power, sr, mode):
    """Bark-band envelopes preserve broadband phonemes and pool random gain fluctuations."""
    hz=np.linspace(0,sr/2,len(gain))
    bark=13*np.arctan(.00076*hz)+3.5*np.arctan((hz/7500)**2)
    centers=np.arange(0,bark[-1]+1,.65)
    weights=np.maximum(1-abs(bark[:,None]-centers)/.9,0)
    # Mean gain weighted weakly by energy, to avoid letting the loudest bin set a band.
    mass=np.maximum(power,1e-18)**.25
    bands=(weights.T@(gain*mass))/np.maximum(weights.T@mass,1e-15)
    pooled=(weights@bands)/np.maximum(weights.sum(1)[:,None],1e-12)
    if mode=="bark":
        return .35*gain+.65*pooled
    if mode=="envelope":
        return np.maximum(gain,.85*pooled)
    return .5*gain+.5*np.sqrt(np.maximum(gain*pooled,0))


def compensate(gain,power,noise):
    """Fit one amplitude correction per branch using an estimated clean cross-power."""
    clean_power=np.maximum(power-noise,0)
    alpha=np.sum(gain*clean_power)/max(np.sum(gain**2*power),1e-18)
    return gain*np.clip(alpha,1,1.6)


def notch_lines(x,sr):
    """Estimate long-lived frequency and phase from disjoint blocks, then subtract."""
    from scipy.signal import find_peaks
    from scipy.fft import next_fast_len, rfft
    n=len(x)
    if n<4096: return x
    nfft=next_fast_len(2*n)
    mag=np.abs(rfft(x*np.hanning(n),nfft))
    peaks,_=find_peaks(mag,distance=4)
    floor=median_filter(mag,size=35)
    peaks=peaks[(mag[peaks]>9*np.maximum(floor[peaks],1e-12)) & (peaks>3)]
    t=np.arange(n)/sr
    y=x.copy()
    for k in peaks[np.argsort(mag[peaks])[-16:]]:
        q=np.log(np.maximum(mag[k-1:k+2],1e-18))
        offset=.5*(q[0]-q[2]) / min(q[0]-2*q[1]+q[2],-1e-12)
        carrier=np.exp(2j*np.pi*(k+np.clip(offset,-.5,.5))*sr/nfft*t)
        blocks=np.array_split(x*carrier.conj(),10)
        coeff=np.array([2*np.mean(b) for b in blocks])
        center=np.median(coeff.real)+1j*np.median(coeff.imag)
        coherent=np.median(abs(coeff-center))/max(abs(center),1e-15)
        if coherent<.65:
            y-=np.real(center*carrier)*(1-(coherent/.65)**2)
    return y


def relative_noise(power, initial, full):
    """Estimate relative local noise changes while retaining the file-level PSD."""
    from scipy.ndimage import percentile_filter
    ratio=np.minimum(power/np.maximum(initial[:,None],1e-18),8)
    bands=np.arange(0,len(initial),16)
    level=np.add.reduceat(ratio,bands,axis=0)/np.diff(np.r_[bands,len(initial)])[:,None]
    local=percentile_filter(level,20,size=(1,35),mode="nearest")
    ref=np.median(local[:,full],axis=1,keepdims=True)
    change=np.repeat(np.clip(local/np.maximum(ref,.05),.5,4),16,axis=0)[:len(initial)]
    return initial[:,None]*gaussian_filter1d(change,4,axis=0)**.5


def band_mask(gain,power,sr,mode):
    hz=np.linspace(0,sr/2,len(gain))
    bark=13*np.arctan(.00076*hz)+3.5*np.arctan((hz/7500)**2)
    centers=np.arange(0,bark[-1]+1,.65)
    weights=np.maximum(1-abs(bark[:,None]-centers)/1.2,0)
    average=(weights.T@np.log(np.maximum(gain,.005)))/weights.sum(0)[:,None].clip(1e-12)
    pooled=np.exp((weights@average)/weights.sum(1)[:,None].clip(1e-12))
    if mode=="band_log": return .45*gain+.55*pooled
    # Preserve original narrow noise nulls while smoothing isolated speech peaks.
    return np.sqrt(gain*np.minimum(gain, pooled*1.6))


def rebalance(gain,power,noise):
    """Estimate amplitude calibration per frame, stabilised by symmetric energy pooling."""
    clean=np.maximum(power-noise,0)
    numerator=gaussian_filter1d(np.sum(gain*clean,axis=0),3)
    denominator=gaussian_filter1d(np.sum(gain*gain*power,axis=0),3)
    return gain*np.clip(numerator/np.maximum(denominator,1e-18),1,2.5)


def spectral_balance(gain,power,noise,sr):
    """Calibrate frequency response from long-term cross-power on a coarse Bark scale."""
    hz=np.linspace(0,sr/2,len(gain))
    bark=13*np.arctan(.00076*hz)+3.5*np.arctan((hz/7500)**2)
    centers=np.arange(0,bark[-1]+2,1.)
    weights=np.maximum(1-abs(bark[:,None]-centers)/3.,0)
    num=weights.T@np.sum(gain*np.maximum(power-noise,0),axis=1)
    den=weights.T@np.sum(gain*gain*power,axis=1)
    alpha=weights@np.clip(num/np.maximum(den,1e-18),1,3)/weights.sum(1)
    return gain*alpha[:,None]


def pitch_protection(z, gain, noise, sr, nfft):
    """Estimate pitch from the first-pass clean spectrum and detect harmonic bins."""
    power=np.abs(z)**2
    freqs=np.fft.rfftfreq(nfft,1/sr)
    excitation=power*gain**2
    excitation[(freqs<65)|(freqs>3200)]=0
    ac=np.fft.irfft(excitation,axis=0,n=nfft)
    lo,hi=int(sr/400),int(sr/65)
    if hi>=nfft//2: return np.zeros_like(gain)
    lag=lo+np.argmax(ac[lo:hi+1],axis=0)
    confidence=ac[lag,np.arange(len(lag))]/np.maximum(ac[0],1e-18)
    f0=sr/lag
    distance=abs((freqs[:,None]+.5*f0)%f0-.5*f0)
    width=sr/nfft+freqs[:,None]*.009
    harmonic=np.exp(-.5*(distance/width)**2)
    return harmonic*np.clip((confidence-.32)/.40,0,1)*np.clip((power/noise-1)/7,0,1)


def persistent_notch(x,sr,mode):
    """Long-window spectral line suppression gated by persistence and local prominence."""
    from scipy import signal
    from scipy.ndimage import maximum_filter1d
    nfft=min(len(x),8192)
    if nfft<512: return x.copy()
    hop=nfft//8
    cfg=dict(fs=sr,nperseg=nfft,noverlap=nfft-hop)
    _,_,z=signal.stft(x,**cfg)
    mag=np.abs(z)
    floor=median_filter(mag,size=(13,1),mode="nearest")
    excess=mag/np.maximum(floor,1e-12)
    core=excess[:,4:-4] if z.shape[1]>12 else excess
    # Accept movement within about 4 Hz; individual voiced partials usually move farther.
    occupancy=np.mean(maximum_filter1d(core,5,axis=0)>3.2,axis=1)
    reliable=np.clip((occupancy-.42)/.42,0,1)[:,None]
    if mode=="strict":
        reliable=np.clip((occupancy-.65)/.25,0,1)[:,None]
    loss=reliable*np.clip(1-1.3/np.maximum(excess,1),0,.95)
    _,y=signal.istft(z*(1-loss),**cfg)
    return y[:len(x)]


def drifting_notch(x,sr,mode):
    """Follow high-contrast ridges that survive a one-second local persistence test."""
    from scipy import signal
    from scipy.ndimage import maximum_filter1d
    n=min(len(x),4096)
    if n<512: return x.copy()
    hop=n//8
    args=dict(fs=sr,nperseg=n,noverlap=n-hop)
    _,_,z=signal.stft(x,**args)
    mag=np.abs(z)
    background=median_filter(mag,size=(15,1),mode="nearest")
    contrast=mag/np.maximum(background,1e-15)
    hit=maximum_filter1d(contrast,5,axis=0)>3.5
    local=uniform_filter1d(hit.astype(float),size=33,axis=1,mode="nearest")
    global_rate=hit.mean(axis=1)[:,None]
    trust=np.clip((local-.62)/.32,0,1)*np.clip((global_rate-.18)/.32,0,1)
    if mode=="phase":
        step=z[:,1:]*z[:,:-1].conj()
        unit=step/np.maximum(abs(step),1e-18)
        coherence=abs(uniform_filter1d(unit,9,axis=1,mode="nearest"))
        trust*=np.pad(np.clip((coherence-.65)/.3,0,1),((0,0),(1,0)),mode="edge")
    residual=1-np.clip(1-1.25/np.maximum(contrast,1),0,.96)*trust
    _,y=signal.istft(z*residual,**args)
    return y[:len(x)]


def cepstral_mask(gain,sr,style):
    """Keep fast envelope changes and pool fine ripple across neighboring frames."""
    n=(len(gain)-1)*2
    cep=np.fft.irfft(np.log(np.maximum(gain,.004)),axis=0)
    quefrency=np.minimum(np.arange(n),n-np.arange(n))/sr
    slow=gaussian_filter1d(cep,3.0,axis=1,mode="nearest")
    blend=1-np.exp(-(quefrency/.0018)**2)
    refined=cep+(slow-cep)*blend[:,None]
    if style=="lifter":
        refined*=1-.4*blend[:,None]
    return np.clip(np.exp(np.fft.rfft(refined,axis=0).real),.002,1)


def reconstruct(x,initial,sr,style):
    """Iterative consistent speech/noise phase estimation with fixed magnitudes."""
    from scipy import signal
    params=dict(fs=sr,nperseg=512,noverlap=384)
    _,_,obs=signal.stft(x,**params)
    _,_,speech=signal.stft(initial,**params)
    mag=np.abs(speech)
    noise_mag=np.abs(obs-speech)
    # Initial residual phase follows the mixture. Iteratively impose mixture consistency.
    power=mag**2
    share=power/np.maximum(power+noise_mag**2,1e-18)
    yy=initial
    for _ in range(3):
        noise=obs-speech
        noise=noise_mag*noise/np.maximum(abs(noise),1e-18)
        speech=mag*speech/np.maximum(abs(speech),1e-18)
        speech+=share*(obs-noise-speech)
        _,yy=signal.istft(speech,**params)
        _,_,speech=signal.stft(yy[:len(x)],**params)
    if style=="shrink":
        # Re-estimate residual size from local covariance across neighboring TF bins.
        from scipy.ndimage import gaussian_filter
        leak=gaussian_filter((speech*(obs-speech).conj()).real,(2,2))
        residual=gaussian_filter(abs(obs-speech)**2,(2,2))
        direction=np.clip(leak/np.maximum(residual,1e-18),0,.4)
        _,yy=signal.istft(speech-direction*(obs-speech),**params)
    return yy[:len(x)]


def transient_mix(x, short, long, sr, mode):
    """Use short FFT around onsets and downweight it around stationary noise lines."""
    from scipy import signal
    cfg=dict(fs=sr,nperseg=512,noverlap=384)
    hz,_,obs=signal.stft(x,**cfg)
    _,_,a=signal.stft(short,**cfg)
    _,_,b=signal.stft(long,**cfg)
    mag=abs(obs)
    floor=np.maximum(np.quantile(mag,.20,axis=1),1e-12)
    noise_line=np.clip(np.log2(floor/np.maximum(median_filter(floor,9),1e-12)),0,3)/3
    level=np.log1p(mag/floor[:,None])
    flow=np.abs(np.diff(level,axis=1,prepend=level[:,:1]))
    transient=gaussian_filter1d(flow,3,axis=0)
    w=.32+.40*np.clip(transient,0,1)+.18*hz[:,None]/(hz[:,None]+900)-.25*noise_line[:,None]
    if mode=="high":
        w=.25+.60*hz[:,None]/(hz[:,None]+900)-.2*noise_line[:,None]
    _,y=signal.istft(b+w*(a-b),**cfg)
    return y[:len(x)]
