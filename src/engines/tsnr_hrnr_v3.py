"""Traditional TSNR+HRNR speech enhancement, Wiener baseline-compatible.
No clean/reference signal is accepted by the enhancement function.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from scipy import signal, ndimage

@dataclass(frozen=True)
class Config:
    n_fft: int = 1024
    hop: int = 256
    noise_fraction: float = 0.2
    noise_scale: float = 1.0
    dd_alpha: float = 0.94
    floor: float = 0.10
    dd_mix: float = 0.40
    harmonic_weight: float = 2.0
    harmonic_drive: float = 0.8
    # Added after v2: gain smoothing to suppress musical noise
    smooth_time: float = 0.0
    smooth_freq: float = 0.0
    # Added after v2: speech-gated adaptation for changing noise
    adapt: float = 0.0
    adapt_gate: float = 1.6
    # Added after v2: soft blend with unprocessed STFT, optional preservation
    dry_mix: float = 0.0


def enhance(x: np.ndarray, sr: int, cfg: Config = Config()) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError('enhance() expects a mono 1-D signal')
    if len(x) < 64 or np.max(np.abs(x)) < 1e-12:
        return x.copy()
    nfft = min(cfg.n_fft, len(x))
    hop = min(cfg.hop, nfft // 2)
    _, times, z = signal.stft(x, fs=sr, window='hann', nperseg=nfft, noverlap=nfft-hop,
                              boundary='zeros', padded=True)
    power = np.abs(z)**2
    centers = np.rint(times*sr).astype(int)
    full = np.flatnonzero((centers >= nfft//2) & (centers + nfft//2 <= len(x)))
    if not len(full): full = np.arange(z.shape[1])
    energies = power[:,full].mean(axis=0)
    k = max(1, int(np.ceil(cfg.noise_fraction*len(full))))
    selected = full[np.argsort(energies, kind='stable')[:k]]
    noise = np.maximum(ndimage.uniform_filter1d(power[:,selected].mean(axis=1),
                      size=3, mode='nearest'), 1e-18) * cfg.noise_scale
    # First stage: decision-directed Wiener estimate.
    gain_dd = np.empty_like(power)
    gamma = np.empty_like(power)
    prev = np.zeros_like(noise)
    noise_frame = noise.copy()
    for t in range(z.shape[1]):
        g = power[:,t]/np.maximum(noise_frame, 1e-18)
        gamma[:,t] = g
        inst = np.maximum(g-1, 0)
        xi = inst if t == 0 else cfg.dd_alpha*prev/np.maximum(noise_frame,1e-18)+(1-cfg.dd_alpha)*inst
        w = np.clip(xi/(1+xi), cfg.floor,1.0)
        gain_dd[:,t] = w
        prev = w*w*power[:,t]
        if cfg.adapt > 0:
            # A bin is updated only if its energy is compatible with noise.
            is_noise = (g < cfg.adapt_gate)
            noise_frame = np.where(is_noise, (1-cfg.adapt)*noise_frame+cfg.adapt*power[:,t], noise_frame)
            # Prevent extreme drift from the robust initial estimate.
            noise_frame = np.maximum(noise * 0.15, np.minimum(noise_frame, noise * 5.0))
    # Second stage: current-frame TSNR estimate without previous-frame delay.
    xi_tsnr = gamma * gain_dd**2
    if cfg.harmonic_weight > 0:
        g0 = np.clip(xi_tsnr/(1+xi_tsnr), cfg.floor,1.0)
        _, prelim = signal.istft(g0*z, fs=sr, window='hann', nperseg=nfft,
                                 noverlap=nfft-hop, boundary=True)
        prelim = prelim[:len(x)]
        rms = max(float(np.sqrt(np.mean(prelim**2))),1e-9)
        harmonic = np.tanh(cfg.harmonic_drive*prelim/rms)*(rms/cfg.harmonic_drive)
        _,_,zh = signal.stft(harmonic, fs=sr, window='hann', nperseg=nfft,
                             noverlap=nfft-hop,boundary='zeros',padded=True)
        regenerated = np.minimum(np.abs(zh)**2, power)
        # Original HRNR-inspired gain correction; bounded harmonic energy.
        xi_tsnr = xi_tsnr + cfg.harmonic_weight*regenerated / (power/np.maximum(gamma,1e-15) + 1e-18)
    gain_tsnr = np.clip(xi_tsnr/(1+xi_tsnr),cfg.floor,1.0)
    gain = cfg.dd_mix*gain_dd+(1-cfg.dd_mix)*gain_tsnr
    if cfg.smooth_freq > 0 or cfg.smooth_time > 0:
        gain = ndimage.gaussian_filter(gain, sigma=(cfg.smooth_freq,cfg.smooth_time),mode='nearest')
    if cfg.dry_mix:
        gain = (1-cfg.dry_mix)*gain+cfg.dry_mix
    _,y = signal.istft(gain*z,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,boundary=True)
    y = y[:len(x)]
    if len(y) < len(x): y = np.pad(y, (0, len(x)-len(y)))
    return y


def process_file(src: Path, dst: Path, cfg: Config = Config()):
    x, sr = sf.read(src, always_2d=True)
    result = np.column_stack([enhance(x[:,i],sr,cfg) for i in range(x.shape[1])])
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Explicitly preserve the source file's metadata where possible.
    info = sf.info(src)
    subtype = info.subtype if info.subtype in ('PCM_16','PCM_24','PCM_32','FLOAT','DOUBLE') else 'PCM_16'
    sf.write(dst, np.clip(result,-1,1), sr, subtype=subtype)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='input'); p.add_argument('--output',default='output')
    for name in Config.__dataclass_fields__:
        p.add_argument('--'+name.replace('_','-'), type=int if name in ('n_fft', 'hop') else float, default=None)
    args=p.parse_args()
    values={name:getattr(args,name) for name in Config.__dataclass_fields__ if getattr(args,name) is not None}
    cfg=Config(**values)
    files=sorted(f for f in Path(args.input).iterdir() if f.is_file() and f.suffix.lower()=='.wav') if Path(args.input).is_dir() else []
    if not files: raise SystemExit(f'No .wav files found in {args.input}')
    for f in files: process_file(f,Path(args.output)/f.name,cfg)
    print('Enhanced',len(files),'WAV files')

if __name__=='__main__': main()
