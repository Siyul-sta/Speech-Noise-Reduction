"""
enhance_gain.py  --  고전 신호처리 전용 음성 향상 이득(gain) 엔진
=====================================================================
딥러닝/머신러닝/학습 모델을 전혀 쓰지 않습니다 (numpy + scipy만 사용).
모든 임계값은 사람이 정한 규칙이며, 학습되는 파라미터가 없습니다.

구조 (STFT 1회 -> 이득 계산 -> iSTFT 1회, 신호를 직렬로 여러 번 처리하지 않음)

  noisy 파워 P (freq x time)
   |-- 잡음 PSD 초기화: 주파수 bin별 하위 분위수 (Stahl 2000 계열, 카이제곱 편향 보정)
   |-- 후보 A : MMSE-SPP 잡음 추적, 양방향 (Gerkmann & Hendriks 2012)
   |            + TSNR 사전 SNR (Plapous, Marro & Scalart 2006)
   |            + OM-LSA 이득 (Cohen 2002; Ephraim & Malah 1985 의 LSA)
   |-- 후보 B : IMCRA (Cohen 2003) + OM-LSA (Cohen & Berdugo 2001)
   |-- 결합   : 이득 영역 가중 기하평균 (가중치/하한은 noisy에서 계산한 특성으로만 결정)
   `-- 후처리 : (선택) percussive 보조 이득 -> SPP 적응형 이득 평활 -> SNR 적응형 하한

주의
 * HRNR은 시간영역 재생(위상 필요)이 필요해서 'magnitude/power만 받는 gain 엔진' 인터페이스에는 넣지 않았습니다.
 * IMCRA 상수(B_min, gamma0, gamma1, zeta0, beta 등)는 제가 논문을 기억에 의존해 옮긴 값입니다.
   원문과 대조 확인하고, diagnostics.py 의 PSD 오차 표(부호)로 편향을 점검하세요.
 * 라우팅 임계값(nonstat_thr 등)은 초기 추정치입니다. diagnostics.py --task features 로 분포를 보고 정하세요.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter, uniform_filter1d
from scipy.special import exp1
from scipy.stats import gamma as gamma_dist

EPS = 1e-18


# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GainConfig:
    # --- STFT (제공 노트북의 Config와 동일) ---
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 128

    # --- 잡음 PSD 초기화 ---
    init_method: str = "bin_quantile"      # "bin_quantile" | "frame_lowest"(노트북 방식)
    init_quantile: float = 0.15            # bin별 시간축 분위수
    init_dof: float = 3.0                  # 평활된 주기도표의 유효 자유도(편향 보정용, 튜닝 대상)
    init_frame_fraction: float = 0.20      # frame_lowest 일 때 하위 프레임 비율
    init_scale_db: float = 0.0             # 초기값 추가 보정(dB)
    init_adaptive_dof: bool = False        # bin별 안정도(분위수 비)로 자유도 추정 -> 톤성 bin은 보정 약화
    init_dof_max: float = 60.0
    init_shrink_beta: float = 0.5          # 시간축 동적 범위가 큰(음성 지배) bin의 추정값 축소 지수 (0=끔)

    # --- 공통 ---
    xi_min_db: float = -25.0               # 사전 SNR 하한
    g_min: float = 0.05                    # OM-LSA 음성 부재 시 이득 (-26 dB)
    a_dd: float = 0.90                     # 후보 A decision-directed 평활 (대리 벤치마크에서 0.98보다 우수)

    # --- 후보 A ---
    spp_xi_h1_db: float = 15.0             # SPP 고정 사전 SNR (Gerkmann & Hendriks)
    spp_prior_h1: float = 0.5
    spp_alpha: float = 0.8                 # 잡음 갱신 평활
    spp_bidirectional: bool = True         # 3초 파일: 앞/뒤 양방향 추적 후 결합
    spp_combine: str = "geomean"           # "geomean" | "min" | "forward"
    use_tsnr: bool = False                 # a_dd=0.9에서는 이득이 없어 기본 OFF (대리 벤치마크)
    use_spp_weight: bool = True            # OM-LSA식 SPP 가중

    # --- 후보 B (IMCRA) ---
    imcra_alpha_s: float = 0.9
    imcra_alpha_d: float = 0.85
    imcra_bmin: float = 1.66
    imcra_gamma0: float = 4.6
    imcra_gamma1: float = 3.0
    imcra_zeta0: float = 1.67
    imcra_a_dd: float = 0.92               # 후보 B 전용 decision-directed 계수 (논문 값 0.92)
    imcra_beta: float = 1.0                # 논문은 ~1.47 사용으로 기억. 기본 1.0, PSD 부호 오차로 확인
    imcra_U: int = 8
    imcra_V: int = 15

    # --- 이득 영역 결합 / 라우팅 (noisy 특성만 사용) ---
    w_a: float = 0.5                       # 후보 A 기본 가중치
    routing: bool = True
    nonstat_thr: float = 0.25              # 추적된 잡음 PSD의 시간 변동(log10 표준편차) 임계 (placeholder)
    w_a_nonstat: float = 0.5               # 변동이 큰 파일의 A 가중치 (기본은 변화 없음 = 비활성)
    snr_lo: float = 0.0                    # 추정 SNR 구간 -> 이득 하한 선형 보간
    snr_hi: float = 15.0
    floor_lo: float = 0.05                 # (대리 벤치마크: SNR 적응 하한은 이득 없음 -> 상수 0.05)
    floor_hi: float = 0.05

    # --- 후처리 ---
    use_gain_smoothing: bool = True
    smooth_min: float = 0.2                # 발화(SPP~1): 약한 평활
    smooth_max: float = 0.8                # 잡음(SPP~0): 강한 평활
    use_percussive: bool = False           # 충격성 보조 이득 (전체 1.5%라 기본 OFF)
    perc_time_k: int = 17
    perc_freq_k: int = 17
    perc_floor: float = 0.3
    perc_fmax: float = 2500.0

    def __post_init__(self):
        if self.sample_rate != 16000:
            raise ValueError("16 kHz 전용입니다.")
        if not 0 < self.hop <= self.n_fft:
            raise ValueError("0 < hop <= n_fft 이어야 합니다.")
        if not signal.check_NOLA("hann", self.n_fft, self.n_fft - self.hop):
            raise ValueError("STFT 창/hop이 NOLA를 만족하지 않습니다.")
        if self.init_method not in ("bin_quantile", "frame_lowest"):
            raise ValueError("init_method 오류")
        if self.spp_combine not in ("geomean", "min", "forward"):
            raise ValueError("spp_combine 오류")
        if not 0 < self.g_min <= 1 or not 0 <= self.a_dd < 1:
            raise ValueError("g_min / a_dd 범위 오류")


GCFG = GainConfig()


# ----------------------------------------------------------------------------
# STFT 도우미 (노트북과 동일한 설정; 팀에서 제공하는 STFT를 쓰면 무시해도 됨)
# ----------------------------------------------------------------------------
def waveform(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("비어 있지 않은 유한값 mono 파형이어야 합니다.")
    return x


def stft_audio(x, cfg: GainConfig = GCFG):
    x = waveform(x)
    if x.size < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - x.size))
    return signal.stft(x, fs=cfg.sample_rate, window="hann", nperseg=cfg.n_fft,
                       noverlap=cfg.n_fft - cfg.hop, nfft=cfg.n_fft,
                       boundary="zeros", padded=True)


def inverse_stft(z, length, cfg: GainConfig = GCFG):
    _, y = signal.istft(z, fs=cfg.sample_rate, window="hann", nperseg=cfg.n_fft,
                        noverlap=cfg.n_fft - cfg.hop, nfft=cfg.n_fft,
                        input_onesided=True, boundary=True)
    if len(y) < length:
        raise ValueError("iSTFT 길이 부족")
    return waveform(y[:length])


def _core(T, cfg):
    """zero padding 때문에 인위적으로 작아진 경계 프레임 제외."""
    m = cfg.n_fft // (2 * cfg.hop)
    return slice(m, T - m) if T > 2 * m + 4 else slice(0, T)


# ----------------------------------------------------------------------------
# 1) 잡음 PSD 초기화
# ----------------------------------------------------------------------------
def init_noise_psd(P, cfg: GainConfig = GCFG):
    """파일 전체에서 잡음 PSD 초기값(freq,)을 구한다. clean 접근 없음."""
    core = P[:, _core(P.shape[1], cfg)]
    if cfg.init_method == "frame_lowest":      # 노트북 방식(비교용)
        e = core.mean(axis=0)
        k = max(1, int(np.ceil(cfg.init_frame_fraction * core.shape[1])))
        idx = np.argsort(e, kind="stable")[:k]
        est = uniform_filter1d(core[:, idx].mean(axis=1), 3, mode="nearest")
    else:                                      # bin별 하위 분위수 + 카이제곱 편향 보정
        sm = uniform_filter(core, size=(3, 5), mode="nearest")
        qs = np.quantile(sm, [cfg.init_quantile, 0.4], axis=1)
        q, q40 = qs[0], qs[1]
        if cfg.init_adaptive_dof:
            # 안정적인(톤성) bin은 분위수 비 q40/q15가 1에 가깝다 -> 큰 자유도 -> 보정 ~1
            ks = np.geomspace(cfg.init_dof, cfg.init_dof_max, 40)
            r_tab = gamma_dist.ppf(0.4, a=ks) / gamma_dist.ppf(cfg.init_quantile, a=ks)   # k 증가 시 감소
            r_obs = q40 / np.maximum(q, EPS)
            k_bin = np.interp(r_obs, r_tab[::-1], ks[::-1])      # r_obs 클수록 k=init_dof 로 수렴
            comp = k_bin / gamma_dist.ppf(cfg.init_quantile, a=k_bin)
        else:
            comp = cfg.init_dof / gamma_dist.ppf(cfg.init_quantile, a=cfg.init_dof)
        est = q * comp
        if cfg.init_shrink_beta > 0:
            # 잡음만 있는 bin의 기대 동적 범위(q85/q15) 대비 초과분이 클수록 '음성 지배 bin' -> 추정은 상한일 뿐
            q85 = np.quantile(sm, 0.85, axis=1)
            r0 = gamma_dist.ppf(0.85, a=cfg.init_dof) / gamma_dist.ppf(cfg.init_quantile, a=cfg.init_dof)
            excess = np.maximum(q85 / np.maximum(q, EPS) / r0, 1.0)
            est = np.maximum(est / excess ** cfg.init_shrink_beta, q / comp)
    est = est * 10.0 ** (cfg.init_scale_db / 10.0)
    return np.maximum(est, EPS)


# ----------------------------------------------------------------------------
# 2) 후보 A : MMSE-SPP 잡음 추적 + TSNR + OM-LSA
# ----------------------------------------------------------------------------
def spp_posterior(P, noise, xi_h1_db, prior_h1):
    """P(H1|y): 고정 사전 SNR 가정의 음성 존재 사후확률 (Gerkmann & Hendriks 2012)."""
    xi = 10.0 ** (xi_h1_db / 10.0)
    ratio = (1.0 - prior_h1) / prior_h1
    return 1.0 / (1.0 + ratio * (1.0 + xi) * np.exp(-(P / noise) * xi / (1.0 + xi)))


def _spp_track(P, init, cfg):
    F, T = P.shape
    xi = 10.0 ** (cfg.spp_xi_h1_db / 10.0)
    ratio = (1.0 - cfg.spp_prior_h1) / cfg.spp_prior_h1
    noise = np.empty_like(P)
    prev = init.copy()
    pbar = np.zeros(F)
    for t in range(T):
        y = P[:, t]
        p = 1.0 / (1.0 + ratio * (1.0 + xi) * np.exp(-(y / prev) * xi / (1.0 + xi)))
        pbar = 0.9 * pbar + 0.1 * p
        p = np.where(pbar > 0.99, np.minimum(p, 0.99), p)     # SPP 고착 방지
        est = (1.0 - p) * y + p * prev
        prev = cfg.spp_alpha * prev + (1.0 - cfg.spp_alpha) * est
        noise[:, t] = prev
    return np.maximum(noise, EPS)


def track_noise_spp(P, init, cfg: GainConfig = GCFG):
    fwd = _spp_track(P, init, cfg)
    if not cfg.spp_bidirectional or cfg.spp_combine == "forward":
        return fwd
    bwd = _spp_track(P[:, ::-1], init, cfg)[:, ::-1]
    if cfg.spp_combine == "min":
        return np.minimum(fwd, bwd)
    return np.exp(0.5 * (np.log(fwd) + np.log(bwd)))


def _lsa_gain(xi, gamma, g_min):
    v = np.maximum(xi * gamma / (1.0 + xi), 1e-8)
    return np.clip(xi / (1.0 + xi) * np.exp(0.5 * exp1(v)), g_min, 1.0)


def gain_candidate_a(P, cfg: GainConfig = GCFG, init=None, noise=None):
    """반환: (gain, noise_psd, spp). noise를 넘기면 추적을 건너뛴다(oracle 실험용)."""
    F, T = P.shape
    if init is None:
        init = init_noise_psd(P, cfg)
    if noise is None:
        noise = track_noise_spp(P, init, cfg)
    spp = spp_posterior(P, noise, cfg.spp_xi_h1_db, cfg.spp_prior_h1)
    p_w = spp if cfg.use_spp_weight else np.ones_like(spp)
    xi_min = 10.0 ** (cfg.xi_min_db / 10.0)
    G = np.empty_like(P)
    prev_clean = np.zeros(F)
    for t in range(T):
        y, lam = P[:, t], noise[:, t]
        gam = y / lam
        inst = np.maximum(gam - 1.0, 0.0)
        xi = inst if t == 0 else cfg.a_dd * prev_clean / lam + (1.0 - cfg.a_dd) * inst
        xi = np.maximum(xi, xi_min)
        if cfg.use_tsnr:                       # 2단계: 한 프레임 지연 보정
            g1 = xi / (1.0 + xi)
            xi = np.maximum(g1 ** 2 * y / lam, xi_min)
        g_lsa = _lsa_gain(xi, gam, cfg.g_min)
        g = g_lsa ** p_w[:, t] * cfg.g_min ** (1.0 - p_w[:, t])
        G[:, t] = g
        prev_clean = g ** 2 * y
    return G, noise, spp


# ----------------------------------------------------------------------------
# 3) 후보 B : IMCRA + OM-LSA
# ----------------------------------------------------------------------------
class _SlidingMin:
    """길이 V 서브윈도 U개에 걸친 최소값 추적 (Cohen 2003). 첫 윈도에서는 init 수준을 사용."""

    def __init__(self, start, U, V):
        self.U, self.V, self.start = U, V, start.copy()
        self.cur = np.full_like(start, np.inf)
        self.hist = []
        self.count = 0

    def update(self, s):
        self.count += 1
        self.cur = np.minimum(self.cur, s)
        if self.count % self.V == 0:
            self.hist.append(self.cur)
            if len(self.hist) > self.U - 1:
                self.hist.pop(0)
            self.cur = s.copy()
        m = np.minimum(self.cur, self.start) if not self.hist else self.cur
        for h in self.hist:
            m = np.minimum(m, h)
        return m


def gain_candidate_b(P, cfg: GainConfig = GCFG, init=None):
    """반환: (gain, noise_psd, speech_presence)."""
    F, T = P.shape
    if init is None:
        init = init_noise_psd(P, cfg)
    b = np.array([0.25, 0.5, 0.25])

    def fsmooth(x):
        return np.convolve(np.pad(x, 1, mode="edge"), b, mode="valid")

    bmin = cfg.imcra_bmin
    S, St = init.copy(), init.copy()
    smin_tr = _SlidingMin(init / bmin, cfg.imcra_U, cfg.imcra_V)
    stmin_tr = _SlidingMin(init / bmin, cfg.imcra_U, cfg.imcra_V)
    lam_hat = init.copy()
    lam = cfg.imcra_beta * lam_hat
    xi_min = 10.0 ** (cfg.xi_min_db / 10.0)
    G, noise, pout = np.empty_like(P), np.empty_like(P), np.empty_like(P)
    prev_clean = np.zeros(F)
    for t in range(T):
        y = P[:, t]
        # 1차 반복: 잠정 잡음 구간 지시자
        S = cfg.imcra_alpha_s * S + (1 - cfg.imcra_alpha_s) * fsmooth(y)
        smin = smin_tr.update(S)
        gmin1, zeta1 = y / (bmin * smin), S / (bmin * smin)
        ind = ((gmin1 < cfg.imcra_gamma0) & (zeta1 < cfg.imcra_zeta0)).astype(float)
        num, den = fsmooth(ind * y), fsmooth(ind)
        stf = np.where(den > 1e-12, num / np.maximum(den, 1e-12), St)
        St = cfg.imcra_alpha_s * St + (1 - cfg.imcra_alpha_s) * stf
        # 2차 반복: 음성 부재 확률 q
        stmin = stmin_tr.update(St)
        gmin2, zeta2 = y / (bmin * stmin), S / (bmin * stmin)
        q = np.where((gmin2 <= 1.0) & (zeta2 < cfg.imcra_zeta0), 1.0,
                     np.where((gmin2 < cfg.imcra_gamma1) & (zeta2 < cfg.imcra_zeta0),
                              (cfg.imcra_gamma1 - gmin2) / (cfg.imcra_gamma1 - 1.0), 0.0))
        q = np.clip(q, 0.0, 0.99)
        # 이득 (사전 SNR: decision-directed)
        gam = y / lam
        inst = np.maximum(gam - 1.0, 0.0)
        xi = inst if t == 0 else cfg.imcra_a_dd * prev_clean / lam + (1.0 - cfg.imcra_a_dd) * inst
        xi = np.maximum(xi, xi_min)
        v = np.maximum(xi * gam / (1.0 + xi), 1e-8)
        p = 1.0 / (1.0 + (q / (1.0 - q)) * (1.0 + xi) * np.exp(-v))
        g_lsa = _lsa_gain(xi, gam, cfg.g_min)
        g = g_lsa ** p * cfg.g_min ** (1.0 - p)
        prev_clean = g ** 2 * y
        # 잡음 갱신
        ad = cfg.imcra_alpha_d + (1 - cfg.imcra_alpha_d) * p
        lam_hat = ad * lam_hat + (1 - ad) * y
        lam = cfg.imcra_beta * lam_hat
        G[:, t], noise[:, t], pout[:, t] = g, lam, p
    return G, np.maximum(noise, EPS), pout


# ----------------------------------------------------------------------------
# 4) noisy 전용 특성 / 규칙 기반 라우팅 (학습 없음)
# ----------------------------------------------------------------------------
def noisy_only_features(P, noise, init, cfg: GainConfig = GCFG):
    c = _core(P.shape[1], cfg)
    y, n = P[:, c], noise[:, c]
    sig = np.maximum(y - n, 0.0).sum()
    snr_est_db = 10.0 * np.log10(max(sig, EPS) / max(n.sum(), EPS))
    flat = float(np.exp(np.mean(np.log(init))) / np.mean(init))   # 잡음 PSD 평탄도
    noise_var = float(np.median(np.std(np.log10(n), axis=1)))     # 추적 잡음 PSD의 시간 변동
    return {"snr_est_db": float(snr_est_db), "flat_est": flat, "noise_var": noise_var}


def routing_weights(feats, cfg: GainConfig = GCFG):
    """규칙: (1) 추정 SNR이 높을수록 이득 하한을 높임 (2) 변동 큰 파일은 A/B 가중치 변경 (기본 비활성)."""
    vals = np.array(list(feats.values()), dtype=float)
    if not cfg.routing:
        return cfg.w_a, cfg.g_min
    if not np.isfinite(vals).all():                  # 불확실 -> 후보 A 단독 + 보수적 하한
        return 1.0, cfg.floor_hi
    r = np.clip((feats["snr_est_db"] - cfg.snr_lo) / (cfg.snr_hi - cfg.snr_lo), 0.0, 1.0)
    floor = cfg.floor_lo + (cfg.floor_hi - cfg.floor_lo) * r
    w_a = cfg.w_a_nonstat if feats["noise_var"] > cfg.nonstat_thr else cfg.w_a
    return float(w_a), float(max(floor, cfg.g_min))


# ----------------------------------------------------------------------------
# 5) 후처리
# ----------------------------------------------------------------------------
def adaptive_gain_smoothing(G, p, cfg: GainConfig = GCFG):
    """SPP 낮은(잡음) 구간은 강하게, 높은(발화) 구간은 약하게 시간 평활."""
    s = cfg.smooth_min + (cfg.smooth_max - cfg.smooth_min) * (1.0 - np.clip(p, 0.0, 1.0))
    out = G.copy()
    for t in range(1, G.shape[1]):
        out[:, t] = s[:, t] * out[:, t - 1] + (1.0 - s[:, t]) * G[:, t]
    return out


def percussive_gain(mag, cfg: GainConfig = GCFG):
    """HPSS(Fitzgerald 2010) 변형: 확실한 percussive 성분만 완화. 고주파(마찰음)는 보호."""
    h = median_filter(mag, size=(1, cfg.perc_time_k), mode="nearest")
    pc = median_filter(mag, size=(cfg.perc_freq_k, 1), mode="nearest")
    mask = pc ** 2 / (h ** 2 + pc ** 2 + EPS)
    gain = 1.0 - (1.0 - cfg.perc_floor) * np.clip((mask - 0.5) / 0.5, 0.0, 1.0)
    freqs = np.linspace(0, cfg.sample_rate / 2, mag.shape[0])
    gain[freqs > cfg.perc_fmax] = 1.0
    return gain


# ----------------------------------------------------------------------------
# 6) 공개 API
# ----------------------------------------------------------------------------
def compute_gain(noisy_mag=None, noisy_power=None, cfg: GainConfig = GCFG,
                 mode: str = "ensemble", return_diag: bool = False):
    """
    noisy_mag 또는 noisy_power (freq x time) -> gain_matrix (freq x time, 0~1).
    mode: "ensemble" | "A" | "B"
    """
    if noisy_power is None:
        if noisy_mag is None:
            raise ValueError("noisy_mag 또는 noisy_power가 필요합니다.")
        noisy_power = np.asarray(noisy_mag, dtype=np.float64) ** 2
    P = np.maximum(np.asarray(noisy_power, dtype=np.float64), EPS)
    if not np.isfinite(P).all():
        raise ValueError("입력에 NaN/Inf가 있습니다.")
    if mode not in ("ensemble", "A", "B"):
        raise ValueError("mode 오류")
    F, T = P.shape
    if T < 6:
        return (np.ones_like(P), {}) if return_diag else np.ones_like(P)

    init = init_noise_psd(P, cfg)
    ga = gb = noise_a = noise_b = p_a = p_b = None
    if mode in ("ensemble", "A"):
        ga, noise_a, p_a = gain_candidate_a(P, cfg, init=init)
    if mode in ("ensemble", "B"):
        gb, noise_b, p_b = gain_candidate_b(P, cfg, init=init)

    feats = noisy_only_features(P, noise_a if noise_a is not None else noise_b, init, cfg)
    w_a, floor = routing_weights(feats, cfg)

    if mode == "A":
        G, p_mix = ga, p_a
    elif mode == "B":
        G, p_mix = gb, p_b
    else:
        G = np.exp(w_a * np.log(np.maximum(ga, 1e-6)) + (1 - w_a) * np.log(np.maximum(gb, 1e-6)))
        p_mix = w_a * p_a + (1 - w_a) * p_b

    if cfg.use_percussive:
        G = G * percussive_gain(np.sqrt(P), cfg)
    if cfg.use_gain_smoothing:
        G = adaptive_gain_smoothing(G, p_mix, cfg)
    G = np.clip(G, floor, 1.0)

    if return_diag:
        return G, {"features": feats, "w_a": w_a, "floor": floor, "init": init,
                   "noise_a": noise_a, "noise_b": noise_b, "p_a": p_a, "p_b": p_b,
                   "gain_a": ga, "gain_b": gb}
    return G


def enhance_wave(x, cfg: GainConfig = GCFG, mode: str = "ensemble"):
    """파형 -> 파형 (입력 위상 사용, 길이 유지, 정규화 없음). mode='noisy'면 항등."""
    x = waveform(x)
    _, _, Z = stft_audio(x, cfg)
    if mode == "noisy":
        G = np.ones(Z.shape)
    else:
        G = compute_gain(noisy_power=np.abs(Z) ** 2, cfg=cfg, mode=mode)
    y = inverse_stft(G * Z, len(x), cfg)
    if not np.isfinite(y).all():
        raise FloatingPointError("출력에 NaN/Inf")
    return y


def process_folder(input_dir="input", output_dir="output", cfg: GainConfig = GCFG,
                   mode: str = "ensemble", subtype: str = "FLOAT"):
    """input/ 의 모든 wav를 처리해 output/ 에 같은 파일명으로 저장."""
    import soundfile as sf
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"wav 파일이 없습니다: {in_dir.resolve()}")
    for path in files:
        x, sr = sf.read(path, dtype="float64")
        if sr != cfg.sample_rate:
            raise ValueError(f"{path.name}: {sr} Hz (16000 Hz 필요)")
        y = enhance_wave(x, cfg, mode)
        if subtype != "FLOAT":
            y = np.clip(y, -1.0, 1.0)
        sf.write(out_dir / path.name, y, sr, subtype=subtype)
    return len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="input/ 폴더의 음성을 처리해 output/ 에 저장")
    ap.add_argument("--input", default="input")
    ap.add_argument("--output", default="output")
    ap.add_argument("--mode", default="ensemble", choices=["ensemble", "A", "B"])
    ap.add_argument("--subtype", default="FLOAT", choices=["FLOAT", "PCM_16"])
    a = ap.parse_args()
    print(f"{process_folder(a.input, a.output, GCFG, a.mode, a.subtype)}개 파일 처리 완료")
