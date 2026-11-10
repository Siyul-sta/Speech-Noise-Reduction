# -*- coding: utf-8 -*-
"""
speech_gain_engine.py
=====================
주파수 영역 '필터 이득(Gain) 계산 엔진' 모듈 (STFT/iSTFT, WAV 입출력은 포함하지 않음).

파이프라인 (100% 전통 신호처리, numpy/scipy만 사용)
    1단계  IMCRA        : 잡음 PSD λ_d(l,k) 와 음성 부재 확률 q(l,k) 추정      [Cohen, 2003]
    2단계  OM-LSA       : 사전/사후 SNR + SPP 기반 이득 G_om(l,k) 계산          [Cohen & Berdugo, 2001]
    3단계  MD-Kalman    : 변조영역(주파수별 시간 궤적) 칼만필터로 이득 G_kal 계산 [So & Paliwal, 2011 계열]
    4단계  Adaptive Fusion : SNR 적응형 가중치로 G_om 과 G_kal 을 결합 → 최종 gain_matrix

입력/출력 규격 (scipy.signal.stft 의 반환 배열과 동일한 축 순서)
    magnitude : shape (n_freq, n_frames), |Y(l,k)|   (주파수 축 = axis 0, 시간 프레임 축 = axis 1)
    power     : shape (n_freq, n_frames), |Y(l,k)|^2
    반환      : gain_matrix, shape (n_freq, n_frames), 값 범위 [g_min, 1.0]
                팀원 쪽에서  Z_enh = gain_matrix * Z  (Z 는 복소 STFT) 로 위상을 보존하면 된다.
"""

# numpy : 배열 연산 (벡터화된 신호처리 수식 전부)
import numpy as np
# exp1 : 지수적분 E1(v) = ∫_v^∞ e^{-t}/t dt  → LSA 이득식의 exp(0.5*E1(v)) 항에 사용
from scipy.special import exp1
# convolve1d : 주파수 축 3-tap 평활(IMCRA 의 주파수 평활 S_f)에 사용
from scipy.ndimage import convolve1d
# lfilter / lfilter_zi : 융합 가중치 w(l,k) 의 시간축 1차 IIR 평활에 사용
from scipy.signal import lfilter, lfilter_zi


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 전역 상수
# ─────────────────────────────────────────────────────────────────────────────
# 잡음 PSD(분모) 하한: 0 으로 나눠 NaN/Inf 가 생기는 것을 막기 위한 수치 안정화 상수
NOISE_FLOOR = 1e-12
# exp1(v) 입력 v 의 하한: v→0 에서 E1(v)→∞ 로 발산하는 것(언더플로우 쪽)을 차단
EXP1_V_MIN = 1e-8
# exp1(v) 입력 v 의 상한: v 가 너무 크면 exp(-v) 등에서 오버/언더플로우가 나므로 700 으로 제한
EXP1_V_MAX = 700.0
# IMCRA 주파수 평활용 3-tap 창 b = [0.25, 0.5, 0.25] (합=1, Cohen 의 w=1 Hann 형 창)
_FREQ_SMOOTH_WIN = np.array([0.25, 0.5, 0.25])
# AR 자기상관의 백색잡음 보정 계수: r[0] 에 (1+1e-3) 를 곱해 Levinson 재귀를 수치적으로 안정화
_AR_WNC = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# 입력 검증
# ─────────────────────────────────────────────────────────────────────────────
def _validate_inputs(magnitude, power):
    """팀원이 넘겨준 magnitude/power 의 형태·값 범위를 검사하고 float64 로 변환한다."""
    # 입력을 float64 ndarray 로 변환 (정수/float32 입력이 와도 정밀도 확보)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    # power 도 동일하게 변환
    power = np.asarray(power, dtype=np.float64)
    # 2차원 (주파수 × 프레임) 이고 두 배열의 shape 이 같아야 한다
    if magnitude.ndim != 2 or magnitude.shape != power.shape:
        raise ValueError("magnitude 와 power 는 shape (n_freq, n_frames) 로 서로 같아야 합니다.")
    # 시간 재귀(IMCRA/DD/Kalman)를 하려면 최소 2프레임이 필요
    if magnitude.shape[1] < 2:
        raise ValueError("시간 프레임이 최소 2개 이상 필요합니다.")
    # NaN/Inf 가 들어오면 재귀 연산 전체로 전파되므로 사전에 차단
    if not (np.isfinite(magnitude).all() and np.isfinite(power).all()):
        raise ValueError("magnitude/power 에 NaN 또는 Inf 가 있습니다.")
    # 진폭/전력은 음수가 될 수 없다
    if (magnitude < 0).any() or (power < 0).any():
        raise ValueError("magnitude/power 는 0 이상이어야 합니다.")
    # 검증이 끝난 배열을 반환
    return magnitude, power


# ─────────────────────────────────────────────────────────────────────────────
# 1단계 보조: 최소 통계(Minimum Statistics) 추적기
# ─────────────────────────────────────────────────────────────────────────────
class _MinStatTracker:
    """U 개의 부분창(각 V 프레임)으로 이루어진 슬라이딩 최소값 S_min(l,k) 추적기 (Cohen 2003 식 (21))."""

    def __init__(self, n_sub, sub_len):
        # U : 부분창 개수 (전체 탐색 구간 D ≈ U*V 프레임)
        self.n_sub = int(n_sub)
        # V : 부분창 하나의 길이(프레임 수)
        self.sub_len = int(sub_len)
        # 현재까지의 전체 구간 최솟값 S_min (첫 호출에서 초기화됨)
        self.s_min = None
        # 현재 진행 중인 부분창 안에서의 최솟값 S_tmp
        self.s_tmp = None
        # 완료된 부분창들의 최솟값 기록 (최근 U 개만 유지)
        self.history = []
        # 호출 횟수 카운터 (V 프레임마다 부분창을 교체하기 위함)
        self.count = 0

    def update(self, s):
        """새 프레임의 평활 전력 s (shape (K,)) 를 받아 갱신된 S_min 을 반환한다."""
        # 첫 호출이면 S_min, S_tmp 를 현재 프레임 값으로 초기화하고 바로 반환
        if self.s_min is None:
            self.s_min = s.copy()
            self.s_tmp = s.copy()
            return self.s_min
        # 호출 횟수 증가
        self.count += 1
        # V 프레임이 지나 부분창이 끝났다면
        if self.count % self.sub_len == 0:
            # 끝난 부분창의 최솟값(S_tmp)을 기록에 추가
            self.history.append(self.s_tmp.copy())
            # 최근 U 개의 부분창만 유지
            self.history = self.history[-self.n_sub:]
            # 전체 구간 최솟값 = 기록된 부분창 최솟값들의 최소
            self.s_min = np.min(self.history, axis=0)
            # 새 부분창을 현재 프레임 값으로 시작
            self.s_tmp = s.copy()
        else:
            # 부분창 진행 중: 전체 최솟값을 현재 값과 비교해 갱신
            self.s_min = np.minimum(self.s_min, s)
            # 부분창 최솟값도 갱신
            self.s_tmp = np.minimum(self.s_tmp, s)
        # 갱신된 S_min 반환
        return self.s_min


# ─────────────────────────────────────────────────────────────────────────────
# 1단계: IMCRA (Improved Minima Controlled Recursive Averaging) 잡음 추정
# ─────────────────────────────────────────────────────────────────────────────
def _imcra(power, alpha_d, alpha_s, beta, gamma0, gamma1, zeta0, b_min,
           n_sub, sub_len, alpha_xi, xi_min, q_max, init_frames):
    """
    잡음 PSD λ_d(l,k) 와 음성 부재 사전확률 q(l,k) 를 프레임 단위(인과적)로 추정한다.
    반환: noise_psd (K,L) [이미 NOISE_FLOOR 적용], q_absent (K,L)
    """
    # K = 주파수 bin 수, L = 시간 프레임 수
    n_bins, n_frames = power.shape
    # 프레임별 잡음 PSD 를 저장할 배열 λ_d(l,k)
    noise_psd = np.empty((n_bins, n_frames))
    # 프레임별 음성 부재 사전확률 q(l,k) 를 저장할 배열
    q_absent = np.empty((n_bins, n_frames))
    # 초기 잡음 추정에 쓸 프레임 수 (전체 길이를 넘지 않도록 제한)
    init_frames = int(min(max(1, init_frames), n_frames))
    # λ̂_d(0,k): 처음 몇 프레임이 음성 없는 구간이라고 가정하고 평균 전력으로 초기화
    lam_hat = np.maximum(power[:, :init_frames].mean(axis=1), NOISE_FLOOR)
    # 1단계 최소 통계 추적기 (S 용)
    tracker1 = _MinStatTracker(n_sub, sub_len)
    # 2단계 최소 통계 추적기 (S̃ 용; 음성 구간을 제외한 평활 전력)
    tracker2 = _MinStatTracker(n_sub, sub_len)
    # 이전 프레임의 시간 평활 전력 S(l-1,k) (첫 프레임에서 초기화)
    s_prev = None
    # 이전 프레임의 '음성 제외' 평활 전력 S̃(l-1,k)
    s_tilde_prev = None
    # 이전 프레임의 LSA 이득 G_H1(l-1,k) (IMCRA 내부 DD 재귀용)
    g_prev = None
    # 이전 프레임의 사후 SNR γ(l-1,k) (IMCRA 내부 DD 재귀용)
    gamma_prev = None

    # 프레임 순서대로(인과적으로) 처리
    for l in range(n_frames):
        # 현재 프레임의 전력 스펙트럼 |Y(l,k)|^2
        y2 = power[:, l]
        # (a) 주파수 평활  S_f(l,k) = Σ_i b(i)|Y(l,k-i)|^2   (Cohen 식 (13))
        s_f = convolve1d(y2, _FREQ_SMOOTH_WIN, mode="nearest")
        # (b) 시간 평활   S(l,k) = α_s S(l-1,k) + (1-α_s) S_f(l,k)   (Cohen 식 (14))
        s = s_f if s_prev is None else alpha_s * s_prev + (1.0 - alpha_s) * s_f
        # (c) 1단계 최소 통계로 S_min(l,k) 추적
        s_min = tracker1.update(s)
        # 최소값 편향 보정 후의 분모 B_min*S_min (0 방지를 위해 하한 적용)
        denom1 = np.maximum(b_min * s_min, NOISE_FLOOR)
        # (d) 1단계 판정: 순간 비율 γ_min = |Y|^2/(B_min S_min)  (Cohen 식 (20))
        gamma_min = y2 / denom1
        # 평활 전력 비율 ζ = S/(B_min S_min)
        zeta = s / denom1
        # 음성 부재로 '잠정 판정'된 bin 지시자 I(l,k)=1  (γ_min<γ0 이고 ζ<ζ0)
        ind = ((gamma_min < gamma0) & (zeta < zeta0)).astype(np.float64)
        # (e) 잠정 음성 부재 bin 만 이용한 가중 주파수 평활의 분자 Σ b(i) I |Y|^2
        num = convolve1d(ind * y2, _FREQ_SMOOTH_WIN, mode="nearest")
        # 같은 가중 평활의 분모 Σ b(i) I
        den = convolve1d(ind, _FREQ_SMOOTH_WIN, mode="nearest")
        # 첫 프레임이면 S̃ 를 S 로 초기화
        if s_tilde_prev is None:
            s_tilde = s.copy()
        else:
            # 분모가 0 이 아닌 bin 은 S̃(l)=α_s S̃(l-1)+(1-α_s)*(num/den), 0 이면 이전 값 유지 (Cohen 식 (24))
            s_tilde = np.where(den > 1e-12,
                               alpha_s * s_tilde_prev + (1.0 - alpha_s) * num / np.maximum(den, 1e-12),
                               s_tilde_prev)
        # (f) 2단계 최소 통계로 S̃_min(l,k) 추적 (음성 영향이 제거된 더 정확한 최소값)
        s_tilde_min = tracker2.update(s_tilde)
        # 2단계 분모 B_min*S̃_min (하한 적용)
        denom2 = np.maximum(b_min * s_tilde_min, NOISE_FLOOR)
        # 2단계 순간 비율 γ̃_min
        gamma_min2 = y2 / denom2
        # 2단계 평활 전력 비율 ζ̃
        zeta2 = s / denom2
        # (g) 음성 부재 사전확률 q(l,k) 계산 (Cohen 식 (26))  — 기본은 0 (음성 존재)
        q = np.zeros(n_bins)
        # 조건 A: γ̃_min ≤ 1 이고 ζ̃ < ζ0 → 확실한 음성 부재, q = 1
        q = np.where((gamma_min2 <= 1.0) & (zeta2 < zeta0), 1.0, q)
        # 조건 B: 1 < γ̃_min < γ1 이고 ζ̃ < ζ0 → 선형 보간 q = (γ1-γ̃_min)/(γ1-1)
        cond_b = (gamma_min2 > 1.0) & (gamma_min2 < gamma1) & (zeta2 < zeta0)
        q = np.where(cond_b, (gamma1 - gamma_min2) / (gamma1 - 1.0), q)
        # q=1 이면 q/(1-q) 가 발산하므로 상한 q_max 로 제한
        q = np.clip(q, 0.0, q_max)
        # (h) 현재 프레임에서 사용할 잡음 PSD: λ_d = β·λ̂_d (편향 보상 계수 β), 분모 안전 하한 적용
        lam_d = np.maximum(beta * lam_hat, NOISE_FLOOR)
        # 사후 SNR γ(l,k) = |Y|^2 / λ_d
        gamma = y2 / lam_d
        # 순간 SNR 항 max(γ-1, 0)
        inst = np.maximum(gamma - 1.0, 0.0)
        # (i) IMCRA 내부 사전 SNR (DD 방식, α=alpha_xi): 첫 프레임은 순간값만 사용
        if g_prev is None:
            xi = inst
        else:
            xi = alpha_xi * (g_prev ** 2) * gamma_prev + (1.0 - alpha_xi) * inst
        # 사전 SNR 하한 ξ_min (예: -25 dB) 적용
        xi = np.maximum(xi, xi_min)
        # (j) v = ξ/(1+ξ)·γ , 특수함수 입력이므로 [1e-8, 700] 으로 클리핑 (지수 연산 안정화)
        v = np.clip(xi / (1.0 + xi) * gamma, EXP1_V_MIN, EXP1_V_MAX)
        # LSA 이득(음성 존재 가정) G_H1 = ξ/(1+ξ)·exp(0.5·E1(v))
        g_h1 = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))
        # DD 재귀 안정성을 위해 G_H1 을 1 로 제한
        g_h1 = np.minimum(g_h1, 1.0)
        # (k) 음성 존재 확률 SPP  p(l,k) = 1 / (1 + q/(1-q)·(1+ξ)·exp(-v))  (Cohen 식 (7))
        p_spp = 1.0 / (1.0 + q / (1.0 - q) * (1.0 + xi) * np.exp(-v))
        # (l) 시변 망각 계수 α̃_d = α_d + (1-α_d)·p  (음성 존재 확률이 높으면 갱신을 멈춤)
        alpha_tilde = alpha_d + (1.0 - alpha_d) * p_spp
        # 잡음 PSD 재귀 갱신 λ̂_d(l+1) = α̃_d λ̂_d(l) + (1-α̃_d)|Y|^2  (Cohen 식 (5))
        lam_hat = np.maximum(alpha_tilde * lam_hat + (1.0 - alpha_tilde) * y2, NOISE_FLOOR)
        # 이번 프레임의 결과 저장 (이득 계산에는 갱신 전 λ_d(l) 사용 → 인과성 유지)
        noise_psd[:, l] = lam_d
        q_absent[:, l] = q
        # 다음 프레임을 위한 상태 갱신
        s_prev = s
        s_tilde_prev = s_tilde
        g_prev = g_h1
        gamma_prev = gamma

    # 최종 안전장치: 잡음 PSD 의 분모 0 방지 (요구 사항: np.maximum(noise_psd, 1e-12))
    noise_psd = np.maximum(noise_psd, NOISE_FLOOR)
    # 잡음 PSD 와 q 반환
    return noise_psd, q_absent


# ─────────────────────────────────────────────────────────────────────────────
# 2단계: OM-LSA (Optimally Modified Log-Spectral Amplitude)
# ─────────────────────────────────────────────────────────────────────────────
def _om_lsa(power, noise_psd, q_absent, g_min, alpha_dd, xi_min):
    """
    DD 사전 SNR + SPP 를 이용한 OM-LSA 이득 G_om(l,k) 계산.
    반환: gain_om (K,L), xi_all (K,L; 이후 융합 가중치 계산에 쓰는 사전 SNR)
    """
    # 주파수 bin 수 K, 프레임 수 L
    n_bins, n_frames = power.shape
    # OM-LSA 이득 저장 배열
    gain_om = np.empty((n_bins, n_frames))
    # 사전 SNR ξ(l,k) 저장 배열 (융합 단계에서 재사용)
    xi_all = np.empty((n_bins, n_frames))
    # 이전 프레임의 G_H1 (DD 재귀용)
    g_prev = None
    # 이전 프레임의 사후 SNR (DD 재귀용)
    gamma_prev = None
    # g_min 이 0 이어도 거듭제곱 계산이 깨지지 않도록 하는 안전한 하한값
    g_floor = max(g_min, 1e-12)

    # 프레임 순서대로 처리 (DD 가 이전 프레임 결과에 의존)
    for l in range(n_frames):
        # 분모 안전 처리된 잡음 PSD 로 사후 SNR 계산: γ(l,k) = |Y|^2 / λ_d
        gamma = power[:, l] / np.maximum(noise_psd[:, l], NOISE_FLOOR)
        # 순간 SNR 항 max(γ-1, 0) (ML 추정치)
        inst = np.maximum(gamma - 1.0, 0.0)
        # Decision-Directed 사전 SNR: ξ = α_dd·G_H1(l-1)^2·γ(l-1) + (1-α_dd)·max(γ-1,0)
        if g_prev is None:
            # 첫 프레임은 이전 프레임이 없으므로 순간값으로 초기화
            xi = inst
        else:
            xi = alpha_dd * (g_prev ** 2) * gamma_prev + (1.0 - alpha_dd) * inst
        # 사전 SNR 하한 ξ_min 적용 (음성 부재 구간의 과도한 감쇠/뮤지컬 노이즈 억제)
        xi = np.maximum(xi, xi_min)
        # v = ξ/(1+ξ)·γ 를 [1e-8, 700] 으로 클리핑 (exp1 / exp 안정화, 요구 사항)
        v = np.clip(xi / (1.0 + xi) * gamma, EXP1_V_MIN, EXP1_V_MAX)
        # 음성 존재 가정 LSA 이득 G_H1 = ξ/(1+ξ)·exp(0.5·E1(v))
        g_h1 = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))
        # G_H1 은 감쇠 이득이므로 1 을 넘지 않게 제한
        g_h1 = np.minimum(g_h1, 1.0)
        # 음성 존재 확률 SPP: p = 1/(1 + q/(1-q)·(1+ξ)·exp(-v))  (q 는 IMCRA 2단계에서 온 값)
        q = np.clip(q_absent[:, l], 0.0, 0.999)
        p_spp = 1.0 / (1.0 + q / (1.0 - q) * (1.0 + xi) * np.exp(-v))
        # OM-LSA 이득: G = G_H1^p · G_min^(1-p)  (불확실 구간은 G_min 쪽으로 기하 보간)
        g = np.power(g_h1, p_spp) * np.power(g_floor, 1.0 - p_spp)
        # 음성 손상 방지: g_min 을 하한으로 클리핑 (요구 사항: np.clip(gain, g_min, 1.0))
        gain_om[:, l] = np.clip(g, g_min, 1.0)
        # 융합 단계에서 쓸 사전 SNR 저장
        xi_all[:, l] = xi
        # 다음 프레임의 DD 재귀를 위한 상태 갱신
        g_prev = g_h1
        gamma_prev = gamma

    # 이득 행렬과 사전 SNR 반환
    return gain_om, xi_all


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 보조: 배치(bin 벡터화) Levinson-Durbin AR 계수 추정
# ─────────────────────────────────────────────────────────────────────────────
def _levinson_batch(r, order):
    """
    자기상관 r (K, order+1) 로부터 모든 bin 의 AR 계수를 한 번에 구한다.
    모델: x_l = Σ_i φ_i x_{l-i} + w_l , 반환: phi (K, order), err (K,) = 예측오차 분산(=프로세스 잡음 분산 Q)
    """
    # bin 개수
    n_bins = r.shape[0]
    # 예측 다항식 계수 a = [1, a_1, …, a_p]  (x_l + Σ a_i x_{l-i} = w_l)
    a = np.zeros((n_bins, order + 1))
    # a_0 = 1
    a[:, 0] = 1.0
    # 0차 예측오차 = r[0] (아주 작은 값 방지 하한)
    err = np.maximum(r[:, 0], 1e-30)
    # 차수를 1 부터 order 까지 늘리며 재귀
    for i in range(1, order + 1):
        # 반사계수 계산용 누적합 acc = r[i] + Σ_{j=1}^{i-1} a_j r[i-j]
        acc = r[:, i] + np.sum(a[:, 1:i] * r[:, i - 1:0:-1], axis=1)
        # 반사계수 k_i = -acc/err  (안정성 보장을 위해 |k|<1 로 제한)
        k = np.clip(-acc / err, -0.999, 0.999)
        # 이전 차수 계수 복사본
        a_new = a.copy()
        # a_j(new) = a_j + k_i·a_{i-j}  (j=1..i-1)
        a_new[:, 1:i] = a[:, 1:i] + k[:, None] * a[:, i - 1:0:-1]
        # a_i(new) = k_i
        a_new[:, i] = k
        # 갱신 반영
        a = a_new
        # 예측오차 갱신 err ← err·(1-k²)
        err = np.maximum(err * (1.0 - k ** 2), 1e-30)
    # φ_i = -a_i 로 부호 변환하여 반환
    return -a[:, 1:], err


# ─────────────────────────────────────────────────────────────────────────────
# 3단계: Modulation-Domain Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────
def _modulation_kalman(magnitude, pre_enhanced, noise_psd, xi, g_min,
                       order, block_len, q_scale, r_scale):
    """
    각 주파수 bin k 의 진폭 시간 궤적 |Y(·,k)| (= 변조영역 신호)에 AR 상태공간 모델 + 칼만필터를 적용한다.
        상태방정식 : x_l = F x_{l-1} + e1·w_l ,  w ~ N(0, Q)        (x_l = [s̃_l, s̃_{l-1}, ..., s̃_{l-p+1}]^T)
        관측방정식 : ỹ_l = H x_l + v_l , H=e1^T , v ~ N(0, R_l)
    s̃ = 깨끗한 진폭 s 에서 블록 평균 s̄ 를 뺀 편차. AR 계수/Q 는 OM-LSA 사전 복원 결과로 블록마다 추정한다.
    반환: gain_kal (K,L) = clip(Ŝ_kalman/|Y|, g_min, 1)
    """
    # 주파수 bin 수 K, 프레임 수 L
    n_bins, n_frames = magnitude.shape
    # 블록 경계 인덱스 [0, B, 2B, ..., L]  (AR 계수를 블록(준정상 구간) 단위로 갱신)
    edges = list(range(0, n_frames, block_len)) + [n_frames]
    # 마지막 블록이 너무 짧으면(블록 길이의 절반 미만) 직전 블록에 합쳐 AR 추정을 안정화
    if len(edges) > 2 and (edges[-1] - edges[-2]) < block_len // 2:
        del edges[-2]
    # 블록 개수
    n_blocks = len(edges) - 1
    # 프레임 → 블록 번호 매핑 테이블
    frame_block = np.repeat(np.arange(n_blocks), np.diff(edges))

    # 블록별 AR 전이행렬 F_b (K, p, p), 프로세스 잡음 분산 Q_b (K,), 평균 s̄_b (K,), 신호 분산 (K,) 저장 리스트
    f_blocks, q_blocks, mean_blocks, var_blocks = [], [], [], []
    # 각 블록에 대해 사전 복원(OM-LSA 출력) 궤적으로부터 AR 파라미터를 추정
    for b in range(n_blocks):
        # 해당 블록의 사전 복원 진폭 궤적 (K, M)
        seg = pre_enhanced[:, edges[b]:edges[b + 1]]
        # 블록 길이 M
        m = seg.shape[1]
        # bin 별 블록 평균 s̄ (AR 은 평균 제거된 신호에 적용)
        seg_mean = seg.mean(axis=1)
        # 평균 제거 궤적 s̃ = s - s̄
        seg0 = seg - seg_mean[:, None]
        # 자기상관 r[lag] 배열 (K, p+1)
        r = np.zeros((n_bins, order + 1))
        # lag = 0..p 에 대해 편향(biased) 자기상관 r[lag] = (1/M) Σ s̃_l s̃_{l+lag}
        for lag in range(order + 1):
            if lag < m:
                r[:, lag] = np.sum(seg0[:, :m - lag] * seg0[:, lag:], axis=1) / m
        # 블록 신호 분산 r[0] 저장 (칼만 초기 공분산에 사용)
        var_blocks.append(r[:, 0].copy())
        # 백색잡음 보정: r[0] 를 살짝 키워 Toeplitz 행렬을 양의 정부호로 만듦
        r[:, 0] = r[:, 0] * (1.0 + _AR_WNC) + 1e-30
        # Levinson-Durbin 으로 AR 계수 φ 와 예측오차 분산(=Q) 계산
        phi, err = _levinson_batch(r, order)
        # 전이행렬 F (K,p,p) 를 동반(companion) 형태로 구성
        f_mat = np.zeros((n_bins, order, order))
        # 첫 행 = AR 계수 [φ_1 ... φ_p]  (s̃_l = Σ φ_i s̃_{l-i})
        f_mat[:, 0, :] = phi
        # 나머지 행 = 시프트 행렬 (s̃_{l-1} 을 한 칸 아래로 내림)
        for i in range(1, order):
            f_mat[:, i, i - 1] = 1.0
        # 블록별 파라미터 저장 (Q 는 사용자 조절 계수 q_scale 로 스케일)
        f_blocks.append(f_mat)
        q_blocks.append(q_scale * err)
        mean_blocks.append(seg_mean)

    # 칼만 상태 벡터 x (K,p): 평균(s̄)에서의 편차를 0 으로 시작
    x = np.zeros((n_bins, order))
    # 상태 공분산 P (K,p,p): 첫 블록의 신호 분산 × 단위행렬로 초기화(큰 불확실성 → 초기에 관측을 신뢰)
    p_cov = var_blocks[0][:, None, None] * np.eye(order)[None, :, :] + 1e-30
    # 칼만 복원 진폭 Ŝ(l,k) 저장 배열
    s_hat = np.empty((n_bins, n_frames))
    # 현재 블록 번호 추적용 (-1 은 아직 시작 전)
    cur_block = -1

    # 프레임 순서대로 칼만 재귀 수행
    for l in range(n_frames):
        # 현재 프레임이 속한 블록 번호
        b = frame_block[l]
        # 블록이 바뀌는 시점이라면
        if b != cur_block:
            # 이전 블록이 있었다면 평균 기준점이 바뀌므로 상태를 재기준화: s̃' = s̃ + (s̄_old - s̄_new)
            if cur_block >= 0:
                x = x + (mean_blocks[cur_block] - mean_blocks[b])[:, None]
            # 새 블록의 전이행렬/프로세스 잡음/평균으로 교체
            cur_block = b
            f_mat = f_blocks[b]
            q_var = q_blocks[b]
            s_bar = mean_blocks[b]
        # 현재 프레임의 잡음 PSD (분모/분산 안전 하한 적용)
        lam = np.maximum(noise_psd[:, l], NOISE_FLOOR)
        # 저SNR 가중치 w_lo = 1/(1+ξ): SNR 이 낮을수록 1, 높을수록 0 (잡음 진폭 평균 오프셋 적용 정도)
        w_lo = 1.0 / (1.0 + xi[:, l])
        # 잡음 진폭 평균 μ_n = sqrt(π λ_d)/2  (잡음을 Rayleigh 분포로 가정한 E|N|)
        mu_n = np.sqrt(np.pi * lam) / 2.0
        # 관측 잡음 분산 R = r_scale·(4-π)/4·λ_d  (Rayleigh 진폭의 분산, r_scale 로 튜닝)
        r_obs = r_scale * (4.0 - np.pi) / 4.0 * lam
        # 평균/오프셋 제거된 관측치 ỹ_l = |Y| - w_lo·μ_n - s̄
        y_tilde = magnitude[:, l] - w_lo * mu_n - s_bar
        # [예측 단계] 상태 예측 x⁻ = F x
        x_pred = np.einsum("kij,kj->ki", f_mat, x)
        # [예측 단계] 공분산 예측 P⁻ = F P F^T
        p_pred = np.matmul(np.matmul(f_mat, p_cov), f_mat.transpose(0, 2, 1))
        # [예측 단계] 프로세스 잡음 Q 는 첫 상태(새로 들어오는 샘플) 성분에만 더해짐: P⁻ += Q e1 e1^T
        p_pred[:, 0, 0] += q_var
        # [갱신 단계] 이노베이션(예측 오차) e = ỹ - H x⁻
        innov = y_tilde - x_pred[:, 0]
        # [갱신 단계] 이노베이션 분산 S = H P⁻ H^T + R
        s_innov = p_pred[:, 0, 0] + r_obs
        # [갱신 단계] 칼만 이득 K = P⁻ H^T / S  (K,p)
        k_gain = p_pred[:, :, 0] / np.maximum(s_innov, 1e-30)[:, None]
        # [갱신 단계] 상태 갱신 x = x⁻ + K e
        x = x_pred + k_gain * innov[:, None]
        # [갱신 단계] 공분산 갱신 P = (I - K H) P⁻
        p_cov = p_pred - k_gain[:, :, None] * p_pred[:, 0:1, :]
        # 수치 오차로 인한 비대칭을 제거하여 공분산을 대칭으로 유지
        p_cov = 0.5 * (p_cov + p_cov.transpose(0, 2, 1))
        # 복원 진폭 Ŝ(l,k) = s̃ 추정치 + 블록 평균 s̄ (진폭은 음수가 될 수 없으므로 0 으로 하한)
        s_hat[:, l] = np.maximum(x[:, 0] + s_bar, 0.0)

    # 칼만 이득 G_kal = Ŝ / |Y| (분모 0 방지), 음성 손상 방지를 위해 [g_min, 1] 클리핑
    gain_kal = np.clip(s_hat / np.maximum(magnitude, NOISE_FLOOR), g_min, 1.0)
    # 칼만 이득 반환
    return gain_kal


# ─────────────────────────────────────────────────────────────────────────────
# 4단계: Adaptive Fusion
# ─────────────────────────────────────────────────────────────────────────────
def _adaptive_fusion(gain_om, gain_kal, xi, g_min, w_max, snr_center_db, slope_db, w_smooth):
    """
    사전 SNR ξ 에 따라 가중치 w(l,k) 를 바꿔 G_om 과 G_kal 을 로그(기하) 영역에서 결합한다.
        w = w_max / (1 + exp((ξ_dB - snr_center_db)/slope_db))   → 저SNR 일수록 칼만(시간 평활) 비중↑
        G = G_om^(1-w) · G_kal^w
    """
    # 사전 SNR 을 dB 로 변환 (0 방지 하한)
    xi_db = 10.0 * np.log10(np.maximum(xi, 1e-12))
    # 시그모이드 지수부 (오버플로우 방지를 위해 ±50 으로 클리핑)
    expo = np.clip((xi_db - snr_center_db) / max(slope_db, 1e-6), -50.0, 50.0)
    # 칼만 가중치 w ∈ [0, w_max]
    w = w_max / (1.0 + np.exp(expo))
    # 가중치의 프레임 간 깜빡임을 줄이기 위한 1차 IIR 평활 w_s(l) = a·w_s(l-1) + (1-a)·w(l)
    b_iir = np.array([1.0 - w_smooth])
    a_iir = np.array([1.0, -w_smooth])
    # 초기 상태를 첫 프레임 값으로 맞춰 시작 구간에서 가중치가 0 에서 올라오는 현상을 방지
    zi = lfilter_zi(b_iir, a_iir)[None, :] * w[:, 0:1]
    # 시간축(axis=1)으로 평활 적용
    w, _ = lfilter(b_iir, a_iir, w, axis=1, zi=zi)
    # 로그 영역 가중 결합: ln G = (1-w)·ln G_om + w·ln G_kal  (두 이득 모두 0 이하 방지)
    log_g = ((1.0 - w) * np.log(np.maximum(gain_om, 1e-12))
             + w * np.log(np.maximum(gain_kal, 1e-12)))
    # 지수로 복원 후 [g_min, 1] 클리핑
    return np.clip(np.exp(log_g), g_min, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 외부 공개 함수 (팀원이 호출하는 유일한 진입점)
# ─────────────────────────────────────────────────────────────────────────────
def compute_gain_matrix(
        magnitude, power,
        # ── 공통 하이퍼파라미터 ──
        g_min=0.05,              # 이득 하한선 G_min (과감쇠로 인한 PESQ 하락 방지)
        alpha_dd=0.98,           # OM-LSA 의 Decision-Directed 스무딩 팩터 α
        xi_min_db=-25.0,         # 사전 SNR 하한 ξ_min [dB]
        # ── Kalman 단계 하이퍼파라미터 ──
        kalman_order=4,          # 변조영역 AR 모델 차수 p
        kalman_block_len=64,     # AR 계수를 다시 추정하는 블록 길이 [프레임] (hop 8 ms 기준 ≈ 0.5 s)
        kalman_q_scale=1.0,      # 프로세스 잡음 Q 배율 (↑ : 관측 추종↑/평활↓, ↓ : 평활↑)
        kalman_r_scale=1.0,      # 관측 잡음 R 배율   (↑ : 평활↑/관측 불신,  ↓ : 관측 추종↑)
        # ── Fusion 단계 하이퍼파라미터 ──
        fusion_w_max=0.7,        # 칼만 이득의 최대 가중치 (0 이면 OM-LSA 단독, 1 이면 저SNR 에서 칼만 단독)
        fusion_snr_center_db=5.0,  # 가중치가 절반(w_max/2)이 되는 사전 SNR [dB]
        fusion_slope_db=5.0,     # 시그모이드 기울기 폭 [dB] (작을수록 급격히 전환)
        fusion_smooth=0.8,       # 가중치 시간 평활 계수 (0 ≤ a < 1)
        # ── IMCRA 하이퍼파라미터 (기본값은 Cohen 2003 논문 값) ──
        imcra_alpha_d=0.85,      # 잡음 PSD 재귀 평균 기본 망각 계수 α_d
        imcra_alpha_s=0.9,       # 전력 스펙트럼 시간 평활 계수 α_s
        imcra_beta=1.47,         # 잡음 PSD 편향 보상 계수 β
        imcra_gamma0=4.6,        # 1단계 음성 부재 판정 임계 γ0
        imcra_gamma1=3.0,        # 2단계 q 보간 상한 임계 γ1
        imcra_zeta0=1.67,        # 평활 전력 비율 임계 ζ0
        imcra_b_min=1.66,        # 최소 통계 편향 보정 계수 B_min
        imcra_num_sub=8,         # 최소 통계 부분창 개수 U
        imcra_sub_len=15,        # 부분창 길이 V [프레임]  (탐색 구간 ≈ U·V 프레임)
        imcra_alpha_xi=0.92,     # IMCRA 내부 DD 계수 (잡음 추정용 SPP 계산)
        imcra_q_max=0.99,        # 음성 부재 사전확률 q 의 상한
        imcra_init_frames=5,     # 잡음 초기화에 쓰는 선두 프레임 수 (음성 없다고 가정)
):
    """
    IMCRA → OM-LSA → Modulation-Domain Kalman → Adaptive Fusion 으로 최종 필터 이득을 계산한다.

    Parameters
    ----------
    magnitude : ndarray (n_freq, n_frames)  현재 STFT 진폭 |Y(l,k)|
    power     : ndarray (n_freq, n_frames)  전력 스펙트럼 |Y(l,k)|^2
    그 외     : 위 시그니처 주석 참고

    Returns
    -------
    gain_matrix : ndarray (n_freq, n_frames)  최종 결합 이득 G(l,k) ∈ [g_min, 1.0]
    """
    # 입력 검증 및 float64 변환
    magnitude, power = _validate_inputs(magnitude, power)
    # 하이퍼파라미터 범위 검증: g_min ∈ [0,1], α ∈ [0,1)
    if not 0.0 <= g_min <= 1.0:
        raise ValueError("g_min 은 [0, 1] 범위여야 합니다.")
    if not 0.0 <= alpha_dd < 1.0:
        raise ValueError("alpha_dd 는 [0, 1) 범위여야 합니다.")
    if not 0.0 <= fusion_smooth < 1.0:
        raise ValueError("fusion_smooth 는 [0, 1) 범위여야 합니다.")
    if kalman_order < 1 or kalman_block_len < 2 * kalman_order:
        raise ValueError("kalman_order>=1 이고 kalman_block_len>=2*kalman_order 여야 합니다.")
    # dB 단위 사전 SNR 하한을 선형 값으로 변환: ξ_min = 10^(dB/10)
    xi_min = 10.0 ** (xi_min_db / 10.0)

    # [1단계] IMCRA: 잡음 PSD λ_d(l,k) 와 음성 부재 확률 q(l,k) 추정 (power 사용)
    noise_psd, q_absent = _imcra(
        power, imcra_alpha_d, imcra_alpha_s, imcra_beta, imcra_gamma0, imcra_gamma1,
        imcra_zeta0, imcra_b_min, imcra_num_sub, imcra_sub_len, imcra_alpha_xi,
        xi_min, imcra_q_max, imcra_init_frames)

    # [2단계] OM-LSA: 사후/사전 SNR 과 SPP 로 이득 G_om 과 사전 SNR ξ 계산
    gain_om, xi = _om_lsa(power, noise_psd, q_absent, g_min, alpha_dd, xi_min)

    # [3단계] Kalman: OM-LSA 로 사전 복원한 진폭(G_om·|Y|)에서 AR 파라미터를 추정하고 변조영역 필터링
    gain_kal = _modulation_kalman(
        magnitude, gain_om * magnitude, noise_psd, xi, g_min,
        int(kalman_order), int(kalman_block_len), kalman_q_scale, kalman_r_scale)

    # [4단계] Fusion: 사전 SNR 적응형 가중치로 G_om 과 G_kal 을 결합
    gain = _adaptive_fusion(
        gain_om, gain_kal, xi, g_min,
        fusion_w_max, fusion_snr_center_db, fusion_slope_db, fusion_smooth)

    # 마지막 안전장치: 혹시 남은 NaN/Inf 를 치환 (NaN→g_min, +Inf→1, -Inf→g_min)
    gain = np.nan_to_num(gain, nan=g_min, posinf=1.0, neginf=g_min)
    # 음성 손상 방지: 최종 이득을 [g_min, 1.0] 으로 클리핑 (요구 사항)
    gain_matrix = np.clip(gain, g_min, 1.0)
    # 최종 결합 이득 G(l,k) 만 반환 (위상 결합/iSTFT/WAV 입출력은 팀원 코드에서 수행)
    return gain_matrix
