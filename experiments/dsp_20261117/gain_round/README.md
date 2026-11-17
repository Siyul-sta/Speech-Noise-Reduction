# 이득 계산식 비교

전통 DSP와 Noisy 한 파일만 사용한다. 학습 모델은 없다. 이전 최고 local_15_0.5의 잡음 PSD와 이후 복원 구조를 고정하고 첫 단계의 Wiener 이득에 로그MMSE 또는 MMSE-STSA 이득을 혼합한다.

## 공개 Validation 164개

|설정|PESQ-WB|SNR dB|STOI|
|---|---:|---:|---:|
|기존 local_15_0.5|2.772536|2.445669|0.878969|
|로그MMSE 25% 혼합|2.775065|2.439910|0.879724|

실제 main.py 진입점으로 PCM16 WAV를 생성하고 조교 원본 평가 함수로 재읽었다. PESQ는 83개 개선, 81개 하락했다. 3 이상인 파일은 기존 60개에서 59개가 됐다. 전체 평균 3.0까지 0.224935 남는다. 공식 Hidden Test는 미측정이다. 공개 Validation은 이전부터 반복적인 후보 비교에 사용됐다. 평균 PESQ 변화는 +0.002529로 작다. 새로운 독립 데이터에서의 개선은 확인하지 않았다.

입력 SNR 0/5/10/15 dB의 기존 평균 PESQ는 1.950228/2.464806/2.830149/3.145030, 새 후보는 1.980018/2.474617/2.829607/3.138331이다. 광대역 잡음 10개는 2.508101에서 2.514490로 올랐다. 저 SNR 개선과 높은 SNR 구간의 손실이 함께 있다.

## 후보 선택

개발 Train 60개에서 기존 대조군과 lsa 50%/100%, stsa 50%/100%, sqrt-Wiener 50%의 5개 신규 후보를 비교했다. lsa 50%가 개발에서 가장 높았지만 별도 화자 Train 45개에서는 lsa 50%와 stsa 50% 모두 기존보다 낮았다.

후속으로 lsa 15%/25%, 최종 이득만 lsa 50% 또는 stsa 50%로 바꾼 총 4개 후보를 개발 60개와 확인 45개에서 평가했다. lsa 25%는 개발 2.699580에서 2.708832, 확인 2.733073에서 2.735109로 올랐다. 추가 Train 70개에서도 2.746836에서 2.753064로 올라, 이 후보를 고정해 Validation에 적용했다. 신규 후보는 총 9개다.

## 계산

xi는 사전 SNR, gamma는 사후 SNR이다. v=xi*gamma/(1+xi), Wiener 이득은 xi/(1+xi)다. LSA는 Wiener 이득에 exp(0.5*E1(v))를 곱한다. STSA는 표준 변형 Bessel 함수 식을 사용하며 수치 안정을 위해 i0e/i1e를 쓴다. 각 대체 이득을 기존 floor~1 범위로 제한한 뒤 Wiener 이득과 혼합한다. 채택 후보는 75% Wiener와 25% LSA다. 첫 결정지향 이득과 두 번째 TSNR 이득에 모두 적용한다. 이득 변화가 다음 프레임 사전 SNR에도 영향을 주는 구조다. final 후보는 두 번째 이득만 교체하는 별도 진단이다. harmonic 재생성, 후속 평활, EQ, 정현파·반복 잡음 차감, 최종 대역 필터는 기존 구조를 사용한다.

## 실행

압축을 푼 상위 폴더에서 Python 3.12와 r6/requirements-lock.txt의 의존성으로 실행한다. gain_round, psd_round, r6 폴더를 함께 유지한다. 추론에는 Clean이나 manifest가 필요하지 않다.

```bash
OPENBLAS_NUM_THREADS=1 python gain_round/main.py --input /path/to/noisy --output /path/to/output --mode lsa_0.25 --workers 4
```

--mode baseline은 이전 최고다. 평가를 재현하려면 원본 noise_reduction_challenge 데이터를 별도로 둔다. 데이터는 압축에 없다.

```bash
OPENBLAS_NUM_THREADS=1 python gain_round/experiment.py dev baseline lsa_0.5 lsa_1.0 stsa_0.5 stsa_1.0 sqrt_0.5
OPENBLAS_NUM_THREADS=1 python gain_round/experiment.py confirm baseline lsa_0.5 stsa_0.5
OPENBLAS_NUM_THREADS=1 python gain_round/experiment.py dev lsa_0.15 lsa_0.25 lsafinal_0.5 stsafinal_0.5
OPENBLAS_NUM_THREADS=1 python gain_round/experiment.py confirm lsa_0.15 lsa_0.25 lsafinal_0.5 stsafinal_0.5
OPENBLAS_NUM_THREADS=1 python gain_round/experiment.py extra baseline lsa_0.25
```

summary.csv는 Train 요약, all_file_scores.csv는 Train 파일별 점수, disk_lsa025는 최종 Validation WAV 평가, stoi.csv는 파일별 명료도 점수다. 이번 변경의 활성화는 gain_round 진입점에서 수행한다. core.GAIN_RULE 기본값은 None이며, 추론 후 finally에서 해제한다.
