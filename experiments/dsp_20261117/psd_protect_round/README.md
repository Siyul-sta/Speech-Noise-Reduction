# 음성 근거를 반영한 잡음 PSD 상승 억제

미채택. 공개 Validation 최고점 2.793228 설정을 유지한다.

기존 local_15_0.5 PSD를 계산하고, Noisy 평활 파워/추정 PSD 비율이 2에서 10 사이일 때 음성 근거를 0~1로 둔다. PSD의 시간 20백분위수를 기준으로 그보다 상승한 로그 성분을 음성 근거와 보호 강도만큼 줄인다. 강도 0.25, 0.5, 1을 비교했다. 이후 LSA 25% 혼합과 noise_4 후처리를 적용한다. 추론은 Noisy만 사용한다. 음성 근거는 휴리스틱이다.

|분할|현재|보호 25%|보호 50%|보호 100%|
|---|---:|---:|---:|---:|
|개발 60 PESQ|2.722739|2.720726|2.718919|2.715566|
|별도 화자 45 PESQ|2.760577|2.759671|2.758033|미측정|

두 분할 모두 PESQ 하락. Validation, Hidden Test, STOI는 이번 후보에 대해 미측정이다. PCM16 변환 후 조교 원본 함수로 평가했다.

## Clean 기반 진단

diagnose.py는 개발 60개 원시 입력의 1024점 STFT에서만 초기 PSD와 local 추정을 재현한다. 최종 설정의 noise_scale까지 곱한다. 다중 해상도 전체 파이프라인의 진단은 아니다. Clean과 noisy-clean을 같은 방식으로 평활해 음성/잡음 파워가 2 초과인 영역과 0.1 미만인 영역을 비교했다. 경계 프레임과 잡음 최대 파워 대비 1e-6 이하 영역은 제외했다.

파일별 PSD 오차 중앙값을 평균하면 음성 우세 영역 +3.067332 dB, 잡음 우세 영역 -0.452415 dB다. 실제 잡음보다 3 dB 넘게 추정한 비율의 파일 평균은 각각 50.7020%, 27.6231%다. 이 오차에는 의도적인 noise_scale도 포함된다. 음성 구간의 상향 편향을 보이지만 원인과 최적 감쇠 강도를 확정하지 않는다. 보호 후보가 PESQ를 낮춘 점을 함께 해석해야 한다. Clean은 진단과 평가에만 사용했다.

## 실행

데이터를 noise_reduction_challenge에 배치하고 r6/requirements-lock.txt 의존성을 설치한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python psd_protect_round/experiment.py dev baseline protect_0.25 protect_0.5 protect_1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python psd_protect_round/experiment.py confirm baseline protect_0.25 protect_0.5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python psd_protect_round/diagnose.py
python psd_protect_round/main.py --input INPUT_DIR --output OUTPUT_DIR --mode protect_0.25 --workers 4
```

데이터와 WAV 출력은 포함하지 않는다. 파일별 CSV와 진단 코드를 포함한다.
