# DSP 실험 기록 · 2026-11-17

현재 최고 공개 Validation PESQ-WB는 2.848777이다. SNR 2.278749 dB, STOI 0.874681이며 평균 3.0까지 0.151223이 남는다. 최고 설정은 depth_round의 12_1500이다. 공개 Validation은 여러 차례 비교에 사용됐고 Hidden Test는 미측정이다. 최근 개선은 PESQ를 우선하며 SNR과 STOI가 소폭 하락했다.

## 실행

이 폴더에서 다음 명령을 실행한다. 데이터셋을 이 폴더의 noise_reduction_challenge에 배치한다. 실제 추론에는 noisy만 사용하며 clean은 평가·진단에 사용한다.

```bash
python -m pip install -r r6/requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python depth_round/main.py --input noise_reduction_challenge/validation/noisy --output depth_round/wav_final --mode 12_1500 --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python depth_round/verify.py
```

## 진행 기록

|단계|공개 Validation PESQ|
|---|---:|
|다중 해상도 복원|2.718377|
|음성 복원 하한 개선|2.766637|
|지역 잡음 PSD 추정|2.772536|
|LSA 25% 혼합|2.775065|
|입력별 추가 감쇠|2.793228|
|1.5 kHz 대역 감쇠 2 dB|2.808424|
|동일 대역 4 dB|2.821296|
|동일 대역 8 dB|2.841050|
|동일 대역 12 dB|2.848777|

각 round 폴더의 코드 및 CSV는 성공한 후보와 미채택 후보를 함께 기록한다. r6는 공통 DSP 의존 코드, experiment와 hybrid_multires 및 history는 작업 폴더에 남아 있는 초기 실험 기록이다. adaptive_round의 파일별 자동 감쇠는 개발·확인 분할에서 약간 올랐으나 추가 Train에서 낮아져 미채택이다. 이 후보의 Validation은 미측정이다.

원본 음성, 출력 WAV, ZIP, Python 캐시는 업로드에서 제외했다. 초기 시점의 일부 진단은 현재 로컬 작업 폴더에 없으며 기존 대화에서 제공한 ZIP에 남아 있다. 이 폴더는 현재 보유한 소스와 결과의 스냅샷이다.
