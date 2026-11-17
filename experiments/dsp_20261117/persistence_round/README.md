# 잡음 지속성에 따른 추가 감쇠 실험

결과는 미채택이다. 기존 공개 Validation 최고점 2.793228 설정을 유지한다.

Noisy 파워를 평활하고 로그 파워의 31프레임(약 0.5초) 이동 표준편차로 지속성을 계산했다. stability=exp(-std/0.5), persistence=(1-amount)+amount*stability다. 마지막 추가 감쇠 4*(1-confidence)에 persistence를 곱한다. amount는 0.25, 0.5, 1을 비교했다. 추론은 Noisy만 사용하며 학습 모델은 사용하지 않는다. 지속성은 음성과 잡음을 완전히 구분하는 지표가 아니다.

|분할|현재 noise_4|지속성 25%|지속성 50%|지속성 100%|
|---|---:|---:|---:|---:|
|개발 60개 PESQ|2.722739|2.722356|2.720357|2.712499|
|별도 화자 45개 PESQ|2.760577|2.757924|2.754635|미측정|

개발 0 dB는 1.638031에서 각각 1.646119, 1.653253, 1.664867로 회복됐다. 개발 15 dB는 3.314561에서 3.307788, 3.298972, 3.277519로 낮아졌다. 전체 SNR은 개선됐지만 전체 PESQ는 두 분할에서 하락했다. Validation 및 Hidden Test, STOI는 이번 후보에 대해 측정하지 않았다. PCM16으로 변환 후 조교 원본 PESQ 함수로 평가했다. 공개 Validation을 반복 비교해온 기존 최고점의 한계도 유지된다.

## 재현

압축을 풀고 데이터셋을 noise_reduction_challenge에 배치한다. r6/requirements-lock.txt 의존성을 설치한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python persistence_round/experiment.py dev noise_4 persist_0.25 persist_0.5 persist_1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python persistence_round/experiment.py confirm noise_4 persist_0.25 persist_0.5
python persistence_round/main.py --input INPUT_WAV_DIR --output OUTPUT_WAV_DIR --mode persist_0.25 --workers 4
```

summary.csv, by_snr.csv와 파일별 CSV를 포함한다. 데이터와 WAV 출력은 포함하지 않는다.
