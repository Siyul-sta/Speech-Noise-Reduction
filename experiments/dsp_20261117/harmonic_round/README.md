# 배음 보호 실험

현재 공개 Validation 최고점 2.793228을 유지한다. 이번 배음 보호 후보는 미채택이다.

처리된 음성 스펙트럼의 로그 진폭에서 주파수 방향 Gaussian 평활 성분을 제거하고 1024점 실수 역 FFT로 켑스트럼을 계산한다. 40~200샘플 주기(80~400 Hz) 중 최대 피크로 기본주파수를 추정한다. 피크의 평균 대비 표준화 크기로 보호 강도를 조절한다. 기본주파수는 5프레임 중앙값으로 평활한다. 정수 배음에서 폭 0.12배의 Gaussian 보호 마스크를 만들고 4 kHz까지 적용한다. 기존 confidence의 제곱근을 곱해 음성 근거를 반영한다. 최종 추가 감쇠는 기존 noise_4 감쇠에서 보호 마스크에 비례해 0.5, 1, 2 dB 완화한다. 이 보호량과 폭은 본 실험에서 정한 휴리스틱이다. 배음 추정 오류 가능성이 있다. 추론에는 Noisy만 사용한다.

|분할|noise_4|보호 0.5 dB|보호 1 dB|보호 2 dB|
|---|---:|---:|---:|---:|
|개발 60 PESQ|2.722739|2.722228|2.721286|2.718248|
|별도 화자 45 PESQ|2.760577|2.760437|2.760155|미측정|
|개발 60 SNR dB|2.480519|2.509283|2.538645|2.599147|
|별도 화자 45 SNR dB|2.475478|2.508099|2.541568|미측정|

개발 0 dB PESQ는 보호 1 dB에서 1.638031에서 1.642907로 올랐지만 전체 PESQ 개선은 확인되지 않았다. 차이는 작으며 통계적 유의성을 검정하지 않았다. 보호는 SNR을 높였지만 PESQ는 두 분할 모두 소폭 낮췄다. Validation, Hidden Test 및 STOI는 이번 후보에 대해 미측정이다. PCM16으로 변환한 출력을 조교 원본 함수로 평가했다. Clean은 평가에만 사용했다.

## 재현

데이터를 noise_reduction_challenge에 배치하고 r6/requirements-lock.txt를 설치한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python harmonic_round/experiment.py dev noise_4 harmonic_0.5 harmonic_1 harmonic_2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python harmonic_round/experiment.py confirm noise_4 harmonic_0.5 harmonic_1
python harmonic_round/main.py --input INPUT_DIR --output OUTPUT_DIR --mode harmonic_0.5 --workers 4
```

파일별 비교 CSV, summary.csv, by_snr.csv를 포함한다. 데이터와 출력 WAV는 포함하지 않는다.
