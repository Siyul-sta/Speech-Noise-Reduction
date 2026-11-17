# 1.5 kHz 감쇠 강도와 중심 주파수 실험

모드 이름은 감쇠 깊이(dB)_중심 주파수(Hz)이며 로그 주파수 Gaussian 표준편차는 0.6옥타브다. 기존 기준은 4_1500, Train 선택 후보는 8_1500다. 동일 입력의 이전 기준 측정값을 재사용했다. 공개 Validation 164개에 이 후보 하나를 적용하고 PCM16 WAV를 다시 읽어 평가했다. 추론에는 noisy만 사용한다. 공개 Validation은 여러 실험에서 반복 사용됐고 Hidden Test는 미측정이다.

실행은 데이터셋을 noise_reduction_challenge에 놓고 r6/requirements-lock.txt를 설치한 뒤 진행한다.

```sh
python center_round/main.py --input noise_reduction_challenge/validation/noisy --output center_round/wav_final --mode 8_1500 --workers 6
python center_round/verify.py
```

결과는 validation_comparison.csv 및 summary.csv에 있다. WAV와 원본 데이터는 제외했다.
