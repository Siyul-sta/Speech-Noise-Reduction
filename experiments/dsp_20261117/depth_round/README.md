# 1.5 kHz 추가 감쇠 실험

기준은 8 dB, 로그 주파수 Gaussian 폭은 0.6옥타브다. 개발 파일 60개에서 12, 16, 24, 32 dB를 비교했다. 선택 후보 12_1500의 공개 Validation 결과는 comparison.csv에 있다. 기존 8 dB는 직전 동일 입력의 결과를 재사용했다. 추론은 noisy만 사용한다. 공개 Validation은 이전 실험에도 사용했으며 Hidden Test는 미측정이다.

데이터를 noise_reduction_challenge에 배치한 뒤 실행한다.

```sh
python -m pip install -r r6/requirements-lock.txt
python depth_round/main.py --input noise_reduction_challenge/validation/noisy --output depth_round/wav_final --mode 12_1500 --workers 6
python depth_round/verify.py
```

평균 PESQ 2.848777, SNR 2.278749 dB, STOI 0.874681. 원본 데이터와 생성 WAV는 압축에서 제외했다.
