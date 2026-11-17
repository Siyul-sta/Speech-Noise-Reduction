# 1.5 kHz 감쇠 강도와 폭 실험

모드 이름은 추가 감쇠 깊이(dB)와 로그 주파수 Gaussian 표준편차(옥타브)를 나타낸다. 중심은 1500 Hz이며 추론은 noisy만 사용한다. 기준은 2_0.6이다. 개발에서 4_0.6, 2_0.3, 2_0.9를 비교하고 별도 화자에서 개발 상위 두 후보를 확인했다. 4_0.6만 공개 Validation에서 평가했다. 기존 기준의 측정값은 동일 입력에 대한 이전 실행에서 가져왔다.

개발 60개에서 잡음과 음성 각각이 우세하고 해당 성분만 상승한 시간·주파수 칸을 clean 및 noisy-clean으로 선별했다. 9개 이웃 주파수 중 2/3 이상에서 파워가 두 배를 넘는 상태가 3프레임 지속되는지 검사했다. 잡음 상승 363729칸 중 20810칸, 음성 상승 557258칸 중 218090칸에서 작동했다. 이 비율은 선택한 구간에서의 검출률이며 시간·주파수 칸들은 서로 상관되어 있다. 잡음 추적 가속은 채택하지 않았다.

데이터를 noise_reduction_challenge에 두고 r6/requirements-lock.txt를 설치한 뒤 다음 명령으로 실행한다.

```sh
python bandwidth_round/main.py --input noise_reduction_challenge/validation/noisy --output bandwidth_round/wav_final --mode 4_0.6 --workers 6
python bandwidth_round/verify.py
python bandwidth_round/diagnose.py
```

validation_comparison.csv는 같은 164개 파일의 비교다. 공개 Validation은 이전에도 사용했으며 Hidden Test는 미측정이다. 원본 데이터와 출력 WAV는 포함하지 않는다.

공개 Validation 평균 PESQ 2.808424 → 2.821296, SNR 2.373134 → 2.349917 dB, STOI 0.875768 → 0.875627.
