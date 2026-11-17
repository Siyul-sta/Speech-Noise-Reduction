# 입력별 최종 대역 감쇠 실험

## 공개 Validation 실측

|설정|PESQ-WB|평균 SNR dB|STOI|
|---|---:|---:|---:|
|기존 최고 lsa_0.25|2.775065|2.439910|0.879724|
|음성 증거가 약한 칸 추가 감쇠 noise_4|2.793228|2.398457|0.875909|

Validation 164개를 실제 main.py로 실행해 PCM16 WAV로 저장한 뒤 조교 원본 함수로 재읽어 평가했다. PESQ는 101개 개선, 63개 하락했고, 3 이상 파일은 59개에서 64개로 늘었다. 전체 평균 3.0까지 0.206772가 남는다. 새 후보는 PESQ를 우선한 선택이며 SNR과 STOI는 낮아졌다.

입력 SNR 0/5/10/15 dB의 기존 평균 PESQ는 1.980018/2.474617/2.829607/3.138331, 새 후보는 1.949472/2.486784/2.848707/3.172689다. 가장 낮은 0 dB 조건에서는 하락했다. 광대역 잡음 10개는 2.514490에서 2.539267로 올랐다. 공식 Hidden Test와 새 독립 데이터 점수는 미측정이다. 공개 Validation은 이전부터 여러 후보 비교에 사용됐다.

## 알고리즘과 비교

기존 local_15_0.5 잡음 추정, lsa_0.25 이득과 후속 다중 해상도 복원은 유지한다. 마지막 500 Hz 중심 필터의 주파수별 감쇠 깊이만 입력에 따라 조절한다. 해당 필터의 기존 전역 진폭 맞춤과 이후 hi_floor_tone_1.0 처리는 유지한다.

다중 해상도 결합 직후 스펙트럼의 파워를 주파수 표준편차 2칸, 시간 1.5프레임으로 평활한다. Noisy 파워도 같은 방식으로 평활한 뒤 시간 20백분위수를 배경으로 삼는다. confidence=speech_power/(speech_power+background)다. noise_4에서는 기존 감쇠 깊이에 4*(1-confidence) dB를 더한다. confidence는 Noisy에서 얻은 처리 결과의 에너지 지표이며 실제 음성 존재 확률을 보장하지 않는다. 500 Hz 중심 응답에서는 감쇠 추가 영향이 작고 바깥 대역에서 커진다. 학습 모델, Clean, 잡음 라벨은 추론에 사용하지 않는다.

처음에는 confidence 또는 시간 대비까지 반영한 contrast에 따라 감쇠를 2/4/8 dB 줄이는 총 6개 후보를 Train 개발 60개에서 평가했다. 모두 SNR은 높였지만 PESQ는 낮아졌다. 이후 confidence에 비례해 2/4 dB 감쇠를 늘린 2개, 1-confidence에 비례해 2/4 dB 늘린 2개를 비교했다. 총 10개 신규 후보 중 noise_4를 선택했다.

기존 대비 noise_4 PESQ는 개발 60개 2.708832에서 2.722739, 별도 화자 확인 45개 2.735109에서 2.760577, 추가 Train 70개 2.753064에서 2.765702로 올랐다. Train 세 분할 비교 후 설정을 고정해 Validation에서 측정했다.

## 실행

압축을 푼 상위 디렉터리에서 실행한다. shape_round, gain_round, psd_round, r6 폴더를 함께 유지한다. Python 3.12와 r6/requirements-lock.txt의 버전을 사용했다.

```bash
OPENBLAS_NUM_THREADS=1 python shape_round/main.py --input /path/to/noisy --output /path/to/output --mode noise_4 --workers 4
```

기존 최고는 --mode baseline으로 실행한다. 추론에는 Clean이나 데이터셋 manifest가 필요하지 않다. 실험 재현에는 원본 noise_reduction_challenge/train 및 validation 데이터가 별도로 필요하다. 데이터는 압축에 포함하지 않는다. Train 표본 ID는 r6/train_split.csv와 extra.csv에 있다.

```bash
OPENBLAS_NUM_THREADS=1 python shape_round/experiment.py dev baseline confidence_2 confidence_4 confidence_8 contrast_2 contrast_4 contrast_8
OPENBLAS_NUM_THREADS=1 python shape_round/experiment.py dev confidence_-2 confidence_-4 noise_2 noise_4
OPENBLAS_NUM_THREADS=1 python shape_round/experiment.py confirm baseline noise_2 noise_4
OPENBLAS_NUM_THREADS=1 python shape_round/experiment.py extra baseline noise_4
```

summary.csv와 all_file_scores.csv는 Train 결과, disk_baseline과 disk_noise4는 실제 Validation WAV 재평가, stoi.csv는 파일별 명료도 결과다. pockets.DEPTH_RULE은 이번 실행 진입점에서 설정하고 finally에서 None으로 복원한다. 이전 코드 복구 후 필요한 평가 의존성을 같은 버전으로 재설치했으며, 개발 대조군과 Validation 대조군이 이전 기록과 일치했다.
