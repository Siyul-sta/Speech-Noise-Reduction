# 음성 잡음 제거 챌린지 후속 실험

## 채택 결과
공개 Validation 164개에서 cross_low의 조교 PESQ-WB는 2.084504675574419, 평균 SNR은 13.617795454241637 dB다. 기존 Multi-resolution은 2.0270171049164563 / 12.89853021896991 dB로 파일별 PESQ까지 재현했다. 신규 코드의 기본 실행으로 PCM16 WAV 164개를 따로 저장하고 원본 노트북으로 다시 읽어 실험표와 164개 점수가 모두 일치함을 확인했다. 2.1과 2.2는 미달이다. 공개 Validation은 후보 선택에 사용했다. 신규 Hidden Test 점수는 없다. 공식 기존 v3 Hidden 1.88227067822995는 인수인계 기록이다.

## 추론 및 평가
Python 3.12, Linux에서 검증했다. WAV 자료는 별도로 준비한다. 아래 명령은 이 README가 있는 폴더에서 실행한다. DATA를 실제 경로로 수정한다. 최종 코드의 기본 입력과 출력은 각각 현재 폴더의 input과 output이다. 모노 16 kHz 입력, 동일 파일명·길이의 PCM16 출력이다.

```bash
python -m pip install -r requirements-lock.txt
export OPENBLAS_NUM_THREADS=1
DATA=../noise_reduction_challenge
python main.py --input "$DATA/validation/noisy" --output selected_wavs
python evaluate_wavs.py --data "$DATA" --notebook noise_reduction_baselines.ipynb --wav-dir selected_wavs --output rescore
python run_hybrid.py --input "$DATA/validation/noisy" --output baseline_wavs
python evaluate_wavs.py --data "$DATA" --notebook noise_reduction_baselines.ipynb --wav-dir baseline_wavs --output baseline_rescore
```
pesq 빌드에서 clang 실행 파일이 없으면 Linux에서 `CC=gcc CXX=g++ LDSHARED='gcc -shared' python -m pip install -r requirements-lock.txt`를 사용한다. evaluate_wavs.py의 method 열은 공통 라벨 optimized이며 폴더 이름이 실제 방법을 식별한다.

## 최종 구조와 설정
main.py가 selected_config.json의 cross_low를 선택한다. best_config.json은 재현 대조군이다. 엔진의 기본값과 구조 상수는 hybrid_v3.py, adaptive.py, enhance_gain.py에 있다. 입력 파형 이외의 추론 인자는 표본률과 고정 설정뿐이다.

1. 전체 구간 스펙트럼에서 국소 바닥보다 강한 피크를 찾고 주파수를 보간한다. 10개 시간 블록의 복소 진폭이 안정된 성분만 추정·차감한다.
2. 1024·2048 FFT, hop 256으로 같은 입력을 별도로 분석한다. 기존 분위수 PSD를 기준으로 16개 bin 단위의 에너지를 계산한다. 35프레임 구간의 20분위수와 파일 내 중앙값의 비로 PSD를 시간에 맞게 조정한다. 변화율을 제한해 순간 음성이 PSD로 흡수되는 것을 줄인다.
3. 기존 DD/TSNR·고조파 보정 이득을 Bark 대역에서 평활한다. 600 Hz를 중심으로 저역에는 에너지 가중 평활과 기존 이득의 기하평균을 섞고, 고역에는 로그 이득을 평활한 값을 섞는다. 각 경로의 진폭 보정계수는 noisy 및 추정 PSD에서 계산한다. 두 해상도의 최종 파형은 기존 70:30 비율로 결합한다.
4. 후보 탐색에서는 시간·주파수 적응 결합도 구현했으나, 최종 공개 PESQ는 이 고정 해상도 결합과 주파수별 이득 결합의 조합이 가장 높았다.

## 데이터 및 탐색
Drive 원본 Train 1296쌍 가운데 35화자 175쌍을 사용했다. train_split.csv는 시드 1611로 20화자 각 3개 개발, 다른 15화자 각 3개 확인 파일을 정했다. 각 화자 안에서 서로 다른 입력 SNR을 우선했다. extra.csv는 시드 11427로 기존 105개를 제외하고 같은 35화자에서 2개씩 추가했다. extra는 새 파일 표본이며 화자 독립 표본은 아니다. 실제 ID와 SHA256은 두 CSV가 기준이다. 잡음·화자·SNR 메타데이터는 표본 구성·진단에만 사용했다.

개발은 대조군 포함 75개 방법이다. 1차 확인 후 공개 Validation에서 3개 신규 후보를 평가했다. 2차에서는 Train에서 협대역 억제와 짧은 FFT 결합 등을 탐색했다. 추가 Train 70개에서 유망했던 tri_short와 tri_auditory를 공개 Validation에서 평가했지만 1차 최고를 넘지 못했다. 모든 후보를 전체 Train이나 Validation에서 평가한 것은 아니다. 각 범위는 all_summary.csv의 stage와 n에서 확인할 수 있다.

```bash
python fetch_train.py --data "$DATA"
python bench.py dev base cross_low --data "$DATA"
python bench.py confirm base cross_low --data "$DATA"
python bench.py extra base cross_low --data "$DATA"
python replay.py --data "$DATA"
python summarize.py
python diagnose_outliers.py --data "$DATA"
```
replay.py는 저장된 실험 배치 전체를 재평가하며 같은 이름의 결과를 갱신하므로 복사본에서 실행한다. 앞의 개별 bench 명령은 예시이므로 전체 재현 시 생략한다. fetch_train.py는 원본 Drive에서 두 manifest의 clean/noisy WAV만 가져와 해시를 검사한다. Validation은 제공 ZIP을 풀어 사용한다.

## 점수표와 진단
- results/all_summary.csv — 방법·분할별 PESQ, SNR, 성공 수, 기존 최고 대비 차이와 개선 파일 수.
- results/all_file_scores.csv — 모든 탐색의 파일별 점수. delta_pesq/delta_snr_db는 noisy 대비, pesq_vs_base/snr_vs_base는 기존 Multi-resolution 대비다.
- results/validation_by_*.csv — 후보별 잡음·목표 SNR·화자 집계. baseline_by_noise_category_target_snr_db_speaker.csv는 기존 모델의 세 조건 교차 집계다. 작은 셀의 평균은 개별 사례로 해석한다.
- results/selected_wav_check 및 results/baseline — 원본 조교 평가 코드의 디스크 WAV 재평가. 원본 6·8·32번 셀을 사용한다. 조교 함수의 파일별 산술평균이며 평가는 PESQ-WB를 사용했다.
- results/outlier_ablation.csv — 가장 악화된 2개 파일에서만 실행한 채택 후 진단. 기계음 speaker037_utt02_seg0004에서 정현파 차감만 우회하면 PESQ가 1.7460에서 2.6591로 회복했다. 해당 사례에서 협대역 차감이 주요 손실 요인이다. 이 두 파일로 선택 설정을 수정하지 않았다.
- original_handoff.zip 및 results/prior_experiment_index.csv — 기존 코드·인수인계·과거 결과와 요약표 색인.
- 초기 개발 점검에서 고정 PSD를 (주파수,1) 배열로 반환한 후보의 시간 인덱스 오류를 발견했다. 수정 이후 해당 후보를 전체 범위에서 다시 실행했다. 중단된 실행은 점수 집계에서 제외했다.
- 전체 실험은 전통 DSP이며 학습된 모델을 사용하지 않았다. 외부 저장소 및 공식 제출 작업은 수행하지 않았다.
