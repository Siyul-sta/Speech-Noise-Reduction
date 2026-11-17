# 2회차 제출 이후 추가 실험 · 2026-11-17

## 결론

**공식 2회차 제출 버전 `418790027ff48c1e4149cc3db371f10f80b15bac`을 유지한다.** 가장 좋은 추가 후보의 공개 Validation 평균은 2.849007912로 기준 2.848777247보다 0.000230665 높다. 164개 중 11개 상승, 9개 하락, 144개 동일이며, 한 파일에서 약 0.395 하락한다. 실질적이고 안정적인 개선이라고 판단하지 않는다. 이 실험 브랜치의 코드를 3회차 제출본으로 지정하지 않았다.

## 실험 설계

기존 `r6/train_split.csv`, `r6/extra.csv`의 segment ID를 제외한 Train 파일에서 screen 32개, confirm 48개, audit 64개를 선정했다. 세 단계는 화자가 겹치지 않는다. 이는 저장된 기존 분할과의 비중복을 확인한 것이며, 모든 과거 실험에서 전혀 사용하지 않았다는 뜻은 아니다. 선정은 고정 SHA256 정렬과 잡음/SNR 그룹별 순환 순서로 수행했고 결과 manifest를 보존했다. 모든 전처리와 추론은 Noisy 입력만 사용한다. Clean은 평가에만 쓴다.

첫 screen에서 다음 17개 변형을 기준과 비교했다. 성능이 낮은 후보의 전체 Validation은 실행하지 않았다. 각 결과의 PESQ는 PCM16으로 양자화하고 재디코딩한 신호의 16 kHz WB 점수다.

| 방법 | screen 최고 PESQ 변화 | 결과 |
|---|---:|---|
| 시간/주파수 중앙값 기반 HPSS | -0.004659 | 미채택 |
| 고립된 스펙트럼 성분 억제 | -0.002134 | 미채택 |
| Noisy 위상으로 부분 회전 | -0.006541 | 미채택 |
| 시간 음량 압축/팽창 | +0.000523 | 이득 미미, 미채택 |
| Noisy 기반 음성 포락선 보정 | -0.001716 | 미채택 |
| 고정 정현파 제거 후 기존 DSP 재실행 | +0.008532 | 후속 확인 |
| 고정 정현파의 최종 출력 직접 제거 | -0.034576 | 미채택 |
| 시간 가변 협대역 성분 추적 후 DSP 재실행 | -0.010227 | 미채택 |

## 후속 확인: 고정 정현파 제거

최종 DSP 출력과 Noisy의 잔차에서 스펙트럼 봉우리를 탐지한다. 주파수는 100–6,500 Hz로 제한하며, 국소 배경 대비 전력비 100 이상인 상위 최대 8개 봉우리를 확인한다. 정밀 주파수는 복소 내적 최대화로 찾고, 6개 시간 구간의 복소 계수 일관성이 0.9 이상일 때만 정현파를 채택한다. 채택한 cosine/sine 기저에 Noisy를 최소제곱 투영하여 추정한 성분을 뺀 뒤 기존 DSP를 다시 실행한다. 기계학습이나 잡음 정답·파일명·잡음 라벨에 따른 선택은 없다.

| 단계 | 파일 수 | 기준 PESQ | 후보 PESQ | 변화 |
|---|---:|---:|---:|---:|
| Train screen | 32 | 3.116883 | 3.125415 | +0.008532 |
| Train confirm | 48 | 2.904102 | 2.904550 | +0.000448 |
| Train audit | 64 | 2.879908 | 2.897878 | +0.017970 |
| 공개 Validation | 164 | 2.848777 | 2.849008 | +0.000231 |

Validation 후보 SNR은 2.280742 dB이며 기준은 2.278749 dB이다. STOI는 이번 추가 실험에서 재평가하지 않았다. 파일별 PESQ 차이의 탐색적 paired bootstrap 95% 구간은 약 [-0.005672, +0.004634]로 0을 포함한다. 이는 파일 재표집이며 화자/잡음 종속성을 해결한 검정이 아니다. 공개 Validation 자체도 기존 개발에 반복 사용되었다. Hidden Test는 미측정이다.

일부 Train 경보음에서는 큰 이득이 있었지만 다른 음성/잡음에서는 재현되지 않았다. 특히 Validation의 `speaker016_utt02_seg0012`는 약 0.395 하락해 작은 여러 이득을 상쇄했다. 해당 파일만 예외 처리하거나 정답 기반 라우팅하지 않는다. 다음 연구 과제는 추정 정현파가 실제 잡음인지 검증하고, 제거 에너지가 작거나 음성 고조파와 겹칠 때의 신호 기반 보호 규칙을 설계하는 것이다.

## 재현

저장소 루트에서 실행한다. 공식 데이터셋의 `train`, `validation` 폴더가 있는 경로를 `--data`로 지정한다. 데이터와 캐시는 GitHub에 올리지 않는다. 캐시는 기준 DSP가 고정된 이번 실험 전용이며 기준 코드가 바뀌면 새 캐시 폴더를 사용한다.

```bash
python -m pip install -r requirements.txt
python experiments/post_submission_20261117/experiment.py --data noise_reduction_challenge --manifest experiments/post_submission_20261117/screen_manifest.csv --cache post_cache_4187900 --result screen.csv --modes baseline hpss_0.15 hpss_0.35 isolated_0.15 isolated_0.35 phase_0.25 phase_0.5
python experiments/post_submission_20261117/experiment.py --data noise_reduction_challenge --manifest experiments/post_submission_20261117/screen_manifest.csv --cache post_cache_4187900 --result envelope_screen.csv --modes baseline compand_-0.1 compand_-0.05 compand_0.05 compand_0.1 envelope_0.25 envelope_0.5
python experiments/post_submission_20261117/experiment.py --data noise_reduction_challenge --manifest experiments/post_submission_20261117/screen_manifest.csv --cache post_cache_4187900 --result lines_screen.csv --modes baseline lines_0.7 lines_1.0 linepost_0.7
python experiments/post_submission_20261117/experiment.py --data noise_reduction_challenge --manifest experiments/post_submission_20261117/screen_manifest.csv --cache post_cache_4187900 --result tracking_screen.csv --modes baseline track_0.5 track_1.0
python experiments/post_submission_20261117/experiment.py --data noise_reduction_challenge --manifest experiments/post_submission_20261117/validation_manifest.csv --cache post_cache_4187900 --result validation.csv --modes baseline lines_1.0
```

confirm/audit도 해당 manifest를 지정하여 `baseline lines_1.0`으로 평가한다. CPU 스레드 중복을 줄이려면 Linux에서 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`을 설정한다.

미채택 정현파 후보의 독립 실행 예시는 다음과 같다. 공식 제출 명령인 루트 `main.py`는 바뀌지 않았다.

```bash
python experiments/post_submission_20261117/main.py --input input --output output_experimental --workers 6
```

`coherent_tones.py`는 평가 의존성이 없는 별도 추론 구현이다. 16 kHz 실파일의 정현파 탐지 경로 및 탐지 없는 합성 신호에서 평가 구현과 출력 배열이 동일함을 확인했다. 다른 샘플링 주파수·스테레오·짧은 파일·빈 WAV의 입출력 규격 4개 사례도 검사했다. 1 kHz 합성 잡음에서 추정 파형과 참 정현파의 상관은 0.99999997이었다. 이 검사는 일반적인 음질 개선을 보장하지 않는다.

## 참고 자료

Derry FitzGerald (2010), *Harmonic/Percussive Separation Using Median Filtering*, DAFx-10. [학회 원문](https://www.dafx.de/paper-archive/2010/DAFx10/DerryFitzGerald_DAFx10_P15.pdf). 시간·주파수 방향 중앙값 필터와 soft mask 원리를 확인해 HPSS 후보에 적용했다. 원 논문은 음악 분리를 다루며, 위 음성 잡음 제거 후보는 본 실험의 변형이다. 논문 코드의 복사는 없다. 고정/가변 정현파 제거 코드는 이 저장소의 기존 복소 carrier 추적 원리와 정현파 최소제곱 투영을 사용했다.
