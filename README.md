# Speech-Noise-Reduction

2026 신호 및 시스템 1차 챌린지 — 전통적 신호 처리 기반 음성 잡음 제거.

## 제출 실행 방법

이 저장소의 **기본 실행은 최종 DSP 설정 `12_1500`**입니다. 저장소 루트에서 아래 명령을 실행합니다. Python 3.12 / Linux의 새 가상환경에서 패키지 설치와 실행을 확인했습니다.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py --input input --output output
```

`input` 폴더에 테스트 WAV 파일을 넣으면 파일별 결과가 `output` 폴더에 같은 이름으로 저장됩니다. `.wav`와 `.WAV`를 모두 처리하며 파일명의 대소문자도 유지합니다. 출력의 샘플링 주파수, 채널 수, 샘플 수는 입력과 같고, 저장 형식은 평가 때와 같은 PCM16 WAV입니다. 입력 폴더와 출력 폴더는 서로 달라야 합니다.

여러 파일을 병렬 처리하려면 다음 명령을 사용합니다.

```bash
python main.py --input input --output output --workers 6
```

추론은 **Noisy WAV만** 사용합니다. Clean 음성, Validation 목록, 평가 노트북, PESQ 또는 학습된 모델을 읽지 않습니다. 설정은 `src/final_dsp.py`와 저장소 내부 DSP 코드·JSON에 고정되어 있습니다. 개인 PC의 절대 경로가 필요하지 않으며 외부 작업 폴더에서 `main.py`의 경로를 지정하여 실행할 수도 있습니다.

16 kHz 모노는 기존 최고 설정과 같은 연산을 수행합니다. 그 외 주파수는 내부적으로 16 kHz로 변환하여 처리한 뒤 원래 주파수로 복원하고, 여러 채널은 각각 처리합니다. 짧은 신호는 끝값 패딩 후 원래 길이로 자릅니다. 다른 형식에서 성능 점수를 측정한 것은 아니며 입출력 형식 보존만 검증했습니다.

## 제출 버전과 팀 정보

- 챌린지: 1차 챌린지
- 코드 버전명: `final-dsp-20261117` (제출 회차 번호와 별개)
- 정확한 제출 커밋: 이 버전을 가리키는 커밋 SHA를 LearnUs 비밀글에 명시합니다. `git rev-parse HEAD`로 확인할 수 있습니다.
- 팀명: **항정살**
- 제출 회차: **2회차**
- AI 활용 내역서: 팀장이 LearnUs에 별도로 첨부합니다.

## 공개 Validation 재검증

2026-11-17, 기본 `main.py`로 WAV를 저장하고 **저장된 PCM16 파일을 다시 읽어** 164개 전체를 평가했습니다.

| 지표 | 결과 |
|---|---:|
| 처리 / 평가 성공 | 164 / 164 |
| 실패 | 0 |
| 평균 PESQ-WB (16 kHz) | 2.8487772469113515 |
| 평균 SNR | 2.278748645453426 dB |
| 평균 STOI | 0.8746806744128797 |

[최종 검증 보고서](docs/FINAL_VERIFICATION.md), [파일별 결과](results/final_submission/validation_per_file.csv), [평가 요약](results/final_submission/summary.json), [검증 환경](results/final_submission/environment.json).

평균은 파일별 점수의 단순 산술평균입니다. SNR은 레벨 정렬 없이 `10*log10(sum(clean**2)/sum((clean-output)**2))`로 계산합니다. 공개 Validation은 반복 비교에 사용됐으며 Hidden Test는 미측정입니다. PESQ 중심으로 선택하여 SNR이 입력보다 낮아지는 대가가 있습니다.

## 평가 재현

공식 데이터셋을 별도로 준비합니다. 원본 음성과 출력 WAV는 저장소에 포함하지 않습니다. `noise_reduction_challenge` 아래에 `validation/samples.csv`, `validation/noisy`, `validation/clean`이 있어야 합니다.

```bash
python main.py --input noise_reduction_challenge/validation/noisy --output output_validation --workers 6
python scripts/evaluate_submission.py --data noise_reduction_challenge --output output_validation --report results/reproduced --workers 6
python -m unittest discover -s tests -p test_submission.py -v
```

Linux에서 CPU 라이브러리의 중복 스레드를 줄이려면 실행 전 `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`을 적용할 수 있습니다. 기본 `--workers 1`도 지원합니다. PESQ 패키지의 wheel이 없는 환경에서는 C/C++ 빌드 도구가 필요할 수 있습니다.

## 코드와 실험 기록

- `main.py`: 최종 제출 진입점. `--method final`이 기본값입니다.
- `src/final_dsp.py`: 평가 코드와 분리한 최종 DSP 설정 및 입출력 변환.
- `experiments/dsp_20261117/r6`: 공통 DSP 구현과 고정 JSON 설정.
- `experiments/dsp_20261117/psd_round/noisy_estimators.py`: Noisy 기반 지역 PSD 추정.
- [실험 기록](experiments/dsp_20261117/README.md): 후보 비교, 미채택 실험 및 기존 결과.
- `scripts/evaluate_submission.py`: 저장 WAV의 독립 평가 도구. 추론에서 import하지 않습니다.

기존 A/B 엔진은 비교용으로 보존하며 `--method a` 또는 `--method b`로 명시적으로 선택합니다. 위 Validation 점수는 기본 `final`에만 해당합니다. 과거 실험의 oracle/clean 기반 진단은 최종 추론에서 사용하지 않습니다.
