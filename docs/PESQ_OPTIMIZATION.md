# PESQ-WB 최적화 계획 — 공식 히든 테스트 기준 1.88227067822995

## 목표
- **히든 테스트 공식 기준점**: 1.88227067822995 (기존 기본 제출 B).
- **목표**: 히든 테스트 PESQ-WB 2.0 초과.
- 이 점수는 공개 Validation PESQ와 다른 데이터에서 측정된 값으로, 직접 비교하거나 개선을 보장할 수 없다.
- 해당 공식 점수를 반복적인 파라미터 선택에 사용하지 않는다.

## 실행

Python 3.10+ 환경에서 `pip install -r requirements.txt`로 패키지를 설치한다. `pesq==0.0.4` 필요.

공식 공개 데이터 ZIP을 로컬에 보관하고 (GitHub에 음성 파일을 업로드하지 않음):

```bash
python scripts/optimize_pesq.py --data noise_reduction_challenge.zip --metric pesq --output-dir results/pesq_search
```

압축을 해제했다면 `--data noise_reduction_challenge`로 지정한다.

빠른 동작 확인(**PESQ가 아닌 SNR 검사**):

```bash
python scripts/optimize_pesq.py --data noise_reduction_challenge.zip --metric snr --limit 12 --configs baseline,less_hrnr,no_hrnr,dry_0p1,freq_0p5 --output-dir results/smoke
```

## 설정 후보

`scripts/optimize_pesq.py`는 **기본 B (TSNR + HRNR + Wiener v3)**에서 파라미터를 한 항목씩 변경한 21가지 후보를 평가한다.

- HRNR 강도: 0, 0.2, 0.5, 1.0, 기본 2.0
- 잡음 추정 배율: 0.7, 0.85, 1.0, 1.15, 1.3
- Dry mix: 0, 0.05, 0.1, 0.2
- Gain floor: 0.05, 0.1, 0.15, 0.2
- Gain 평활화: 주파수 / 시간 / 병합
- FFT: 512/128, 768/192, 기본 1024/256

## 데이터 분리 및 과적합 완화

Validation의 Clean은 **평가 단계에서만** 사용하며 `enhance(noisy, sr, Config(...))`로 들어가지 않는다.

입력의 `segment_id`에서 화자 ID를 추출하고, 해시로 결정적 정렬하여 화자별로 Tune / Holdout으로 분리한다. 공개 데이터는 현재 5명의 화자, 총 164개 음성으로 확인되었으며 실험 환경에서는 Tune 130개 / Holdout 34개로 분리된다.

1. **Tune**에 대해서만 최고 평균 PESQ-WB 후보 선택
2. 선택한 설정의 **Holdout** 점수를 사후 확인
3. Holdout 결과를 보고 재튜닝한다면 그 데이터는 더 이상 독립 검증셋이 아니므로 새로운 평가 분할을 확보할 것
4. 전체 공개 데이터의 PESQ-WB, SNR, 음원별 오류와 잡음 종류별 경향을 분석
5. 검증된 설정을 코드에 반영할 경우 **새 Git 커밋/태그 및 AI 활용 내역서 갱신**

## 출력 파일

- `results/pesq_search/per_file.csv` — 파일별 Noisy 및 파라미터별 SNR/PESQ
- `results/pesq_search/summary_by_subset.csv` — Tune/Holdout 평균 점수
- `results/pesq_search/selection.json` — Tune 기준 선택 설정과 Holdout 결과

**주의:** `--metric snr` 실행 시 `pesq_wb` 열은 비어 있다. SNR이 높아도 PESQ가 높다고 단정하면 안 된다.

## 현재까지 확인된 결과 (SNR 기능 점검 전용)

공식 ZIP으로 164개를 실행한 5개 설정의 SNR 점검은 아래와 같다. 이것은 PESQ 최적화 결과가 아니다.

| 설정 | Tune 평균 SNR (130개) | Holdout 평균 SNR (34개) |
|---|---:|---:|
| baseline | 14.3354 | 14.5631 |
| less_hrnr (0.5) | 14.2537 | 14.6609 |
| no_hrnr (0) | 14.1280 | 14.6591 |
| dry_0p1 | 14.1926 | 14.2958 |
| freq_0p5 | 14.2609 | 14.5715 |

**PESQ-WB는 현재 환경에서 `pesq` 패키지 설치 네트워크 제약으로 실행되지 않았다.** 최적 설정을 선택했다고 주장하지 않는다.

## 추론/제출 전 주의

파라미터 최적화 코드는 **개발·평가 전용**이며 현재 `main.py`의 기본 B 설정을 자동으로 바꾸지 않는다. 최종 설정이 확인되면 별도 PR로 B의 기본값을 바꾸고, 실제 입력/출력 길이·채널·샘플링레이트 및 조교 실행 방법을 재점검해야 한다.
