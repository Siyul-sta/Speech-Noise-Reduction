# Validation 결과 — 164개 음성 쌍 (2026-11-10)

## 데이터와 방법

- 데이터: 수업에서 제공한 Validation **164개 Clean/Noisy 쌍**.
- 두 알고리즘 모두 음성 복원 시에는 **Noisy 입력만 사용**했음.
- Clean 음성은 오직 자체 성능 평가의 참조값으로만 사용했음.
- SNR 계산: `10 * log10(sum(clean**2) / sum((clean - estimate)**2))` (dB).
- 파일별 원자료: [dual_engine_validation_snr.csv](dual_engine_validation_snr.csv) — **492행 = 164개 × 3가지 방법**.
- 이 자료는 로컬에서 실행한 독립 엔진 테스트 결과이며, **히든 테스트 공식 점수가 아님**.

## 평균 SNR

| 방법 | 평균 SNR (dB) | Noisy 대비 개선 (dB) |
|---|---:|---:|
| Noisy 원본 | 9.6341 | — |
| A: IMCRA + OM-LSA + MD Kalman + Fusion | 10.3798 | +0.7457 |
| B: TSNR + HRNR + Wiener v3 | **14.3826** | **+4.7485** |

## 파일별 비교

- B가 A보다 SNR이 높은 파일: **152 / 164**.
- A가 B보다 SNR이 높은 파일: **12 / 164**.
- B가 Noisy보다 SNR이 높은 파일: **163 / 164**.
- A가 Noisy보다 SNR이 높은 파일: **101 / 164**.

## 주의 사항

1. **PESQ-WB는 미측정**. CSV의 `pesq_wb` 열은 빈 값이며 0점이 아니다.
2. **대회 최종 순위는 히든 테스트 평균 PESQ-WB**로 결정되므로, 이 SNR 결과만으로 우승 후보를 확정할 수 없다.
3. 본 결과는 기존 코드 기본 파라미터로 측정한 164개 파일에 한정된다. 다른 잡음, 파일 길이 및 설정에서는 결과가 달라질 수 있다.
4. 최종 제출 버전이 변경되면 해당 커밋 해시, 설정값 및 평가 결과를 새로 기록해야 한다.

## 후속 작업

- `python -m src.evaluation --data noise_reduction_challenge` 명령으로 두 엔진의 PESQ-WB를 계산하고, 결과를 별도 CSV로 추가.
- AI 활용 내역서(`docs/AI_Usage_Report.md`) 및 최종 실행 검증은 별도 제출 준비 작업.
