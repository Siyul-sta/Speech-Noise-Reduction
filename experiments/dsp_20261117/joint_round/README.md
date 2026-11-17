# 잡음 PSD 보호와 최종 감쇠 동시 비교

이번 조합은 미채택이다. 공개 Validation 최고점 2.793228 설정을 유지한다.

모드는 PSD 보호 강도_최종 추가 감쇠 dB 형식이다. 보호는 Noisy 평활 파워/기존 PSD 비율에서 clip((ratio-2)/8,0,1)을 계산한다. 기존 PSD의 시간 20백분위수보다 높은 로그 성분을 보호 강도와 위 지표의 곱만큼 줄인다. 마지막 감쇠는 shape_round의 noise 계열이며 음성 근거가 약한 칸에 설정 dB만큼 추가 감쇠한다. LSA 25% 혼합과 다중 해상도 처리 및 복원을 적용한다. 추론은 Noisy만 사용하며 학습 모델은 사용하지 않는다.

개발 60개와 별도 화자 45개에서 6개 변경 조합 모두 기준 0_4보다 전체 PESQ가 낮았다. 0_4 기준 및 확인 분할의 0.5_4는 직전 실험의 동일 코드·데이터 결과를 재사용했다. 나머지는 새로 실행했다. 아래 표와 파일별 CSV를 참조한다. 추가 감쇠를 높이면 개발 15 dB 입력 점수는 오르지만 0 dB가 낮아졌다. 차이의 통계적 유의성은 검정하지 않았다. 이번 후보의 Validation, Hidden Test 및 STOI는 미측정이다. 기존 공개 Validation은 여러 실험에서 반복 사용됐다. 평가는 조교 원본 함수로 PCM16 디코딩 출력을 채점했다.

|mode|stage|PESQ|SNR dB|
|---|---|---:|---:|
|0.5_2|confirm|2.750121|2.492209|
|0.5_2|dev|2.715916|2.507266|
|0.5_4|confirm|2.758033|2.470567|
|0.5_4|dev|2.718919|2.484605|
|0.5_6|confirm|2.757548|2.450103|
|0.5_6|dev|2.714617|2.462971|
|0.5_8|confirm|2.749260|2.430716|
|0.5_8|dev|2.704929|2.442313|
|0_4|confirm|2.760577|2.475478|
|0_4|dev|2.722739|2.480519|
|0_6|confirm|2.760224|2.454832|
|0_6|dev|2.718219|2.458717|
|0_8|confirm|2.751971|2.435279|
|0_8|dev|2.708337|2.437914|

## 재현

데이터를 noise_reduction_challenge에 배치하고 r6/requirements-lock.txt 의존성을 설치한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python joint_round/experiment.py dev 0_4 0_6 0_8 0.5_2 0.5_4 0.5_6 0.5_8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python joint_round/experiment.py confirm 0_4 0_6 0_8 0.5_2 0.5_4 0.5_6 0.5_8
python joint_round/main.py --input INPUT_DIR --output OUTPUT_DIR --mode 0_4 --workers 4
```

파일별 CSV와 summary.csv 및 by_snr.csv를 포함한다. 데이터와 WAV는 포함하지 않는다.
