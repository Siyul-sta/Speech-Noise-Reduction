# Hybrid v3 Multi-resolution — 2026-11-16

## 최종 후보
- 공개 Validation 164개: PESQ-WB 2.027017105, 평균 SNR 12.898530 dB, 평균 SNR 개선량 +3.264385 dB. 모든 후보 비교는 PCM16 WAV 양자화 후 조교 노트북 원본 지표/집계 함수로 측정.
- 직전 후보 1.999140079 대비 PESQ +0.027877026, 기존 v3 1.784602 대비 약 +0.242415.
- 48개 탐색, 나머지 116개 확인(2.018124322). 전체 Validation을 반복 사용해 후보를 선택했으므로 독립적인 일반화 검증은 아님. Hidden Test 성능은 미확인.
- 원본 Noisy만 추론 입력. 학습 모델, 정답 참조, 잡음 라벨 기반 추론 없음.

## 구조
분위수 기반 PSD(quantile=.08, Gamma shape=3, shrink=.5), 저주파 PSD 보정(0.7^(1/(1+(f/700)^2))), noise_scale=1.5. DD Wiener 60% + TSNR/HRNR 40%; floor=.01, DD alpha=.94, HRNR weight=2. 병렬 STFT 1024/2048, hop 256. 시간 평활 sigma=2 프레임, 주파수 평활 sigma=1.0/1.5 bins. 최종 파형 70%/30% 선형 결합. 출력 길이와 PCM subtype 유지.

## 실행
```
pip install -r requirements.txt
python run_hybrid.py --input input --output output
```
기존 v3 대조군: `python run_hybrid.py --estimator v3 --input input --output output_v3`. 기본값은 `best_config.json`의 최종 후보. 코드 독립 실행용으로 GitHub main은 변경하지 않음.

## 재현 평가
조교 원본 `noise_reduction_baselines.ipynb`와 데이터 루트(예: `noise_reduction_challenge`) 필요.
```
pip install -r requirements-eval.txt
python run_hybrid.py --input noise_reduction_challenge/validation/noisy --output validation_output
python evaluate_wavs.py --data noise_reduction_challenge --notebook noise_reduction_baselines.ipynb --wav-dir validation_output --output verification
```
`evaluate_wavs.py`는 원본 노트북의 셀 6, 8, 32에서 설정/manifest/지표 함수를 재사용. 노트북이 바뀌면 셀 번호 확인. 실험 코드는 `search.py`, `multi_search.py`, `final_search.py` 순서이며 각각 `--data`, `--notebook`, `--output` 지정 (출력 디렉터리는 각각 패키지 내 results, multi_results, final_results). `screen_ids.csv`는 탐색용 48개 ID. 각 디렉터리의 per_file, summary, configs CSV/JSON으로 점수 검증 가능. `verification`은 실제 생성 WAV 재평가. 원본 데이터/WAV 자체는 패키지에 포함하지 않음.
