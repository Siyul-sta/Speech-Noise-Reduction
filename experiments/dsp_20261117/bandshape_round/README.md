# 주파수별 감쇠 구조 개선

## 공개 Validation 실측

|설정|PESQ-WB|SNR dB|STOI|
|---|---:|---:|---:|
|직전 최고 noise_4|2.793228|2.398457|0.875909|
|새 후보 presence_2|2.808424|2.373134|0.875768|

PESQ +0.015196. 164개 중 129개 개선, 35개 하락. PESQ 3 이상은 64개에서 66개로 늘었다. 전체 평균 3.0까지 0.191576이 남는다. SNR과 STOI는 낮아진 PESQ 우선 후보다. Hidden Test는 미측정이다. 공개 Validation은 이전 실험에서도 반복 사용됐다.

입력 SNR 0/5/10/15 dB별 PESQ는 기존 1.949472/2.486784/2.848707/3.172689에서 1.954970/2.494860/2.867024/3.192251로 올랐다. 전체 평균이 모든 개별 파일 개선을 뜻하지는 않는다. hf_channel과 phone_dialing 카테고리 평균은 하락했다.

## 알고리즘

기존 local_15_0.5 잡음 PSD, LSA 25% 혼합, 다중 해상도 복원과 noise_4의 입력별 추가 감쇠를 적용한다. 마지막 500 Hz 중심 필터에서 1.5 kHz 중심 폭 0.6 octave의 Gaussian 대역 가중치에 비례해 감쇠 깊이를 최대 2 dB 더한다.

weight=exp(-0.5*(log2(f/1500)/0.6)^2), depth=기존 주파수별 깊이+4*(1-confidence)+2*weight다. 실제 필터 dB 응답은 기존 500 Hz 중심 primary 응답과 결합하므로 모든 위치에서 정확히 2 dB 더 감소하는 것은 아니다. 기존 전역 진폭 보정과 후속 hi_floor_tone_1.0 복원을 적용한다. 추론에 Noisy만 사용한다. Clean과 잡음 라벨은 추론에 사용하지 않는다. 학습 모델은 사용하지 않는다.

## 선택과 검증

개발 60개에서 저역 400 Hz 이하, 고역 2 kHz 이상, 1.5 kHz 중심 대역의 감쇠 깊이를 각각 +/-2 dB 조절한 6개 후보를 비교했다. presence_2는 기존 2.722739에서 2.737058로 개선했다. 별도 화자 45개에서는 2.760577→2.775916, 추가 Train 70개에서는 2.765702→2.782202로 개선했다. 비교 기준값은 직전 동일 파이프라인 측정값을 재사용했다. 고역 전체 +2 dB 후보는 개발에서 개선했지만 별도 화자에서 하락했다. presence_2를 고정한 뒤 공개 Validation을 측정했다.

main.py 정상 진입점으로 164개 PCM16 WAV를 생성한 뒤 verify.py가 디스크 파일을 다시 읽어 조교 원본 PESQ 및 SNR 함수와 pystoi로 평가했다. PESQ 오류는 없었고 길이, 샘플레이트, PCM16 형식과 유한값을 검사했다. 파일별 결과와 잡음·SNR별 결과 CSV를 포함한다.

## 실행

압축을 풀고 데이터셋을 noise_reduction_challenge에 배치한다. r6/requirements-lock.txt 의존성을 설치한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bandshape_round/main.py --input noise_reduction_challenge/validation/noisy --output bandshape_round/wav_presence2 --mode presence_2 --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bandshape_round/verify.py
```

실험 재현은 experiment.py에 dev, confirm 또는 extra와 모드들을 전달한다. 후보는 low_-2, low_2, high_-2, high_2, presence_-2, presence_2다. 기본 데이터와 WAV 출력은 압축에 포함하지 않는다.
