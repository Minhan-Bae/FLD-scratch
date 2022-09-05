## 데이터셋 구성
- 31200 kface dataset(400 * 13 * 6(aug))
- 20124 aflw dataset(1,677 * 2(mirror) * 6(aug))

## Augmentation 작업
Albumentations 프레임워크를 이용하여 총 5종의 Augmentation 작업을 수행했습니다.
- rotate10m: 반시계방향으로 10도 회전
- rotate10: 시계 방향으로 10도 회전
- gausianblur: 가우시안블러
- clahe: CLAHE
- birght: random brightness contrast(limit contrast, brightness 각각 0.05)

## 학습&검증 데이터셋 분리
- 학습데이터와 검증데이터간 명확한 구분을 위해 Augmentation 작업을 하지 않은 데이터셋들을 기준으로 분리하였습니다.

- **Trainset(40,284)**
  - kface: 320명 기준, 총 24,960건
  - aflw: 1,277건 기준, 총 15,324건
- **Validset(11,040)**
  - kface: 80명 기준, 총 6,240건
  - aflw: 400건 기준, 총 4,800건