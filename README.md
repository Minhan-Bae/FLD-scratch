# Facial Landmark Detection for Scratch

### Directory Configuration

```
FLD-SCRATCH
	├─ APP/
	│	├─ deployments/
	│	│	├─ backend(underconstruct)/
	│	│	├─ fronted(underconstruct)/	
	│	│	│	├─ utils/
	│	│	│	│	├─ face_detector.py
	│	│	│	│	├─ fig2img.py
	│	│	│	│	├─ index.py
	│	│	│	│	├─ transform.py
	│	│	│	│	├─ visualization.py	
	│	│	│	│	└─ xception.py
	│	│	│	├─ __init__.py
	│	│	│	├─ front.py
	│	│	│	└─ landmark_detector.py
	│	│	│
	│	│	├─ demo_end.sh
	│	│	├─ demo_run.sh
	│	│	├─ docker_run.sh
	│	│	├─ Dockerfile
	│	│	└─ requirements.txt	
	│	├─ docs
	│	└─ README.md
	├─ SRC/
	│	├─ configs/
	│	│	├─ __init__.py
	│	│	├─ config.py
	│	│
	│	├─ dataset/
	│	│	├─ __init__.py
	│	│	├─ augmentation.py
	│	│	├─ dataloader.py
	│	│	└─ dataset.py
	│	│
	│	├─ models/
	│	│	├─ loss/
	│	│	│	├─ __init__.py
	│	│	│	├─ custom_mseloss.py
	│	│	│	└─ wing_loss.py
	│	│	├─ metric/
	│	│	│	├─ __init__.py
	│	│	│	└─ nme.py
	│	│	├─ pretrained/		
	│	│	├─ __init__.py
	│	│	└─ xception.py
	│	│
	│	├─ preprocessing/
	│	│	├─ __init__.py
	│	│	├─ face_detector/
	│	│	├─ facial_data_augmentation/
	│	│	└─ facial_data_generation/
	│	│
	│	├─ utils/
	│	│	├─ __init__.py
	│	│	├─ averagemeter.py	
	│	│	├─ fix_seed.py
	│	│	├─ optimizer.py
	│	│	├─ str2bool.py
	│	│	└─ visualize.py
	│	│	
	│	├─ main.py
	│	├─ requirements.txt
	│	├─ run_front.sh	
	│	└─ run_train.sh
	│		
	└─ README.md

```

<br>

## 학습 및 평가

### 사용 Face Detection model: facenet-mtcnn
안면 랜드마크 탐지를 위해 Facenet의 mtcnn face-detection 모델을 이용하였습니다.

<br>

### 학습 실행

```bash
#  Quick start
$ cd SRC
$ sh run_train.sh

or

$ python SRC/main.py \
  --image-dir {이미지 파일 경로}\
  --train-csv-path {학습 csv 파일 경로}\
  --valid-csv-path {검증 csv 파일 경로}\

```

<br>

### 평가 metric: NME
![nme_metric](https://user-images.githubusercontent.com/84002905/186812535-34758507-3f16-42f0-864d-6f73c02e6f42.jpg)

<br>

### 평가 결과
**정량적 평가**
- NME metric의 normalize 부분을 입력 이미지 사이즈로 설정하여 %로 계산한 결과 입니다.
  
	|        항목                  |      결과      | 목표 | |
	| :-------------------:        | :----: | :----: | :--: |
	| 데이터 신청: K-face 안면 데이터셋 신청 |  | <=3.5 | |

**정성적 평가**
![capture](https://user-images.githubusercontent.com/84002905/186813851-c2657461-4831-450a-93f4-a379c1d2f086.JPG)