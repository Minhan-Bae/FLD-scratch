import os
import pandas as pd
import torch

from src.utils.get_mkdir import *
from src.utils.get_list import *

## Define Environment

WORKERS = 0 if os.name == 'nt' else 4

## Define Hyperparameter
EPOCHS = 200
BATCH_SIZE = {
    "TRAIN":16,
    "VALID":2
}
LEARNING_RATE= 1e-4
WEIGHT_DECAY = 1e-6

PRETRAINED_ROOT = "/home/ubuntu/workspace/FLD-scratch/src/pretrained_model/face_landmarks.pth"


## Define Path

INPUT_ROOT = "/data/komedi/k_face_100"

IMAGE_ROOT = os.path.join(INPUT_ROOT, "rename_image")
LABEL_ROOT = os.path.join(INPUT_ROOT, "result_lmk")

IMAGE_LIST = get_file_list(IMAGE_ROOT, case="image")
LABEL_LIST = get_file_list(LABEL_ROOT, case="label")

## Dataset

RATIO  = 0.1
LEN_VALID_SET = int(RATIO*len(IMAGE_LIST))
LEN_TRAIN_SET = len(IMAGE_LIST)-LEN_VALID_SET

## Loss and Optimizer