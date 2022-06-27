import os
import pandas as pd
import torch

import os
import natsort

def get_file_list(path, case):
    if case=="image":
        image_extension = [".jpg", ".png", ".jpeg"]
    else:
        image_extension = [".csv"]
    file_list = []
    for (root, _, files) in os.walk(path):
        if len(files) > 0:
            for file_name in files:
                if file_name[-4:] in image_extension:
                    file_list.append(os.path.join(root,file_name))
    return natsort.natsorted(file_list)




from pathlib import Path

def get_mkdir(path, dir_type):
    dir_dict = {}
    try:
        for d in dir_type:
            dir_name = d+"_path"
            dir_list = os.path.join(path,d)
            dir_dict[d]=dir_list
            
            if not os.path.exists(dir_list):
                Path(dir_list).mkdir(parents=True, exist_ok=True)
                print(f"| {dir_name} was maked")
    except IndexError:
        pass

## Define Environment

WORKERS = 0 if os.name == 'nt' else 4
SEED = 2022

## Define Hyperparameter
EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE= 1e-8
WEIGHT_DECAY = 1e-6

PRETRAINED_ROOT = "/home/ubuntu/workspace/FLD-scratch/src/pretrained_model/face_landmarks_2.pth"


## Define Path

INPUT_ROOT = "/data/komedi/dataset/k-face-100"

IMAGE_ROOT = os.path.join(INPUT_ROOT, "cropped_img")
LABEL_ROOT = os.path.join(INPUT_ROOT, "cropped_lmk")

IMAGE_LIST = get_file_list(IMAGE_ROOT, case="image")
LABEL_LIST = get_file_list(LABEL_ROOT, case="label")

## Dataset

RATIO  = 0.2
LEN_VALID_SET = int(RATIO*len(IMAGE_LIST))
LEN_TRAIN_SET = len(IMAGE_LIST)-LEN_VALID_SET

## Loss and Optimizer