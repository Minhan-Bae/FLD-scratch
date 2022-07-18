from datetime import date

import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.timm_swin import timm_Net_54
from models.xception import XceptionNet
from models.pfld import *

DEVICE = '0,1'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "pfld",
    "EPOCH" : 500,
    "LR" : 1e-4,
}

SEED = 2022

BATCH_SIZE = 512
WORKERS = 16 # number of gpu * 4

TYPE = "ver2" # change nme metric
MODEL_NAME="pfld"
# MODEL_NAME = "swin_base_patch4_window7_224"
# PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-15/swin_v15/v15_swin_base_patch4_window7_224_best.pt"

MODEL = {
    "PFLD": get_pfld(),
    "ANGLE": get_auxiliarynet()
}

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}"
SAVE_IMAGE_PATH = os.path.join(SAVE_PATH,"image_logs")
SAVE_MODEL_PATH = os.path.join(SAVE_PATH,"model_logs")

SAVE_MODEL = os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_{MODEL_NAME}_best.pt")
LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"], weight_decay=1e-6)
SCHEDULER =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMIZER, mode='min', patience=40, verbose=True)
EARLY_STOPPING_CNT = 50

"""
Note: 22-07-18 03:21

Epoch 200 -> 500
Add angle

roll: 코 가운데 점 턱점 두점의 기울어진 정도로 지정
pitch: 모르겠다..#TODO 옆모습은 할 수 있는데, 이게 문제네..?
yaw: 눈 끝점과 가운데점의 거리가 같다고 했을 때, 비율로 계산

"""
#TODO set gt angle
#TODO change loss

