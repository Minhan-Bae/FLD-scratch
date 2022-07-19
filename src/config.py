from datetime import date

import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from models.timm_swin import timm_Net_54
from models.pfld import *
from loss.loss import PFLDLoss
DEVICE = '0,1'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "pfld",
    "EPOCH" : 10,
    "LR" : 2e-5,
}

SEED = 2022

BATCH_SIZE = 1024
WORKERS = 16 # number of gpu * 4

TYPE = "test" # v00_H_M

MODEL_NAME = EXP["MODEL"]

if MODEL_NAME == "pfld":
    PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-18/pfld_v05_19_00/v05_19_00_pfld_best.pt"
    LMK_MODEL, ANG_MODEL = get_model(pfld_pretrained=None, auxil_pretrained=None)
    
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}"
SAVE_IMAGE_PATH = os.path.join(SAVE_PATH,"image_logs")
SAVE_MODEL_PATH = os.path.join(SAVE_PATH,"model_logs")

SAVE_MODEL = os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_{MODEL_NAME}_best.pt")

LOSS = PFLDLoss()
OPTIMIZER = optim.Adam([{'params': LMK_MODEL.parameters()},
                        {'params': ANG_MODEL.parameters()}],
                        lr = EXP["LR"], weight_decay=1e-6)
SCHEDULER = ReduceLROnPlateau(OPTIMIZER, mode='min', patience=40, verbose=True)

EARLY_STOPPING_CNT = 50