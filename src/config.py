from datetime import date

import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from models.timm_swin import timm_Net_54
from models.pfld import *

DEVICE = '0,1'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "pfld",
    "EPOCH" : 500,
    "LR" : 2e-5,
}

SEED = 2022

BATCH_SIZE = 1024
WORKERS = 16 # number of gpu * 4

TYPE = "v07_10_00" # v00_H_M

MODEL_NAME = EXP["MODEL"]

if MODEL_NAME == "pfld":
    PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-18/pfld_v05_19_00/v05_19_00_pfld_best.pt"
    MODEL = get_model(pretrained=PRETRAINED_WEIGHT_PATH)
    
elif MODEL_NAME == "swin_base_patch4_window7_224":
    PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-15/swin_v15/v15_swin_base_patch4_window7_224_best.pt"    
    MODEL = timm_Net_54(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs/kface", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs/aflw", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}"
SAVE_IMAGE_PATH = os.path.join(SAVE_PATH,"image_logs")
SAVE_MODEL_PATH = os.path.join(SAVE_PATH,"model_logs")

SAVE_MODEL = os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_{MODEL_NAME}_best.pt")

LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"], weight_decay=1e-6)
SCHEDULER = ReduceLROnPlateau(OPTIMIZER, mode='min', patience=40, verbose=True)

EARLY_STOPPING_CNT = 50