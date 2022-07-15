from datetime import date

import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.timm_swin import timm_Net_54
DEVICE = '0,1,2,3'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "swin",
    "EPOCH" : 500,
    "LR" : 1e-4,
}

SEED = 2022

BATCH_SIZE = 256
WORKERS = 16 # number of gpu * 4

TYPE = "v16"

MODEL_NAME = "swin_base_patch4_window7_224"
PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/high_performance_pretrained/v15_swin_base_patch4_window7_224_best.pt"

MODEL = timm_Net_54(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}"
SAVE_IMAGE_PATH = os.path.join(SAVE_PATH,"image_logs")
SAVE_MODEL_PATH = os.path.join(SAVE_PATH,"model_logs")

SAVE_MODEL = os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_{MODEL_NAME}_best.pt")
LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"])
SCHEDULER = CosineAnnealingWarmRestarts(OPTIMIZER, T_0=EXP["EPOCH"], T_mult=1)
EARLY_STOPPING_CNT = 20 