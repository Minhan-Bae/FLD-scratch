from datetime import date

import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pfld import *
from loss.loss import PFLDLoss
DEVICE = '1'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "pfld",
    "EPOCH" : 500,
    "LR" : 1e-4,
}

SEED = 2022

BATCH_SIZE = 256
WORKERS = 4 # number of gpu * 4

TYPE = "v09_17_00" # v00_H_M

MODEL_NAME = EXP["MODEL"]

if MODEL_NAME == "pfld":
    pfld_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v08_14_30/v08_14_30_angle_best.pt"
    auxil_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v08_14_30/v08_14_30_angle_best.pt"
    LMK_MODEL, ANG_MODEL = get_model(pfld_pretrained=pfld_pretrained_path, auxil_pretrained=auxil_pretrained_path)
    
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

EARLY_STOPPING_CNT = 25