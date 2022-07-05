from datetime import date
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.timm_swin import timm_Net

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "swin",
    "EPOCH" : 40,
    "LR" : 1e-4,
}

SEED = 2022

BATCH_SIZE = 32
WORKERS = 4

MODEL_NAME = "swin_base_patch4_window7_224"
PRETRAINED_WEIGHT_PATH = None

MODEL = timm_Net(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/kface_progresses_Nopretrained_{EXP['MODEL']}", exist_ok=True)
os.makedirs(f"/data/komedi/pretrained_model/{EXP['DAY']}",exist_ok=True)

SAVE_IMAGE_PATH = f"/data/komedi/logs/{EXP['DAY']}/kface_progresses_Nopretrained_{EXP['MODEL']}"
SAVE_MODEL_PATH = f"/data/komedi/pretrained_model/{EXP['DAY']}"

SAVE_MODEL = os.path.join(SAVE_MODEL_PATH, f"{MODEL_NAME}_Nopretrained_{EXP['EPOCH']}.pt")
LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"])