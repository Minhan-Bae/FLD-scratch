from datetime import date
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.timm_swin import timm_Net_54

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "swin",
    "EPOCH" : 200,
    "LR" : 1e-5,
}

SEED = 2022

BATCH_SIZE = 32
WORKERS = 4

TYPE = "from_raw_crop"

MODEL_NAME = "swin_base_patch4_window7_224"
PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-08/all_27pt_swin/model_logs/all_27pt_swin_base_patch4_window7_224_100.pt"

MODEL = timm_Net_54(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_IMAGE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs"
SAVE_MODEL_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs"

SAVE_MODEL = os.path.join(SAVE_MODEL_PATH, f"{TYPE}_{MODEL_NAME}_{EXP['EPOCH']}.pt")
LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"])

EARLY_STOPPING_CNT = 9999