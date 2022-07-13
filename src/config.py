import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from datetime import date
import timm

# define model
def timm_Net_54(model_name, pretrained=None, num_classes=27*2):
    model = timm.create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        dropout_rate=0.2
        )

    if not pretrained==None:
        model.eval()
        model.load_state_dict(torch.load(pretrained, map_location = 'cpu'), strict=False)
            
    return model

TYPE = "v7"

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
EXP = {
    "DAY": date.today().isoformat(),
    "MODEL" : "swin",
    "EPOCH" : 100,
    "LR" : 2e-4,
}

SEED = 2022

BATCH_SIZE = {
    "train": 256,
    "valid": 16
}

WORKERS = 4

# model part
MODEL_NAME = "swin_base_patch4_window7_224"
PRETRAINED_WEIGHT_PATH = "/data/komedi/logs/2022-07-13/swin_v6/model_logs/v6_swin_base_patch4_window7_224_best.pt"
MODEL = timm_Net_54(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)

os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs",exist_ok=True)

SAVE_IMAGE_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/image_logs"
SAVE_MODEL_PATH = f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/model_logs"

SAVE_MODEL = os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_{MODEL_NAME}_best.pt")

LOSS = nn.MSELoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr = EXP["LR"])
SCHEDULER = CosineAnnealingWarmRestarts(OPTIMIZER, T_0=EXP["EPOCH"], T_mult=1)

EARLY_STOPPING_CNT = 20