import os
from datetime import date
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pfld import *
from loss.loss import PFLDLoss

device = '0,1'
log_dirs = "v12_13_30" # v00_H_M
experiment = {
    "day": date.today().isoformat(),
    "model" : "pfld",
    "epoch" : 200,
    "lr" : 0.00016,
    "seed" : 2022,
    "batch_size" : 1024,
    "workers" : 4 * len(device.split(',')), # number of gpu * 4
    "early_stop" : 999
}

os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/model_logs",exist_ok=True)

save_path = f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}"
save_image_path = os.path.join(save_path,"image_logs")
save_model_path = os.path.join(save_path,"model_logs")
save_best_model = os.path.join(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}", f"{log_dirs}_{experiment['model']}_best.pt")

pfld_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v06_00_30/v06_00_30_pfld_best.pt"
auxil_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v09_17_00/v09_17_00_angle_best.pt"

pfld_benchmark, auxiliarynet = get_model(pfld_pretrained=pfld_pretrained_path,
                                         auxil_pretrained=None)
    
criterion = PFLDLoss()
optimizer = optim.Adam([{'params': pfld_benchmark.parameters()},
                        {'params': auxiliarynet.parameters()}],
                        lr = experiment["lr"], weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=40, verbose=True)
validation_term = 5
# run CUDA_VISIBLE_DEVICES=0,0 python ~/main.py