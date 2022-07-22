import os
from datetime import date
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pfld import *
from models.timm_swin import *
from models.xception import *
from loss.loss import PFLDLoss

device = '0,1,2'
devices_id = [int(d) for d in device.split(',')]

log_dirs = "v20_13_00" # v00_H_M
experiment = {
    "day": date.today().isoformat(),
    "model" : "xception",
    "epoch" : 100,
    "lr" : 1e-4,
    "seed" : 2022,
    "batch_size" : 256,
    "workers" : 4 * len(device.split(',')), # number of gpu * 4
    "early_stop" : 999
}

os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs/kface", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs/aflw", exist_ok=True)

os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/model_logs",exist_ok=True)

save_path = f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}"
save_image_path = os.path.join(save_path,"image_logs")
save_model_path = os.path.join(save_path,"model_logs")
save_best_model = os.path.join(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}", f"{log_dirs}_{experiment['model']}_best.pt")


pretrained_path = None
xception_Net = XceptionNet(num_classes=27*2)

xception_Net = nn.DataParallel(xception_Net, device_ids=devices_id).cuda()

if pretrained_path:
    xception_Net.eval()
    xception_Net.module.load_state_dict(torch.load(pretrained_path, map_location = 'cpu'), strict=False)
    
criterion = nn.MSELoss()
optimizer = optim.Adam(xception_Net.parameters(), lr = experiment["lr"], weight_decay = 1e-6) 

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=40, verbose=True)
validation_term = 5


# run CUDA_VISIBLE_DEVICES=0,0 python ~/main.py