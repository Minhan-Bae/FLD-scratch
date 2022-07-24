import os
from datetime import date

from models.pfld import *
from models.timm_swin import *
from models.xception import *
from loss.loss import PFLDLoss

device = '0,1,2'
devices_id = [int(d) for d in device.split(',')]

log_dirs = "24_00" # H_M
experiment = {
    "day": date.today().isoformat(),
    "model" : "xception",
    "epoch" : 60,
    "lr" : 4e-4,
    "seed" : 2022,
    "batch_size" : 256,
    "workers" : 4 * len(device.split(',')), # number of gpu * 4
    "early_stop" : 999
}

os.makedirs(f"/data/komedi/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs/kface", exist_ok=True)
os.makedirs(f"/data/komedi/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs/aflw", exist_ok=True)

os.makedirs(f"/data/komedi/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/model_logs",exist_ok=True)

save_path = f"/data/komedi/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}"
save_image_path = os.path.join(save_path,"image_logs")
save_model_path = os.path.join(save_path,"model_logs")
save_best_model = os.path.join(f"/data/komedi/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}", f"{log_dirs}_{experiment['model']}_best.pt")


pretrained_path = "/data/komedi/komedi/logs/2022-07-24/xception_23_00/23_00_best.pt"
xception_Net = XceptionNet(num_classes=27*2)
if len(devices_id) != 1:
    xception_Net = nn.DataParallel(xception_Net, device_ids=devices_id)

if pretrained_path:
    xception_Net.eval()
    xception_Net.module.load_state_dict(torch.load(pretrained_path, map_location = 'cpu'), strict=False)
    
validation_term = 5


# run CUDA_VISIBLE_DEVICES=0,0 python ~/main.py