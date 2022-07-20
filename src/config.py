import os
from datetime import date
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pfld import *
from models.timm_swin import *
from loss.loss import PFLDLoss

device = '0,1'
log_dirs = "v13_15_00" # v00_H_M
experiment = {
    "day": date.today().isoformat(),
    "model" : "swin",
    "epoch" : 200,
    "lr" : float(1e-4),
    "seed" : 2022,
    "batch_size" : 128,
    "workers" : 4 * len(device.split(',')), # number of gpu * 4
    "early_stop" : 999
}

os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/image_logs", exist_ok=True)
os.makedirs(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}/model_logs",exist_ok=True)

save_path = f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}"
save_image_path = os.path.join(save_path,"image_logs")
save_model_path = os.path.join(save_path,"model_logs")
save_best_model = os.path.join(f"/data/komedi/logs/{experiment['day']}/{experiment['model']}_{log_dirs}", f"{log_dirs}_{experiment['model']}_best.pt")

if experiment['model'] == 'pfld':
    pfld_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v06_00_30/v06_00_30_pfld_best.pt"
    auxil_pretrained_path = "/data/komedi/logs/2022-07-19/pfld_v09_17_00/v09_17_00_angle_best.pt"

    pfld_benchmark, auxiliarynet = get_model(pfld_pretrained=pfld_pretrained_path,
                                             auxil_pretrained=None)

    criterion = PFLDLoss()
    optimizer = optim.Adam([{'params': pfld_benchmark.parameters()},
                        {'params': auxiliarynet.parameters()}],
                        lr = experiment["lr"], weight_decay=1e-6)

elif experiment['model'] == 'swin':
    swin_pretrained_path = "/data/komedi/logs/high_performance_pretrained/v15_swin_base_patch4_window7_224_best.pt"
    
    swin_net = timm_Net_54(model_name = 'swin_base_patch4_window7_224',
                           pretrained=swin_pretrained_path)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(swin_net.parameters(), lr = experiment["lr"], weight_decay = 1e-6) 

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=40, verbose=True)
validation_term = 1


# run CUDA_VISIBLE_DEVICES=0,0 python ~/main.py