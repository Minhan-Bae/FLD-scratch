# Import Moudles and Packages
import warnings
warnings.filterwarnings("ignore")

import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import os

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from src.config import *
from src.loss.wingloss import *
from src.utils.fix_seed import *
from src.utils.visualize import *
from src.validate import *
from src.dataset.dataloader import *
from src.metric.nme import *
from src.models.timm_swin import *

import torch.backends.cudnn as cudnn
cudnn.benchmark=True

# set single&multi device
devices_id = [int(d) for d in DEVICE.split(',')]

# fix seed
seed_everything(SEED)

# fet dataloader
train_loader, valid_loader = kfacedataloader(batch_size_train=BATCH_SIZE["train"],
                                             batch_size_valid=BATCH_SIZE["valid"],
                                             workers=WORKERS)

# initialize earlystopping count, best_score
early_cnt = 0
best_loss = np.inf
best_nme = np.inf
loss_list = []

# check time
start_time = time.time()

# set model and dp
model = timm_Net_54(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHT_PATH)
model = nn.DataParallel(model, device_ids=devices_id).cuda()
torch.cuda.set_device(devices_id[0])

# set loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = EXP["LR"])
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EXP["EPOCH"], T_mult=1)

optimizer.zero_grad()

# run
for epoch in range(EXP["EPOCH"]):
    cum_loss = 0.0 # define current loss
    scaler = GradScaler() 
    
    model.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, labels) in pbar:
        
        features = features.cuda()
        labels = labels.cuda()

        with autocast(enabled=True):            
            outputs = model(features)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}"
        pbar.set_description(description_train)   
        
    scheduler.step()
    loss_list.append(f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}")
    
    if epoch%5==0: 
        model.eval()
        mean_nme, std_nme = validate(valid_loader, model, os.path.join(f'{SAVE_IMAGE_PATH}',
                                        f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))
        loss_list.append(f"     EPOCH : {epoch}/{EXP['EPOCH']}\tNME_MEAN : {mean_nme:.8f}\tNME_STD : {std_nme:.8f}")
        torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{TYPE}_{MODEL_NAME}_{epoch}.pt"))
        if mean_nme < best_nme:
            early_cnt = 0
            best_nme = mean_nme
            print(f'|   >> Saving model..   Best NME : {best_nme:.8f}')
            torch.save(model.state_dict(), SAVE_MODEL)
        
        else:
            early_cnt += 1
            print(f"Early stopping cnt... {early_cnt}")
            if early_cnt >= EARLY_STOPPING_CNT:
                break

        df = pd.DataFrame(loss_list)
        df.to_csv(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}/validation_log.csv", index=None, header=None)
        
print(f"best mean nme is : {best_nme:.8f}")
print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    