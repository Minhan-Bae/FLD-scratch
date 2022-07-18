# Import Moudles and Packages
import warnings
warnings.filterwarnings("ignore")

import gc
gc.collect()

import torch
import torch.nn as nn
torch.cuda.empty_cache()

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from config import *
from loss.wingloss import *
from utils.fix_seed import *
from utils.visualize import *
from dataset.dataloader import *
from metric.nme import *

import torch.backends.cudnn as cudnn
cudnn.benchmark=True

devices_id = [int(d) for d in DEVICE.split(',')]

# Fix seed
seed_everything(SEED)

# Set dataloader
train_loader, valid_loader = kfacedataloader(batch_size=BATCH_SIZE, workers=WORKERS)

# define validate function
def validate(save = None):
    cum_loss = 0.0
    cum_mean_nme = 0.0
    cum_std_nme = 0.0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for idx, (features, labels) in pbar:
            features = features.cuda()
            labels = labels.cuda()
            
            with autocast(enabled=True):
                outputs = model(features).cuda()
                loss = LOSS(outputs, labels)
                mean_nme, std_nme = NME(outputs, labels)
            
            cum_loss += loss.item()
            cum_mean_nme += mean_nme.item()
            
            description_valid = f"| # mean_nme: {cum_mean_nme/(idx+1):.8f}, cum_loss: {cum_loss/(idx+1):.8f}"
            pbar.set_description(description_valid)
            
        visualize_batch(features[:16].cpu(), outputs[:16].cpu(), labels[:16].cpu(),
                    shape = (4, 4), size = 16, title = None, save = save)
    
    return cum_mean_nme/len(valid_loader), cum_loss/len(valid_loader)

# initialize earlystopping count, best_score
early_cnt = 0
best_loss = np.inf
best_nme = np.inf

log_list = []

OPTIMIZER.zero_grad()

# check time
start_time = time.time()

# load model
model = nn.DataParallel(MODEL, device_ids=devices_id).cuda()
torch.cuda.set_device(devices_id[0])

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
            loss = LOSS(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update()
        
        OPTIMIZER.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}"
        pbar.set_description(description_train)   
    SCHEDULER.step(cum_loss)
    log_list.append(f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}")
    
    if epoch%5==0: 
        model.eval()
        mean_nme, val_loss = validate(os.path.join(f'{SAVE_IMAGE_PATH}',
                                        f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))
        log_list.append(f"     EPOCH : {epoch+1}/{EXP['EPOCH']}\tNME_MEAN : {mean_nme:.8f}\tVAL_LOSS : {val_loss:.8f}")
        torch.save(MODEL.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{TYPE}_{MODEL_NAME}_{epoch}.pt"))
        if val_loss < best_loss:
            early_cnt = 0
            best_loss = val_loss
            print(f'|   >> Saving model..  NME : {mean_nme:.8f}')
            log_list.append(f"|   >> Saving model..   NME : {mean_nme:.8f}")
            torch.save(MODEL.state_dict(), SAVE_MODEL)
        
        else:
            early_cnt += 1
            print(f"Early stopping cnt... {early_cnt}")
            if early_cnt >= EARLY_STOPPING_CNT:
                break

        df = pd.DataFrame(log_list)
        df.to_csv(f"{SAVE_PATH}/{TYPE}_validation.csv", index=None, header=None)
        
print(f"best mean nme is : {best_nme:.8f}")
print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    

log_list.append(f"best mean nme is : {best_nme:.8f}")
log_list.append("Training Complete")
log_list.append("Total Elapsed Time : {} s".format(time.time()-start_time))

df = pd.DataFrame(log_list)
df.to_csv(f"{SAVE_PATH}/{TYPE}_validation.csv", index=None, header=None)