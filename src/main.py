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
from loss.loss import *
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
train_loader, valid_loader = AFLWDataloader(batch_size=BATCH_SIZE, workers=WORKERS)

# define validate function
def validate(valid_loader, type, save = None):
    cum_loss = 0.0
    cum_mean_nme = 0.0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for idx, (features, landmarks_gt, euler_angle_gt) in pbar:
            features = features.cuda()
            landmarks_gt = landmarks_gt.cuda()
            euler_angle_gt = euler_angle_gt.cuda()
            
            with autocast(enabled=True):
                _, predicts = pfld_model(features)
                # angle = auxi_model(out)
                
                loss = torch.mean(torch.sum((landmarks_gt - predicts)**2, axis=1))
                mean_nme, _ = NME(predicts, landmarks_gt)
            
            cum_loss += loss.item()
            cum_mean_nme += mean_nme.item()
            
            description_valid = f"| # {type}_mean_nme: {cum_mean_nme/(idx+1):.4f}, cum_loss: {cum_loss/(idx+1):.8f}"
            pbar.set_description(description_valid)
            
        visualize_batch(features[:16].cpu(), predicts[:16].cpu(), landmarks_gt[:16].cpu(),
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
pfld_model = nn.DataParallel(LMK_MODEL, device_ids=devices_id).cuda()
auxi_model = nn.DataParallel(ANG_MODEL, device_ids=devices_id).cuda()

torch.cuda.set_device(devices_id[0])

# run
for epoch in range(EXP["EPOCH"]):
    cum_loss = 0.0 # define current loss
    scaler = GradScaler() 
    
    pfld_model.train()
    auxi_model.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, landmarks_gt, euler_angle_gt) in pbar:
        
        features = features.cuda()
        landmarks_gt = landmarks_gt.cuda()
        euler_angle_gt = euler_angle_gt.cuda()

        with autocast(enabled=True):            
            out, predicts = pfld_model(features)
            angle = auxi_model(out)
            
            weighted_loss, loss = LOSS(landmarks_gt, euler_angle_gt, angle, predicts)
        
        scaler.scale(weighted_loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update(loss.item())
        
        OPTIMIZER.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}"
        pbar.set_description(description_train)   
    SCHEDULER.step(cum_loss)
    log_list.append(f"| # Epoch: {epoch+1}/{EXP['EPOCH']}, Loss: {cum_loss/(idx+1):.8f}")
    
    if epoch%5==0: 
        pfld_model.eval()
        auxi_model.eval()
        mean_nme, val_loss = validate(valid_loader,
                                      type="kface",
                                      save=os.path.join(f'{SAVE_IMAGE_PATH}',
                                                        f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))

        
        log_list.append(f"     EPOCH : {epoch+1}/{EXP['EPOCH']}\tNME_MEAN : {mean_nme:.4f}\tVAL_LOSS : {val_loss:.8f}")
        
        torch.save(pfld_model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{TYPE}_pfld_{epoch}.pt"))
        torch.save(auxi_model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{TYPE}_angle_{epoch}.pt"))

        if mean_nme < best_nme:
            best_nme = mean_nme

        if val_loss < best_loss:
            early_cnt = 0
            best_loss = val_loss
            print(f'|   >> Saving model..  NME : {mean_nme:.4f}')
            log_list.append(f"|   >> Saving model..   NME : {mean_nme:.4f}")
            torch.save(pfld_model.state_dict(),
                       os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_pfld_best.pt"))
            torch.save(auxi_model.state_dict(),
                       os.path.join(f"/data/komedi/logs/{EXP['DAY']}/{EXP['MODEL']}_{TYPE}", f"{TYPE}_angle_best.pt"))

        
        else:
            early_cnt += 1
            print(f"Early stopping cnt... {early_cnt}  Best_NME : {mean_nme:.4f}")
            if early_cnt >= EARLY_STOPPING_CNT:
                break

        df = pd.DataFrame(log_list)
        df.to_csv(f"{SAVE_PATH}/{TYPE}_validation.csv", index=None, header=None)
        
print(f"best mean nme is : {best_nme:.4f}")

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    

log_list.append(f"best mean nme is : {best_nme:.4f}")
log_list.append("Training Complete")
log_list.append("Total Elapsed Time : {} s".format(time.time()-start_time))

df = pd.DataFrame(log_list)
df.to_csv(f"{SAVE_PATH}/{TYPE}_validation.csv", index=None, header=None)