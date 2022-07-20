#!/usr/bin/env python3
#-*- coding:utf-8 -*-

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

import config as C
from loss.loss import *
from utils.fix_seed import *
from utils.visualize import *
from dataset.dataloader import *
from metric.nme import *

from validation import validate

import torch.backends.cudnn as cudnn
cudnn.benchmark=True

# check start time
start_time = time.time()
print(f"exp model : {C.experiment['model']}     version : {C.log_dirs}\n")
# Fix seed
seed_everything(C.experiment["seed"])

# Set dataloader
print("Set dataloader")
train_loader, valid_loader = AFLWDataloader(batch_size=C.experiment["batch_size"],
                                            workers=C.experiment["workers"])

# initialize earlystopping count, best_score
best_loss = np.inf
best_nme = np.inf
early_cnt = 0

log_list = list()


C.optimizer.zero_grad()

# load model
devices_id = [int(d) for d in C.device.split(',')]
torch.cuda.set_device(devices_id[0])

pfld_benchmark = nn.DataParallel(C.pfld_benchmark, device_ids=devices_id).cuda()
auxiliarynet = nn.DataParallel(C.auxiliarynet, device_ids=devices_id).cuda()

# run
print("Start train")
for epoch in range(C.experiment["epoch"]):
    cum_loss = 0.0 # define current loss
    scaler = GradScaler() 
    
    # train mode
    pfld_benchmark.train()
    auxiliarynet.train()
    
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, landmarks_gt, euler_angle_gt) in pbar:
        
        features = features.cuda()
        landmarks_gt = landmarks_gt.cuda()
        euler_angle_gt = euler_angle_gt.cuda()

        with autocast(enabled=True):            
            out, predicts = pfld_benchmark(features)
            angle = auxiliarynet(out)
            
            weighted_loss, loss = C.criterion(predicts, landmarks_gt,
                                              angle, euler_angle_gt)
        
        scaler.scale(weighted_loss).backward()
        scaler.step(C.optimizer)
        scaler.update(loss.item())
        
        C.optimizer.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {epoch+1}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}"
        pbar.set_description(description_train)

    C.scheduler.step(cum_loss)
    log_list.append(f"| # Epoch: {epoch+1}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}")
    
    if epoch%5==0: 
        # valid mode
        pfld_benchmark.eval()
        mean_nme, val_loss = validate(valid_loader,
                                      pfld_benchmark,
                                      save=os.path.join(f'{C.save_image_path}',
                                                        f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))

        
        log_list.append(f"     EPOCH : {epoch+1}/{C.experiment['epoch']}\tNME_MEAN : {mean_nme:.4f}\tVAL_LOSS : {val_loss:.4f}")
        
        torch.save(pfld_benchmark.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_pfld_{epoch}.pt"))
        torch.save(auxiliarynet.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_angle_{epoch}.pt"))

        if mean_nme < best_nme:
            best_nme = mean_nme

        if val_loss < best_loss:
            early_cnt = 0
            best_loss = val_loss
            print(f'|           >> Saving model..  NME : {mean_nme:.4f}')
            log_list.append(f"|   >> Saving model..   NME : {mean_nme:.4f}")
            torch.save(pfld_benchmark.state_dict(),
                       os.path.join(f"/data/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_pfld_best.pt"))
            torch.save(auxiliarynet.state_dict(),
                       os.path.join(f"/data/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_angle_best.pt"))

        
        else:
            early_cnt += 1
            print(f"Early stopping cnt... {early_cnt}  Best_NME : {best_nme:.4f}")
            if early_cnt >= C.experiment["early_stop"]:
                break

        df = pd.DataFrame(log_list)
        df.to_csv(f"{C.save_path}/{C.log_dirs}_validation.csv", index=None, header=None)
        
print(f"best mean nme is : {best_nme:.4f}")

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    

log_list.append(f"best mean nme is : {best_nme:.4f}")
log_list.append("Training Complete")
log_list.append("Total Elapsed Time : {} s".format(time.time()-start_time))

df = pd.DataFrame(log_list)
df.to_csv(f"{C.save_path}/{C.log_dirs}_validation.csv", index=None, header=None)