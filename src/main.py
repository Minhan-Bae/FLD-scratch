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

# init parameters & logs
log_list = list()

best_loss = np.inf
best_nme = np.inf
early_cnt = 0

def logging(text):
    global log_list
    print(text)
    log_list.append(text)

# Fix seed
seed_everything(C.experiment["seed"])

# check start time
start_time = time.time()
logging(f"exp model : {C.experiment['model']}     version : {C.log_dirs}\n")

# Set dataloader
logging("Set dataloader")
train_loader, valid_loader = AFLWDataloader(batch_size=C.experiment["batch_size"],
                                            workers=C.experiment["workers"])

# load model
devices_id = [int(d) for d in C.device.split(',')]
torch.cuda.set_device(devices_id[0])

if C.experiment['model'] == "pfld":
    flmk_model = nn.DataParallel(C.pfld_benchmark, device_ids=devices_id).cuda()
    angl_model = nn.DataParallel(C.auxiliarynet, device_ids=devices_id).cuda()
elif C.experiment['model'] == "swin":
    flmk_model = nn.DataParallel(C.swin_net, device_ids=devices_id).cuda()
    angl_model = None
elif C.experiment['model'] == "xception":
    flmk_model = nn.DataParallel(C.xception_Net, device_ids=devices_id).cuda()
    angl_model = None

C.optimizer.zero_grad()

# set validate
def validate(valid_loader, save = None):
    flmk_model.eval()
    
    nme = []
    nme_20 = []
    losses = []
    
    with torch.no_grad():
        for img, landmark_gt, euler_angle_gt in valid_loader:
            img = img.cuda()
            landmark_gt = landmark_gt.cuda()
            euler_angle_gt = euler_angle_gt.cuda()
            
            flmk_model.cuda()
            
            landmark = flmk_model(img)
                
            mean_nme = NME(landmark, landmark_gt)
            mean_nme_20 = NME_20(landmark, landmark_gt)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            
            nme.append(mean_nme)
            nme_20.append(mean_nme_20)
            losses.append(loss.cpu().numpy())
            
        visualize_batch(img[:16].cpu(), landmark[:16].cpu(), landmark_gt[:16].cpu(),
                    shape = (4, 4), size = 16, title = None, save = save)

    logging("|     ===> Evaluate:")
    logging('|          Eval set: Normalize Mean Error_20 : {:.4f} '.format(np.mean(nme_20)))          
    logging('|          Eval set: Normalize Mean Error: {:.4f} '.format(np.mean(nme)))
    logging('|          Eval set: Average loss: {:.4f} '.format(np.mean(losses)))    
    return np.mean(nme), np.mean(losses)

# run
logging("Start train")
for epoch in range(1, C.experiment["epoch"]+1):
    cum_loss = 0.0 # define current loss
    scaler = GradScaler() 
    
    # train mode
    flmk_model.train()
    if angl_model:
        angl_model.train()
    
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, landmarks_gt, euler_angle_gt) in pbar:
        
        features = features.cuda()
        landmarks_gt = landmarks_gt.cuda()
        euler_angle_gt = euler_angle_gt.cuda()

        with autocast(enabled=True):         
               
            if angl_model:
                out, predicts = flmk_model(features)
                angle = angl_model(out)
            else:
                predicts = flmk_model(features)
                
            if C.experiment['model'] == 'pfld':
                weighted_loss, loss = C.criterion(predicts, landmarks_gt,
                                                  angle, euler_angle_gt)
            else:
                weighted_loss = C.criterion(predicts, landmarks_gt)
                loss = weighted_loss
    
        scaler.scale(weighted_loss).backward()
        scaler.step(C.optimizer)
        scaler.update(weighted_loss.item())
        
        C.optimizer.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}"
        pbar.set_description(description_train)
        
    
    C.scheduler.step(cum_loss)
    
    logging(f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}")

    if epoch%C.validation_term == 0: 
        # valid mode
        mean_nme, val_loss = validate(valid_loader,
                                      save=os.path.join(f'{C.save_image_path}',
                                                        f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))

        torch.save(flmk_model.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_flmk_{epoch}.pt"))
        if angl_model:        
            torch.save(angl_model.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_angl_{epoch}.pt"))

        # update nme
        if mean_nme < best_nme:
            best_nme = mean_nme
        
        # early-stopping part
        if val_loss < best_loss:
            early_cnt = 0
            best_loss = val_loss
            logging(f"           >> Saving model..   Best_NME : {best_nme:.4f}")
            
            torch.save(flmk_model.state_dict(),
                       os.path.join(f"/data/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_pfld_best.pt"))
            if angl_model:        
                torch.save(angl_model.state_dict(),
                       os.path.join(f"/data/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_angle_best.pt"))
        else:
            early_cnt += 1
            logging(f"           >> Early stopping cnt... {early_cnt}  Best_NME : {best_nme:.4f}")
            if early_cnt >= C.experiment["early_stop"]:
                break

        # save log
        df = pd.DataFrame(log_list)
        df.to_csv(f"{C.save_path}/{C.log_dirs}_logs.csv", index=None, header=None)
        
logging(f"best mean nme is : {best_nme:.4f}")
logging('Training Complete')
logging("Total Elapsed Time : {} s".format(time.time()-start_time))    

df = pd.DataFrame(log_list)
df.to_csv(f"{C.save_path}/{C.log_dirs}_logs.csv", index=None, header=None)