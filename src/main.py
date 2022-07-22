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
train_loader, valid_loader_aflw, valid_loader_face = Dataloader(batch_size=C.experiment["batch_size"],
                                            workers=C.experiment["workers"])

# load model
devices_id = [int(d) for d in C.device.split(',')]
torch.cuda.set_device(devices_id[0])

model = C.xception_Net

C.optimizer.zero_grad()

# set validate
def validate(types, valid_loader, save = None):
    model.eval()
    
    nme = []
    losses = []
    
    with torch.no_grad():
        for img, landmark_gt in valid_loader:
            img = img.cuda()
            landmark_gt = landmark_gt.cuda()
            
            model.cuda()
            
            landmark = model(img)
                
            mean_nme = NME(landmark, landmark_gt)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            
            nme.append(mean_nme)
            losses.append(loss.cpu().numpy())
            
        visualize_batch(img[:16].cpu(), landmark[:16].cpu(), landmark_gt[:16].cpu(),
                    shape = (4, 4), size = 16, title = None, save = save)

    logging(f"|     ===> Evaluate {types}:")
    logging(f'|          Eval set: Normalize Mean Error: {np.mean(nme):.4f}')
    logging(f'|          Eval set: Average loss: {np.mean(losses):.4f}')    
    return np.mean(nme), np.mean(losses)

# run
logging("Start train")
for epoch in range(1, C.experiment["epoch"]+1):
    cum_loss = 0.0 # define current loss
    scaler = GradScaler() 
    
    # train mode
    model.train()
    
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, landmarks_gt) in pbar:
        
        features = features.cuda()
        landmarks_gt = landmarks_gt.cuda()

        with autocast(enabled=True):         
               
            predicts = model(features)
                
            loss = C.criterion(predicts, landmarks_gt)
    
        scaler.scale(loss).backward()
        scaler.step(C.optimizer)
        scaler.update(loss.item())
        
        C.optimizer.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}"
        pbar.set_description(description_train)
        
    
    C.scheduler.step(cum_loss)
    
    logging(f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.4f}")

    if epoch%C.validation_term == 0: 
        # valid mode
        k_nme, k_loss = validate(types='kface',
                                      valid_loader=valid_loader_face,
                                      save=os.path.join(f'{C.save_image_path}/kface',
                                                        f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))
        a_nme, a_loss = validate(types='aflw',
                                      valid_loader=valid_loader_face,
                                      save=os.path.join(f'{C.save_image_path}/aflw',
                                                        f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))
        ratio = [398/(398+387), 387/(398+387)]
        mean_nme = a_nme*ratio[0]+k_nme*ratio[1]
        val_loss= a_loss*ratio[0]+k_loss*ratio[1]
        
        torch.save(model.module.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_flmk_{epoch}.pt"))

        # update nme
        if mean_nme < best_nme:
            best_nme = mean_nme
        
        # early-stopping part
        if val_loss < best_loss:
            early_cnt = 0
            best_loss = val_loss
            logging(f"           >> Saving model..   Best_NME : {best_nme:.4f}")
            
            torch.save(model.module.state_dict(),
                       os.path.join(f"/data/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_pfld_best.pt"))
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