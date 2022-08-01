#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import gc
gc.collect()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.empty_cache()
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import autograd
# autograd.set_detect_anomaly(True)
import config as C
# from validate import *
from loss.loss import *
from utils.fix_seed import *
from utils.visualize import *
from utils.logging import *
from dataset.dataloader import *
from metric.nme import *

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

def validate(types, valid_loader, model, criterion, save = None):
    cum_loss = 0.0
    cum_nme = 0.0
    model.eval()
    with torch.no_grad():
        for features, labels in valid_loader:
            features = features.cuda()
            labels = labels.cuda()

            outputs = model(features).cuda()

            loss = custom_loss(outputs, labels)
            nme=NME(outputs, labels)
            
            cum_nme += nme.item()
            cum_loss += loss.item()
            break
            
    visualize_batch(features[:16].cpu(), outputs[:16].cpu(), labels[:16].cpu(), shape = (4, 4), size = 16, save = save)

    logging(f'|     ===> Evaluate {types}:')
    logging(f'|          Eval set: Normalize Mean Error: {cum_nme/len(valid_loader):.4f}')
    logging(f'|          Eval set: Average loss: {cum_loss/len(valid_loader):.8f}')    

    return cum_loss/len(valid_loader), cum_nme/len(valid_loader)

# Fix seed
seed_everything(C.experiment["seed"])

# check start time
start_time = time.time()
logging(f"exp model : {C.experiment['model']}     version : {C.log_dirs}")

# Set dataloader
logging("Set dataloader")
train_loader, valid_loader_aflw, valid_loader_face = Dataloader(batch_size=C.experiment["batch_size"],
                                            workers=C.experiment["workers"])

# load model
devices_id = [int(d) for d in C.device.split(',')]
torch.cuda.set_device(devices_id[0])

model = C.xception_Net.cuda()


criterion = torch.nn.MSELoss()

def custom_loss(preds, labels):
    loss1 = criterion(preds[:24], labels[:24])
    loss2 = criterion(preds[24:], labels[24:])
    # loss3 = criterion(preds[36:], labels[36:])
    return 0.8*loss1+0.2*loss2

optimizer = optim.Adam(model.parameters(), lr = C.experiment["lr"], weight_decay = 1e-6) 
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=40, verbose=True)

optimizer.zero_grad()

# init validate
validate(types='kface',
         valid_loader=valid_loader_face,
         model = model,
         criterion = criterion,
         save=os.path.join(f'{C.save_image_path}/kface',
                           f'epoch({str(0).zfill(len(str(C.experiment["epoch"])))}).jpg'))
validate(types='aflw',
         valid_loader=valid_loader_aflw,
         model = model,
         criterion = criterion,
         save=os.path.join(f'{C.save_image_path}/aflw',
                           f'epoch({str(0).zfill(len(str(C.experiment["epoch"])))}).jpg'))

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
            loss = custom_loss(predicts, landmarks_gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
        # loss = torch.nan_to_num(loss)
        
        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.8f}"
        pbar.set_description(description_train)
        
    scheduler.step(cum_loss/len(train_loader))
    log_list.append(f"| # Epoch: {str(epoch).zfill(len(str(C.experiment['epoch'])))}/{C.experiment['epoch']}, Loss: {cum_loss/(idx+1):.8f}")

    if epoch%C.validation_term == 0: 
        # valid mode
        k_loss, k_nme = validate(types='kface',
                                 valid_loader=valid_loader_face,
                                 model = model,
                                 criterion = criterion,
                                 save=os.path.join(f'{C.save_image_path}/kface',
                                                   f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))
        a_loss, a_nme = validate(types='aflw',
                                 valid_loader=valid_loader_aflw,
                                 model = model,
                                 criterion = criterion,
                                 save=os.path.join(f'{C.save_image_path}/aflw',
                                                   f'epoch({str(epoch).zfill(len(str(C.experiment["epoch"])))}).jpg'))

        ratio = [len(valid_loader_aflw)/(len(valid_loader_aflw)+len(valid_loader_face)),
                 len(valid_loader_face)/(len(valid_loader_aflw)+len(valid_loader_face))]
        mean_nme = a_nme*ratio[0]+k_nme*ratio[1]
        val_loss= a_loss*ratio[0]+k_loss*ratio[1]
        
        torch.save(model.module.state_dict(), os.path.join(C.save_model_path, f"{C.log_dirs}_flmk_{epoch}.pt"))

        # early-stopping part
        if mean_nme < best_nme:
            best_nme = mean_nme
            early_cnt = 0
            logging(f"           >> Saving model..   Best_NME : {best_nme:.4f}")
            torch.save(model.module.state_dict(),
                       os.path.join(f"/data/komedi/komedi/logs/{C.experiment['day']}/{C.experiment['model']}_{C.log_dirs}", f"{C.log_dirs}_best.pt"))
        else:
            early_cnt += 1
            logging(f"           >> Early stopping cnt... {early_cnt}  Best_NME : {best_nme:.4f}")
            if early_cnt >= C.experiment["early_stop"]:
                logging("Early stop")
                break

    # save log
    df = pd.DataFrame(log_list)
    df.to_csv(f"{C.save_path}/{C.log_dirs}_logs.csv", index=None, header=None)

# Finish
logging(f"best mean nme is : {best_nme:.4f}")
logging('Training Complete')
logging("Total Elapsed Time : {} s".format(time.time()-start_time))    

df = pd.DataFrame(log_list)
df.to_csv(f"{C.save_path}/{C.log_dirs}_logs.csv", index=None, header=None)