# Import Moudles and Packages
import warnings
warnings.filterwarnings("ignore")

import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from config import *
from utils.fix_seed import *
from utils.visualize import *
from dataset.dataloader import *
from metric.nme import *

# Fix seed
seed_everything(SEED)

# Set dataloader
train_loader, valid_loader = kfacedataloader(batch_size=BATCH_SIZE, workers=WORKERS)

def validate(save = None):
    cum_loss = 0.0
    cum_mean_nme = 0.0
    cum_std_nme = 0.0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for idx, (features, labels) in pbar:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with autocast(enabled=True):
            outputs = MODEL(features)
            loss = LOSS(outputs, labels)
            mean_nme, std_nme = NME(outputs, labels)
        
        cum_loss += loss.item()
        cum_mean_nme += mean_nme.item()
        cum_std_nme += std_nme.item()
        
        
        description_valid = f"| mean_nme: {cum_mean_nme/(idx+1):.4f}, std_nme: {cum_std_nme/(idx+1):.4f}"
        pbar.set_description(description_valid)
        
    visualize_batch(features[:16].cpu(), outputs[:16].cpu(), labels[:16].cpu(),
                shape = (4, 4), size = 16, title = 'Validation sample predictions', save = save)
    
    return cum_loss/len(valid_loader), cum_mean_nme/len(valid_loader)

early_cnt = 0
batches = len(train_loader)
best_loss = np.inf
best_nme = np.inf
OPTIMIZER.zero_grad()
loss_list = []

start_time = time.time()
for epoch in range(EXP["EPOCH"]):
    cum_loss = 0.0
    scaler = GradScaler() 
    
    MODEL.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for idx, (features, labels) in pbar:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast(enabled=True):
            MODEL = MODEL.to(DEVICE)
            
            outputs = MODEL(features)
        
            loss = LOSS(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update()
                
        OPTIMIZER.zero_grad()

        cum_loss += loss.item()
        
        description_train = f"| # Epoch: {epoch+1}"
        pbar.set_description(description_train)   
     
    MODEL.eval()
    val_loss, nme_value = validate(os.path.join(f'{SAVE_IMAGE_PATH}',
                                     f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))
    loss_list.append((epoch, val_loss, nme_value))

    if nme_value < best_nme:
        early_cnt = 0
        best_nme = nme_value
        print('Saving model....................')
        torch.save(MODEL.state_dict(), SAVE_MODEL)
    else:
        early_cnt += 1
        print(f"Early stopping cnt... {early_cnt}")
        
    if early_cnt >= EARLY_STOPPING_CNT:
        break
    print(f'Epoch({epoch + 1}/{EXP["EPOCH"]}) -> Loss: {val_loss:.8f} | nme: {nme_value:.8f}')

    df = pd.DataFrame(loss_list)
    df.to_csv(f"{SAVE_IMAGE_PATH}/lateral_raw_data_validation.csv", index=None, header=None)
print(f"best mean nme is : {best_nme}")
print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    