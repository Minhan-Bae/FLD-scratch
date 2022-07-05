# Import Moudles and Packages
import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import random
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

# Import pytorch modules
import torch
import torch.nn as nn
from torch.utils import data

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from madgrad import MADGRAD

import gc
import torch

from mpl_toolkits.axes_grid1 import ImageGrid

gc.collect()
torch.cuda.empty_cache()

from config import *
from utils.fix_seed import *
from utils.visualize import *
from dataset.dataloader import *
from metric.nme import NME
from models.timm_swin import timm_Net

seed_everything(SEED)

# Set dataloader
train_loader, valid_loader = dataloader(batch_size=BATCH_SIZE, workers=WORKERS)

def validate(save = None):
    cum_loss = 0.0
    cum_nme = 0.0
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for idx, (features, labels) in pbar:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with autocast(enabled=True):
            outputs = MODEL(features)
            loss = LOSS(outputs, labels)
            nme = NME(outputs, labels)
        
        cum_loss += loss.item()
        cum_nme += nme.item()
        
        description_valid = f"| Loss: {cum_loss/len(valid_loader):.4f}, NME: {cum_nme/len(valid_loader)*100:.4f}"
        pbar.set_description(description_valid)
        
    visualize_batch(features[:4].cpu(), outputs[:4].cpu(), labels[:4].cpu(),
                shape = (2, 2), size = 16, title = 'Validation sample predictions', save = save)
    
    return cum_loss/len(valid_loader), cum_nme/len(valid_loader)*100

batches = len(train_loader)
best_loss = np.inf
OPTIMIZER.zero_grad()

start_time = time.time()
for epoch in range(EXP["EPOCH"]):
    cnt = 0
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
        
        description_train = f"| # Epoch: {epoch+1}, Loss: {cum_loss/len(train_loader):.4f}"
        pbar.set_description(description_train)        
        
    val_loss, nme_value = validate(os.path.join(f'{SAVE_IMAGE_PATH}',
                                     f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))

    if val_loss < best_loss:
        cnt = 0
        best_loss = val_loss
        print('Saving model....................')
        torch.save(MODEL.state_dict(), SAVE_MODEL)
        
    else:
        cnt += 1
        print(f"Early stopping cnt... {cnt}")
        
    if cnt > 10:
        break
    
    print(f'Epoch({epoch + 1}/{EXP["EPOCH"]}) -> Loss: {val_loss:.8f} | nme: {nme_value:.8f}')

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    