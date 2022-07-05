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
    for features, labels in valid_loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with autocast(enabled=True):
            outputs = MODEL(features)
            loss = LOSS(outputs, labels)
            nme = NME(outputs, labels)
        
        cum_loss += loss.item()
        cum_nme += nme.item()
        break
    
    visualize_batch(features[:4].cpu(), outputs[:4].cpu(), labels[:4].cpu(),
                shape = (2, 2), size = 16, title = 'Validation sample predictions', save = save)
    
    return cum_loss/len(valid_loader), cum_nme/len(valid_loader)*100

batches = len(train_loader)
best_loss = np.inf
OPTIMIZER.zero_grad()

start_time = time.time()
for epoch in range(EXP["EPOCH"]):

    cum_loss = 0.0
    scaler = GradScaler() 
    
    MODEL.train()
    for idx, (features, labels) in enumerate(tqdm(train_loader, desc= 'Training')):
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
        
    val_loss, nme_value = validate(os.path.join(f'{SAVE_IMAGE_PATH}',
                                     f'epoch({str(epoch + 1).zfill(len(str(EXP["EPOCH"])))}).jpg'))

    if val_loss < best_loss:
        best_loss = val_loss
        print('Saving model....................')
        torch.save(MODEL.state_dict(), SAVE_MODEL)

    print(f'Epoch({epoch + 1}/{EXP["EPOCH"]}) -> Training Loss: {cum_loss/batches:.8f} | Validation Loss: {val_loss:.8f} | Validation Rsme: {nme_value:.8f}')

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))    