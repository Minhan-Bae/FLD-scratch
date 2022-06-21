import gc
import os

import time
import random

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from madgrad import MADGRAD

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

print("\n| Pytorch version: {}".format(torch.__version__))
print("| GPU: {}".format(torch.cuda.is_available()))
print("| Device : ",device)
print("| Device name: ", torch.cuda.get_device_name(0))
print("| Device count: ", torch.cuda.device_count())

torch.cuda.empty_cache()
gc.collect()

# Import local modules
from src import config as C
from src.models import hrnet, resnet, basenet

from src.utils.collate_fn import *
from src.utils.print_overwrite import *
from src.utils.seed import *

from src.models import resnet50d

from src.dataset.kface_dataset import *
from src.dataset.album_transform import *
from src.dataset.w300_dataset import *

seed_everything(C.SEED)

print(f"\n| Get Dataset")
print(f"|   Number of image : {len(C.IMAGE_LIST)}")
print(f"|   Number of label : {len(C.LABEL_LIST)}")
print(f"|   Number of trainset : {C.LEN_TRAIN_SET}")
print(f"|   Number of validset : {C.LEN_VALID_SET}")

w_dataset = FaceLandmarksDataset(Transforms())

len_valid_set = int(0.2*len(w_dataset))
len_train_set = len(w_dataset) - len_valid_set

train_dataset , valid_dataset,  = torch.utils.data.random_split(w_dataset , [len_train_set, len_valid_set])

# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

train_images, train_landmarks = next(iter(train_loader))
valid_images, valid_landmarks = next(iter(valid_loader))

print(f"\n| Get Dataloader")
print(f"|   Size of image in train_loader : {train_images.shape}")
print(f"|   Size of label in train_loader : {train_landmarks.shape}")
print(f"|   Size of image in train_loader : {valid_images.shape}")
print(f"|   Size of label in train_loader : {valid_landmarks.shape}")


model = resnet50d()
criterion = nn.L1Loss(reduction='mean')
optimizer = torch.optim.SGD(params=model.parameters(), lr =C.LEARNING_RATE, momentum=0.9, weight_decay = 0.0005)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=C.EPOCHS, T_mult=1)

loss_min = np.inf

print(f"\n| Start training...")

start_time = time.time()
for epoch in range(C.EPOCHS):
    
    model.train()
    
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    scaler = GradScaler()    
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, landmarks) in pbar:

        landmarks = landmarks.view(landmarks.size(0),-1)
        
        images = images.to(device)
        landmarks = landmarks.to(device)
        
        with autocast(enabled=True):
            model = model.to(device)
            
            predictions = model(images)
            # prediction = predictions.squeeze()
            loss_train_step = criterion(predictions, landmarks)
        
        scaler.scale(loss_train_step).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
                
        loss_train += loss_train_step
        running_loss = loss_train/(step+1)
        
        description = f"|    # Train-Epoch : {epoch + 1} Loss : {(running_loss):.4f}"
        pbar.set_description(description)
        
    with torch.no_grad():
        
        model.eval() 
        
        pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, (images, landmarks) in pbar_valid:
                  
            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0),-1).to(device)
        
            predictions = model(images).to(device)
                    
            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step
            running_loss = loss_valid/(step+1)

            description = f"|    # Valid-Epoch : {epoch + 1} Loss : {(running_loss):.4f}"
            pbar_valid.set_description(description)
            
            
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(), '/home/ubuntu/workspace/FLD-scratch/result/face_landmarks.pth') 
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, C.EPOCHS))
        print('Model Saved\n')

print('| Training Complete')
print("| Total Elapsed Time : {} s".format(time.time()-start_time))