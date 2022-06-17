# Import Moudles and Packages
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"]=False

import torch
import torch.nn as nn
from torch.utils import data as D

from pytorch_toolbelt import losses
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from madgrad import MADGRAD

from src import config as C
from src.models import hrnet, resnet
from src.dataset import kface_dataset as K
from src.dataset import kface_transform as T
from src.utils.collate_fn import *
from src.utils.print_overwrite import *

print(f"| Number of image : {len(C.IMAGE_LIST)}")
print(f"| Number of label : {len(C.LABEL_LIST)}")

print(f"| Number of train : {C.LEN_TRAIN_SET}")
print(f"| Number of valid : {C.LEN_VALID_SET}")

dataset = K.kfacedataset(image_list=C.IMAGE_LIST, label_list=C.LABEL_LIST, transform=T.Transforms())

train_dataset, valid_dataset = D.random_split(dataset, [C.LEN_TRAIN_SET, C.LEN_VALID_SET])

train_loader = D.DataLoader(train_dataset, batch_size=C.BATCH_SIZE["TRAIN"], shuffle=True, num_workers=C.WORKERS)
valid_loader = D.DataLoader(valid_dataset, batch_size=C.BATCH_SIZE["VALID"], shuffle=True, num_workers=C.WORKERS)

train_images, train_landmarks = next(iter(train_loader))
valid_images, valid_landmarks = next(iter(valid_loader))

print(f"| Size of image in train_loader : {train_images.shape}")
print(f"| Size of label in train_loader : {train_landmarks.shape}")
print(f"| Size of image in train_loader : {valid_images.shape}")
print(f"| Size of label in train_loader : {valid_landmarks.shape}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = resnet.Network()
model_dict = model.state_dict()
pretrained_dict = model.load_state_dict(torch.load(C.PRETRAINED_ROOT))
# for layer in pretrained_dict:
#     print(layer, '\t', pretrained_dict[layer].size())
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict, strict=False)

x = torch.randn([1, 1, 567, 864])
out = model(x)
print(f"input : {x.shape} | output : {out.size()}")