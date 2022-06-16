import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.dataset.kface_transform import *

class kfacedataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.label_list[idx]
        label = pd.read_csv(label).values.tolist()
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label