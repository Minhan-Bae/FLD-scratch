import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.dataset.kface_transform import *

class kfacedataset(Dataset):
    def __init__(self, image_list, label_list, type="train", transform=None):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.type = type
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx],0)
        
        label = self.label_list[idx]
        label = pd.read_csv(label).values.tolist()
        label = np.array(label).astype("float32")
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=label)
            image = transformed['image']
            label = transformed['keypoints']
            
                
        image = torch.tensor(image, dtype=torch.float)
        image /= 255
        label = torch.tensor(label, dtype=torch.float)
        label /= 512
        
        return image, label