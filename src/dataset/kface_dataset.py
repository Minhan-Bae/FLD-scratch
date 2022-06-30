import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class kfacedataset(Dataset):
    def __init__(self, data_path = "/data/komedi/dataset/540_kface_cropped_noresize/labels/535_27pt_kface.csv", type="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.data_list = pd.read_csv(self.data_path,header=None).values.tolist()
        self.type = type
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # image = cv2.cvtColor(cv2.imread(self.data_list[idx][1],0),cv2.COLOR_BGR2RGB)
        image = cv2.imread(self.data_list[idx][1],0)
        
        labels = self.data_list[idx][2:]
        label_list = []
        for label in labels:
            x,y = eval(label[1:-1])
            label_list.append((x,y))
        label = np.array(label_list)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=label)
            image = transformed['image']
            label = transformed['keypoints']
            
                
        image = torch.tensor(image, dtype=torch.float)
        image = (2 * image) - 1
        image /= 255
        label = torch.tensor(label, dtype=torch.float)
        label /= 224
        label = label.reshape(-1) - 0.5
        
        return image, label