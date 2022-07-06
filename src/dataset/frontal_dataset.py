import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

"""
Dataset configuration

0: image name
1: accessory type
2: light type
3: image path
4 ~ 7: bbox(left-top-x, left-top-y, left_low_x, left_low_y) # float
8 ~ : landmarks(x, y) # str

"""

class kfacedataset(Dataset):
    def __init__(self, type="train", transform=None):
        super().__init__()
        self.type = type
        if self.type == "train":
            self.data_path = "/data/komedi/dataset/split_frontal_train.csv"
        else:
            self.data_path = "/data/komedi/dataset/split_frontal_valid.csv"
        
        self.transform = transform
        self.data_list = pd.read_csv(self.data_path,header=None).values.tolist()


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        image = cv2.imread(self.data_list[idx][0])
        
        labels = self.data_list[idx][1:]
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