import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
"""
Dataset configuration

idx 0: file name
idx 1: file type(aflw)
idx 2: file path
idx 3 ~ 6: bbox(left, top, width, height)
idx 7 ~ 9: pose para
idx 10 ~ : landmark
"""

class Datasets(Dataset):
    def __init__(self, data_path, type="train", transform=None, aug_data_num=1):
        super().__init__()
        self.type = type
        self.transform = transform
        self.data_list = pd.read_csv(data_path,header=None).values.tolist()
        
        # (Optional) add dataset of train
        if self.type == "train":
            self.data_list *= aug_data_num

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        landmarks = []
        x_list, y_list = [], []
        for label in data[10:]:
            x,y = eval(label[1:-1])
            if self.type=='valid': 
                x = int(x+data[3])
                y = int(y+data[4])
            x_list.append(x)
            y_list.append(y)
            landmarks.append([x,y])
        landmarks = np.array(landmarks).astype('float32')
        

        image = cv2.imread(data[2],0)
        pil_image = Image.fromarray(image)
        crops = [np.min(x_list), 
                 np.min(y_list),
                 np.max(x_list),
                 np.max(y_list)]
        image = pil_image.crop(crops)
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks)
            image = transformed['image']
            landmarks = transformed['keypoints']
            
        image = torch.tensor(image, dtype=torch.float)
        image = (2 * image) - 1 # contrast 조절
        image /= 255
        
        label = torch.tensor(landmarks, dtype=torch.float)
        label /= 128
        landmark = label.reshape(-1) - 0.5
        
        return image, landmark