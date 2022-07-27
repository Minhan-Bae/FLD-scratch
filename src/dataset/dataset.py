import cv2
import torch
import random
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
        random.shuffle(self.data_list)
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
            x_list.append(x)
            y_list.append(y)

        image = cv2.imread(data[2],0)
        pil_image = Image.fromarray(image)
        offset = 10
        crops = [np.min(x_list)-offset, 
                 np.min(y_list)-offset,
                 np.max(x_list)+offset,
                 np.max(y_list)+offset]
        image = pil_image.crop(crops)
        image = np.array(image).astype(float)

        for label in data[10:]:
            x,y = eval(label[1:-1])
            landmarks.append([x-np.min(x_list)+offset,y-np.min(y_list)+offset])
        landmarks = np.array(landmarks).astype(float)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks)
            image = transformed['image']
            landmarks = transformed['keypoints']
            
        image = torch.tensor(image, dtype=torch.float)
        image = (image - image.min())/(image.max() - image.min())
        image = (2 * image) - 1 # set image value (-1, 1)
        
        
        label = torch.tensor(landmarks, dtype=torch.float)
        label /= 128
        landmark = label.reshape(-1) - 0.5
        
        return image, landmark