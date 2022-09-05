import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

"""
Dataset configuration(0831 new)

idx 0: file case (kface, nc)
idx 1: aug case (normal, blur, rotate, ...)
idx 2: file name (image.jpg)
idx 3: file path
idx 4 ~ 7: bbox (left, top, width, height)
idx 8 ~ : landmark
"""

class Datasets(Dataset):
    def __init__(self, args, type="train", transform=None):
        super().__init__()
        self.type = type
        self.transform = transform
        
        self.image_dir = args.image_dir
        self.kface_image_dir = args.image_dir+"/kface"            
        self.aflw_image_dir = args.image_dir+"/aflw"
        
        self.data_list = pd.read_csv(args.train_csv_path).values.tolist()
        random.shuffle(self.data_list)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        offset = (random.randint(1,50),random.randint(1,50))
        crops = [data[3]-offset[0], 
                 data[4]-offset[1],
                 data[5]+offset[0],
                 data[6]+offset[1]]
        
        if data[1] == "kface":
            image_dir = self.kface_image_dir
        else:
            image_dir = self.aflw_image_dir
        
        image = cv2.imread(os.path.join(image_dir,data[0]),0) # set 1 channel
        pil_image = Image.fromarray(image)
        
        image = pil_image.crop(crops)
        image = np.array(image)
        
        labels = np.array(data[7:]).reshape(-1,2)
        landmarks = []
        for x,y in labels:
            landmarks.append([x-crops[0], y-crops[1]])

        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks)
            image = transformed['image']
            landmarks = transformed['keypoints']
            
        image = torch.tensor(image, dtype=torch.float)
        image = (image - image.min())/(image.max() - image.min()) # set image value (0, 1)
        image = (2 * image) - 1 # set image value (-1, 1)
        
        label = torch.tensor(landmarks, dtype=torch.float)
        label /= 128 # set landmark value (0,1)
        landmark = label.reshape(-1) - 0.5 # set landmark value(-1, 1)
        
        return image, landmark