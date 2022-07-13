import cv2
import torch
import numpy as np
import pandas as pd
import random

from PIL import Image

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
            self.data_path = "/home/ubuntu/workspace/FLD-scratch/src/data/kface_w300_train_v3.csv"
        else:
            self.data_path = "/home/ubuntu/workspace/FLD-scratch/src/data/kface_w300_valid_v3.csv"
        
        self.transform = transform
        self.data_list = pd.read_csv(self.data_path,header=None).values.tolist()

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data_list = self.data_list
        # random.shuffle(data_list)
                
        image = cv2.imread(data_list[idx][3])
        # if image.shape(3)==4:
        #     image = image[:,:,:-1]
        margin = 200
        crop_area = (data_list[idx][4]-margin//2, # get bbox area with margin
                    data_list[idx][5]-margin//2,
                    data_list[idx][6]+margin//2,
                    data_list[idx][7]+margin//2)

        pil_image = Image.fromarray(image)
        image = pil_image.crop(crop_area)
        image = np.array(image)
        
        labels = data_list[idx][8:]
        label_list = []
        for label in labels:
            x,y = eval(label[1:-1])
            # if self.type=="train":
            #     x = x-(data_list[idx][4]-margin//2)
            #     y = y-(data_list[idx][5]-margin//2)
            label_list.append((x,y))
        label = np.array(label_list)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=label)
            image = transformed['image']
            label = transformed['keypoints']
            
        image = torch.tensor(image, dtype=torch.float)
        image = (2 * image) - 1 #TODO 07M-11D-10H-10M
        image /= 255
        
        label = torch.tensor(label, dtype=torch.float)
        label /= 224
        label = label.reshape(-1) - 0.5
        
        return image, label