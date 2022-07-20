import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import math
"""
Dataset configuration

idx 0: file name
idx 1: file type(aflw)
idx 2: file path
idx 3 ~ 6: bbox(left, top, width, height)
idx 7 ~ 9: pose para
idx 10 ~ : landmark
"""

class AFLWDatasets(Dataset):
    def __init__(self, type="train", transform=None, aug_data_num=1):
        super().__init__()
        self.type = type
        
        if self.type == "train":
            self.data_path = "/home/ubuntu/workspace/FLD-scratch/src/data/train_df.csv"
        else:
            self.data_path = "/home/ubuntu/workspace/FLD-scratch/src/data/valid_df.csv"
        
        self.transform = transform
        self.data_list = pd.read_csv(self.data_path,header=None).values.tolist()
        
        # pluses dataset of train
        if self.type == "train":
            self.data_list *= aug_data_num

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data_list = self.data_list

        # read image
        margin = 200
        crop_area = (data_list[idx][3]-margin//2, # get bbox area with margin
                    data_list[idx][4]-margin//2,
                    data_list[idx][5]+margin//2,
                    data_list[idx][6]+margin//2)

        image = cv2.imread(data_list[idx][2])
        pil_image = Image.fromarray(image)
        
        image = pil_image.crop(crop_area)
        image = np.array(image)
        
        # read pose
        
        euler_angle = np.asarray([data_list[idx][i]/180 for i in (7,8,9)], dtype=np.float32)
        
        # read label
        labels = data_list[idx][10:]
        label_list = []
        for label in labels:
            x,y = eval(label[1:-1])
            label_list.append((x,y))
        label = np.asarray(label_list, dtype=np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=label)
            image = transformed['image']
            label = transformed['keypoints']
            
        image = torch.tensor(image, dtype=torch.float)
        image = (2 * image) - 1 # contrast 조절
        image /= 255
        
        label = torch.tensor(label, dtype=torch.float)
        label /= 112
        landmark = label.reshape(-1) - 0.5
        
        return (image, landmark, euler_angle)