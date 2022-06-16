import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

import imutils
import random
from math import *
from PIL import Image

class Transforms():
    def __init__(self):
        pass

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def __call__(self, image, landmarks):
        image = Image.fromarray(image)
        image, landmarks = self.color_jitter(image, landmarks)
        
        landmarks = torch.tensor(landmarks)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks