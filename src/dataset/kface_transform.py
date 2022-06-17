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

    # def rotate(self, image, landmarks, angle):
    #     angle = random.uniform(-angle, +angle)

    #     transformation_matrix = torch.tensor([
    #         [+cos(radians(angle)), -sin(radians(angle))], 
    #         [+sin(radians(angle)), +cos(radians(angle))]
    #     ])

    #     image = imutils.rotate(np.array(image), angle)

    #     landmarks = landmarks - 0.5
    #     new_landmarks = np.matmul(landmarks, transformation_matrix)
    #     new_landmarks = new_landmarks + 0.5
    #     return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    # def crop_face(self, image, landmarks):

    #     img_shape = np.array(image).shape
    #     # landmarks = torch.tensor(landmarks) - torch.tensor([img_shape[1], img_shape[0]])
    #     landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
    #     return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks   


    def __call__(self, image, landmarks):
        image = Image.fromarray(image)
        landmarks = torch.tensor(landmarks)
        
        image, landmarks = self.resize(image, landmarks, (224, 224))
        # image, landmarks = self.crop_face(image, landmarks)
        image, landmarks = self.color_jitter(image, landmarks)
        # image, landmarks = self.rotate(image, landmarks, angle=10)

        landmarks = landmarks * torch.tensor((1/874, 1/576))
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks