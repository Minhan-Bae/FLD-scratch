import torch
import torch.nn as nn
from torchvision import models

from src import config as C

class Network(nn.Module):
    def __init__(self,num_classes=54):
        super().__init__()
        self.model_name='resnext50_32x4d'
        self.model=models.resnext50_32x4d(pretrained=True)
    
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
# class Swin(nn.Module):
#     def __init__(self,num_classes=54):
#         super().__init__()
#         self.model_name = "swin"
#         self.model=models.