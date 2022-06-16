import torch
import torch.nn as nn
from torchvision import models

from src import config as C

class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        first_conv = nn.Conv2d(3, 1, (1,1), 1, 1)
        fc=nn.Linear(136, 54)

        x = first_conv(x)
        x = self.model(x)
        x = fc(x)
        return x