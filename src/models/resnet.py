import torch.nn as nn
from torchvision import models

from src import config as C

class resnet18(nn.Module):
    def __init__(self,num_classes=54):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
    
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        
        return x