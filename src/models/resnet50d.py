import torch.nn as nn
import timm

class FaceSynthetics(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name="resnet50d"
        self.model = timm.create_model(
            model_name=self.model_name,
            pretrained=True,
            )

    def forward(self, x):
        x = self.model(x)
        
        return x
