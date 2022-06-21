import torch.nn as nn
import timm

class FaceSynthetics(nn.Module):
    def __init__(self, num_classes=27*2):
        super().__init__()
        self.model_name="resnet50d"
        self.model = timm.create_model(
            model_name=self.model_name,
            pretrained=True,
            num_classes=num_classes
            )

    def forward(self, x):
        x = self.model(x)
        
        return x
