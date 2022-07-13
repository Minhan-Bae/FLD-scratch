import warnings
warnings.filterwarnings("ignore")

import torch
import timm

def timm_Net_54(model_name, pretrained=None, num_classes=54):
    model = timm.create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        )

    if not pretrained==None:
        model.eval()
        model.load_state_dict(torch.load(pretrained, map_location = 'cpu'), strict=False)
        
    return model
