import timm

import torch
import torch.nn as nn

def timm_Net_54(model_name, pretrained=None, num_classes=27*2):
    model = timm.create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        dropout_rate=0.2
        )

    if not pretrained==None:
        model.eval()
        model.load_state_dict(torch.load(pretrained, map_location = 'cpu'), strict=False)
            
    return model