import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from vis_src.model import *
from vis_src.transform import *

def visualization(image, crop_img, landmarks, w, h, save=False):
    plt.figure(figsize=(22,44))
    plt.subplot(1,2,1)
    plt.axis("off")

    # visualization raw image
    plt.imshow(image)

    plt.subplot(1,2,2)
    plt.axis("off")
    for idx in range(len(landmarks)):
        plt.scatter(landmarks[idx][0]*w,landmarks[idx][1]*h,s=20,c='dodgerblue',marker='X')
        
        plt.annotate(idx, (landmarks[idx][0]*w,landmarks[idx][1]*h))
        plt.imshow(crop_img)
    
    if save:
        plt.savefig(save)

# TIMM swin 
def swin_run(image_path, img_size=224,
        model_name = "swin_base_patch4_window7_224",
        pretrained = "/data/komedi/logs/2022-07-15/swin_v15_nme_42/v15_swin_base_patch4_window7_224_best.pt",
        save=False):

    image = Image.open(image_path)
    
    image_tensor, crop_img, h, w = transform(image, img_size)
    
    model = timm_Net_54(model_name = model_name,
                        pretrained = pretrained)
    
    predict = model(image_tensor)

    landmarks = ((predict.view(-1,2)+0.5)).detach().numpy().tolist()
    landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])
    
    visualization(image, crop_img, landmarks, w, h, save)

# PFLD
def pfld_run(image_path, img_size=112,
        pretrained = "/data/komedi/logs/2022-07-18/pfld_ver1/ver1_pfld_best.pt",
        save=False):

    image = Image.open(image_path)
    
    image_tensor, crop_img, h, w = transform(image)
    
    model = PFLDInference()
    
    checkpoint = torch.load(pretrained, map_location = 'cpu')
    model.load_state_dict(checkpoint,strict=False)
    
    predict = model(image_tensor)

    landmarks = ((predict.view(-1,2)+0.5)).detach().numpy().tolist()
    landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])
    
    visualization(image, crop_img, landmarks, w, h, save)