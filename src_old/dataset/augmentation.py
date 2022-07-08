import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_augmentation(data_type):    
    if data_type=="train":
        return A.Compose([
        A.Resize(width = 224,height = 224),
        A.Rotate(limit = 15,border_mode = cv2.BORDER_CONSTANT,p=0.8),
        # A.IAAAffine(shear = 15,scale = 1.0,mode = 'constant',p = 0.2),
        A.RandomBrightnessContrast(contrast_limit=0.5,brightness_limit=0.5,p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ],p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ],p=1.0),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.5, 0),rotate_limit=30, border_mode=0 ,p=0.8,),
        A.Normalize(
            mean=[0.4897,0.4897,0.4897],
            std = [0.2330,0.2330,0.2330],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )
    
    elif data_type=="valid":
        return A.Compose([ 
        A.Resize(height=224,width=224),
        A.Rotate(limit = 15,border_mode=0,p=0.8),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.5, 0),rotate_limit=30, border_mode=0 ,p=0.8,),
        A.Normalize(
            mean=[0.4897,0.4897,0.4897],
            std = [0.2330,0.2330,0.2330],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),    
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )
        