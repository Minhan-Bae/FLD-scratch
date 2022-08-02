import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(data_type):    
    if data_type=="train":
        return A.Compose([
        # A.Rotate(limit = 15,border_mode =0, p=0.8),
        # A.RandomBrightnessContrast(contrast_limit=0.5,brightness_limit=0.5,p=0.2),
        # A.CLAHE(p=0.8),
        # A.OneOf([
        #     A.GaussNoise(p=0.8),
        #     A.ImageCompression(p=0.8),
        #     A.RandomGamma(p=0.8),
        #     A.Posterize(p=0.8),
        #     A.Blur(p=0.8),
        # ],p=1.0),
        # A.OneOf([
        #     A.GaussNoise(p=0.8),
        #     A.ImageCompression(p=0.8),
        #     A.RandomGamma(p=0.8),
        #     A.Posterize(p=0.8),
        #     A.Blur(p=0.8),
        # ],p=1.0),
        # # A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.5, 0),rotate_limit=30, border_mode=0 ,p=0.8,),
        A.Resize(width = 128,height = 128),
        A.ToGray(always_apply=True),
        ToTensorV2(),
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )
    
    elif data_type=="valid":
        return A.Compose([ 
        A.Resize(height=128,width=128),
        A.ToGray(always_apply=True),
        ToTensorV2(),    
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )
        