import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(data_type):    
    if data_type=="train":
        return A.Compose([
        A.Rotate(limit = 15,border_mode =0, p=0.8),
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