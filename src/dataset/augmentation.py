import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(data_type):    
    if data_type=="train":
        return A.Compose([
        A.Resize(width = 112,height = 112),
        ToTensorV2(),
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )
    
    elif data_type=="valid":
        return A.Compose([ 
        A.Resize(height=112,width=112),
        ToTensorV2(),    
        ],keypoint_params = A.KeypointParams(format="xy",remove_invisible = False)
    )