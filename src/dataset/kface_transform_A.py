import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(data_type):    
    if data_type=="train":
        return A.Compose([
                A.Resize(224,224),
                # A.Rotate(limit=10, p=0.5),
                A.HorizontalFlip(p=0.5), # 좌, 우 데이터 존재로 인해 제거
                # A.Normalize(),
                A.OneOf([
                    A.HueSaturationValue(p=0.5), 
                    # A.RGBShift(p=0.7)
                ], p=1),                          
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2()
            ], 
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
        )
    elif data_type=="valid":
        return A.Compose([
            A.Resize(224,224),
            # A.Normalize(),
            ToTensorV2()
            ],
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
        )