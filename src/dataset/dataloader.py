from dataset import augmentation, aug_dataset
from torch.utils import data

def kfacedataloader(batch_size, workers):
    k_dataset_train = aug_dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        ) # normalize만 넣어서,

    k_dataset_valid = aug_dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=64, shuffle=False, num_workers=workers) #TODO 확인을 위해 shuffle=True 로 변경
    
    return train_loader, valid_loader