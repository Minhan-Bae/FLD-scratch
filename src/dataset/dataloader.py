from dataset import augmentation, dataset
from torch.utils import data

def kfacedataloader(batch_size, workers):
    k_dataset_train = []
    for _ in range(10):
        aug_data = dataset.kfacedataset(
            type="train",
            transform=augmentation.get_augmentation(data_type="train")
            )
        k_dataset_train += aug_data

    k_dataset_valid = dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=64, shuffle=False, num_workers=workers) #TODO 확인을 위해 shuffle=True 로 변경
    
    return train_loader, valid_loader