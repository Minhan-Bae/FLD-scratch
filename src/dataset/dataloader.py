from dataset import augmentation, dataset
from torch.utils import data

def kfacedataloader(batch_size, workers):
    k_dataset_train = dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=64, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader