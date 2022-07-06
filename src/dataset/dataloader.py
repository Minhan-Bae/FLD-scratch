from dataset import augmentation, dataset, lateral_dataset, frontal_dataset
from torch.utils import data

def lateral_dataloader(batch_size, workers):
    k_dataset_train = lateral_dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    k_dataset_valid = lateral_dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, valid_loader

def frontal_dataloader(batch_size, workers):
    k_dataset_train = frontal_dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    k_dataset_valid = frontal_dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, valid_loader

def dataloader(batch_size, workers):
    k_dataset_train = dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    k_dataset_valid = dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, valid_loader