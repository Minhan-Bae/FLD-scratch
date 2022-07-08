from dataset import augmentation, dataset, lateral_dataset, frontal_dataset, dataset_axis, frontal_dataset_pt27, lateral_dataset_pt27
from torch.utils import data

def lateral_dataloader(batch_size, workers):
    k_dataset_train = lateral_dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = lateral_dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader

def frontal_dataloader(batch_size, workers):
    k_dataset_train = frontal_dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = frontal_dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader

def dataloader(batch_size, workers):
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
        k_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader

def axis_dataloader(batch_size, workers):
    k_dataset_train = dataset_axis.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    k_dataset_valid = dataset_axis.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader
def frontal_dataloader_27(batch_size, workers):
    k_dataset_train = frontal_dataset_pt27.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = frontal_dataset_pt27.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    len_train = int(len(k_dataset_train)*0.8)
    len_valid = len(k_dataset_train)-len_train
    
    train_dataset, _ = data.random_split(k_dataset_train, [len_train, len_valid])
    _, valid_dataset = data.random_split(k_dataset_valid, [len_train, len_valid])
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader
def lateral_dataloader_27(batch_size, workers):
    k_dataset_train = lateral_dataset_pt27.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = lateral_dataset_pt27.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    len_train = int(len(k_dataset_train)*0.8)
    len_valid = len(k_dataset_train)-len_train
    
    train_dataset, _ = data.random_split(k_dataset_train, [len_train, len_valid])
    _, valid_dataset = data.random_split(k_dataset_valid, [len_train, len_valid])
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader