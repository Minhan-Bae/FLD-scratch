from dataset import augmentation, dataset
from torch.utils import data

def kfacedataloader(batch_size, workers):
    k_dataset_train = dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        ) # normalize만 넣어서,

    k_dataset_valid_kface = dataset.kfacedataset(
        type="valid_kface",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    k_dataset_valid_aflw = dataset.kfacedataset(
        type="valid_aflw",
        transform=augmentation.get_augmentation(data_type="valid")
        )

    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    valid_loader_kface = data.DataLoader(
        k_dataset_valid_kface, batch_size=64, shuffle=False, num_workers=workers)
    
    valid_loader_aflw = data.DataLoader(
        k_dataset_valid_aflw, batch_size=64, shuffle=False, num_workers=workers) 
    
    return train_loader, valid_loader_kface, valid_loader_aflw