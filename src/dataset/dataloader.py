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
    
    len_train = int(len(k_dataset_train)*0.8)
    len_valid = len(k_dataset_train)-len_train
    
    train_dataset, _ = data.random_split(k_dataset_train, [len_train, len_valid])
    _, valid_dataset = data.random_split(k_dataset_valid, [len_train, len_valid])
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader