from dataset import augmentation, dataset
from torch.utils import data

def kfacedataloader(batch_size_train, batch_size_valid, workers):
    k_dataset_train = dataset.kfacedataset(
        type="train",
        transform=augmentation.get_augmentation(data_type="train")
        )

    k_dataset_valid = dataset.kfacedataset(
        type="valid",
        transform=augmentation.get_augmentation(data_type="valid")
        )
    
    # len_train = int(len(k_dataset_train)*0.8)
    # len_valid = len(k_dataset_train)-len_train
    
    # train_dataset, _ = data.random_split(k_dataset_train, [len_train, len_valid])
    # _, valid_dataset = data.random_split(k_dataset_valid, [len_train, len_valid])
    
    train_loader = data.DataLoader(
        k_dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=workers)
    valid_loader = data.DataLoader(
        k_dataset_valid, batch_size=batch_size_valid, shuffle=False, num_workers=workers)
    
    return train_loader, valid_loader