from dataset.augmentation import get_augmentation
from dataset.dataset import AFLWDatasets
from torch.utils import data

def AFLWDataloader(batch_size, workers):
    # set datasets
    dataset_train = AFLWDatasets(
        type="train",
        transform=get_augmentation(data_type="valid")
        )

    dataset_valid = AFLWDatasets(
        type="valid",
        transform=get_augmentation(data_type="valid")
        )

    # get loader(image, landmark, angle)
    train_loader = data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)
    
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=64, shuffle=False, num_workers=workers, drop_last=False)

    return train_loader, valid_loader