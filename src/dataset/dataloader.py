from dataset.augmentation import get_augmentation
from dataset.dataset import Datasets
from torch.utils import data

def Dataloader(batch_size, workers):
    # set datasets
    dataset_train = Datasets(
        data_path = "/data/komedi/komedi/dataset/versioning/22-07-22-1200-train.csv",
        type="train",
        transform=get_augmentation(data_type="train")
        )

    dataset_valid_aflw = Datasets(
        data_path = "/data/komedi/komedi/dataset/versioning/22-07-22-1200-valid-aflw.csv",
        type="valid",
        transform=get_augmentation(data_type="valid")
        )
    
    dataset_valid_face = Datasets(
        data_path = "/data/komedi/komedi/dataset/versioning/22-07-22-1200-valid-kface.csv",
        type="valid",
        transform=get_augmentation(data_type="valid")
        )

    # get loader(image, landmark)
    train_loader = data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)
    
    valid_loader_aflw = data.DataLoader(
        dataset_valid_aflw, batch_size=64, shuffle=False, num_workers=workers, drop_last=True)

    valid_loader_face = data.DataLoader(
        dataset_valid_face, batch_size=64, shuffle=False, num_workers=workers, drop_last=True)

    return train_loader, valid_loader_aflw, valid_loader_face