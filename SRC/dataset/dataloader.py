from .augmentation import get_augmentation
from .datasets import Datasets
from torch.utils import data

import sys
sys.path.append('..')

def Dataloader(args):
    dataset_train = Datasets(
        args,
        type="train",
        transform=get_augmentation(data_type="train")
        )

    dataset_valid = Datasets(
        args,
        type="valid",
        transform=get_augmentation(data_type="valid")
        )

    # get loader(image, landmark)
    train_loader = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=64, shuffle=False, num_workers=args.workers, drop_last=True)

    return train_loader, valid_loader