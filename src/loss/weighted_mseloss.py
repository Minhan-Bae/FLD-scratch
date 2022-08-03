import torch
criterion = torch.nn.MSELoss()

def weighted_mseloss(preds, labels):
    loss1 = criterion(preds[:24], labels[:24])
    loss2 = criterion(preds[24:], labels[24:])
    return 0.7*loss1+0.3*loss2