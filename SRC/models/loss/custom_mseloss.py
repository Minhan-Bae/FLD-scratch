import torch
criterion = torch.nn.MSELoss()

def custom_loss(preds, labels, ratio=(0.1,0.2,0.3,0.1,0.3),activate=False):
    if activate:
        total_loss = 0.0
        total_loss += ratio[0]*criterion(preds[:18], labels[:18])
        total_loss += ratio[1]*criterion(preds[18:27], labels[18:27])
        total_loss += ratio[2]*criterion(preds[27:40], labels[27:40])
        total_loss += ratio[3]*criterion(preds[40:42], labels[40:42])
        total_loss += ratio[4]*criterion(preds[42:], labels[42:])
        return total_loss
    else:
        return criterion(preds, labels)
        