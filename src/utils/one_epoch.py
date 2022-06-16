import torch

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(epoch, model, data_loader, criterion, optimizer, scheduler, device):
    model.train()

    cnt = 0
    correct = 0
    scaler = GradScaler()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (image, label) in pbar:

        image = torch.stack(image).float()
        label = torch.stack(label).long()

        image = image.to(device)
        label = label.to(device)

        with autocast(enabled=True):
            model = model.to(device)

            output = model(image)
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        _, preds = torch.max(output, 1)
        correct += torch.sum(preds == label.data)
        cnt += 1

        description = f"| # Epoch : {epoch + 1} Loss : {(loss.item()):.4f}"
        pbar.set_description(description)

    acc = correct / cnt
    scheduler.step()
    
def valid_one_epoch(model, data_loader, split_df, device):
    print(f"Start Validation")

    model.eval()
    correct = 0

    pbar_valid = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (image, label) in pbar_valid:

        image = torch.stack(image).float()
        label = torch.stack(label).long()

        image = image.to(device)
        label = label.to(device)
        model = model.to(device)

        output = model(image)

        _, preds = torch.max(output, 1)
        correct += torch.sum(preds == label.data)

        description_valid = f"| Acc : {(correct.item()/len(split_df)):.4f}"
        pbar_valid.set_description(description_valid)
    acc = correct / len(split_df)

    return acc, output