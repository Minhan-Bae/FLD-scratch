import torch
torch.cuda.empty_cache()

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from utils.visualize import *
from utils.logging import *
from metric.nme import NME

def validate(types, valid_loader, model, criterion, log_list, save = None):
    cum_loss = 0.0
    cum_nme = 0.0
    model.eval()
    with torch.no_grad():
        for features, labels in valid_loader:
            features = features.cuda()
            labels = labels.cuda()

            outputs = model(features).cuda()

            loss = criterion(outputs, labels)
            nme=NME(outputs, labels)
            
            cum_nme += nme.item()
            cum_loss += loss.item()
            break
            
    visualize_batch(features[:16].cpu(), outputs[:16].cpu(), labels[:16].cpu(), shape = (4, 4), size = 16, title = 'Validation sample predictions', save = save)

    log_list = logging(f"|     ===> Evaluate {types}:",log_list)
    log_list = logging(f'|          Eval set: Normalize Mean Error: {cum_nme/len(valid_loader):.4f}',log_list)
    log_list = logging(f'|          Eval set: Average loss: {cum_loss/len(valid_loader):.4f}',log_list)    

    return cum_loss/len(valid_loader), cum_nme/len(valid_loader), log_list