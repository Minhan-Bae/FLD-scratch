import torch
torch.cuda.empty_cache()

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from src.utils import visualize
from src.metric.nme import NME

def validate(valid_loader, model, save = None):
    cum_mean_nme = 0.0
    cum_std_nme = 0.0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for idx, (features, labels) in pbar:
            features = features.cuda()
            labels = labels.cuda()
            
            with autocast(enabled=True):
                outputs = model(features).cuda()
                mean_nme, std_nme = NME(outputs, labels)
            
            cum_mean_nme += mean_nme.item()
            cum_std_nme += std_nme.item()
            
            description_valid = f"| # mean_nme: {cum_mean_nme/(idx+1):.8f}, std_nme: {cum_std_nme/(idx+1):.8f}"
            pbar.set_description(description_valid)
            
        visualize.visualize_batch(features[:16].cpu(), outputs[:16].cpu(), labels[:16].cpu(),
                    shape = (4, 4), size = 16, title = None, save = save)
    
    return cum_mean_nme/len(valid_loader), cum_std_nme/len(valid_loader)