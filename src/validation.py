import torch
from utils.visualize import visualize_batch
from metric.nme import NME

def validate(valid_loader, landmark_model, save = None):
    cum_loss = 0.0
    cum_nme = 0.0
    
    with torch.no_grad():
        for features, landmarks_gt, _ in valid_loader:
            features = features.cuda()
            landmarks_gt = landmarks_gt.cuda()

            _, predicts = landmark_model(features)

            loss = torch.mean(torch.sum((landmarks_gt - predicts)**2, axis=1))
            mean_nme = NME(predicts, landmarks_gt)

            cum_loss += loss.item()
            cum_nme += mean_nme.item()
            
    visualize_batch(features[:16].cpu(),
                    predicts[:16].cpu(),
                    landmarks_gt[:16].cpu(),
                    shape = (4, 4), size = 16, title = None, save = save)
    
    return cum_nme/len(valid_loader), cum_loss/len(valid_loader)