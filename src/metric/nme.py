import numpy as np
from math import *
def NME(label_pd, label_gt):
    nme_list = []
    
    for i in range(label_gt.shape[0]):
        
        landmarks_gt = label_gt[i].view(-1,2)
        landmarks_gt = (landmarks_gt+0.5) * 256
        landmarks_gt = landmarks_gt.detach().cpu().numpy()
        
        landmarks_pv = label_pd[i].view(-1,2)
        landmarks_pv = (landmarks_pv+0.5) * 256
        landmarks_pv = landmarks_pv.detach().cpu().numpy()
        
        minx, maxx = np.min(landmarks_gt[0,:]), np.max(landmarks_gt[0,:])
        miny, maxy = np.min(landmarks_gt[1,:]), np.max(landmarks_gt[1,:])
        
        llength = sqrt((maxx - minx) * (maxy - miny))
        
        for idx in range(len(landmarks_gt)):
            dis = landmarks_pv[idx] - landmarks_gt[idx]
            dis = np.sqrt(np.sum(np.power(dis,2),0))
            nme = dis/llength
            nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    mean_nme = np.mean(nme_list)*100
    return mean_nme