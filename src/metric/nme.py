import numpy as np
from math import *
def NME(label_pd, label_gt):
    label_pd = label_pd.data.cpu().numpy()
    label_gt = label_gt.data.cpu().numpy()
    
    nme_list = []
    length_list = []
    
    for i in range(label_gt.shape[0]):
        landmarks_gt = label_gt[i]
        landmarks_pv = label_pd[i]
        
        minx, maxx = np.min(landmarks_gt), np.max(landmarks_gt)

        llength = (maxx - minx)
        length_list.append(llength)
        
        dis = landmarks_pv - landmarks_gt
        dis = np.sqrt(np.sum(np.power(dis,2),0))
        nme = dis/llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    mean_nme = np.mean(nme_list)*100
    std_nme = np.std(nme_list)*100
    return mean_nme, std_nme