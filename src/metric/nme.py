import numpy as np

def NME(label_pd, label_gt):
    sum_nme = 0
    total_cnt = 0
    label_pd = label_pd.data.cpu().numpy()
    label_gt = label_gt.data.cpu().numpy()
    
    for i in range(label_gt.shape[0]):
        landmarks_gt = label_gt[i]
        landmarks_pv = label_pd[i]
        pupil_distance = np.linalg.norm(landmarks_gt[9] - landmarks_gt[12])
        landmarks_delta = landmarks_pv - landmarks_gt
        nme = np.linalg.norm(landmarks_delta) / pupil_distance
        sum_nme+= nme
        total_cnt+=1

    total_nme = sum_nme/total_cnt
    return total_nme