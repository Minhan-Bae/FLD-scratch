import numpy as np

def NME(label_pd, label_gt):
    sum_nme = 0
    total_cnt = 0
    label_pd = label_pd.data.cpu().numpy()
    label_gt = label_gt.data.cpu().numpy()
    
    for i in range(label_gt.shape[0]):
        landmarks_gt = label_gt[i]
        # print(landmarks_gt)
        landmarks_pv = label_pd[i]
        # print(landmarks_pv)
        pupil_distance = landmarks_gt[9] - landmarks_gt[12]
        # print(pupil_distance)
        if pupil_distance <= 0.0:
            pupil_distance = 0.01
        landmarks_delta = np.linalg.norm(landmarks_pv - landmarks_gt)
        nme = np.linalg.norm(landmarks_delta) / pupil_distance
        sum_nme+= nme
        total_cnt+=1

    total_nme = sum_nme/total_cnt
    return total_nme