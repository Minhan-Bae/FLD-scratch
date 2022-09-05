from fastapi import UploadFile
from PIL import Image, ImageOps
import numpy as np
import json

import sys
sys.path.append('..')
from utils.transforms import transform_function

def inference_landmark_detection(model, cropped_img_byte: UploadFile):
    
    pil_image = Image.open(cropped_img_byte.file)
    pil_image_mirror = ImageOps.mirror(pil_image)
    
    image_tensor_normal = transform_function(pil_image)
    image_tensor_mirror = transform_function(pil_image_mirror)
    
    
    predict_normal = model(image_tensor_normal)
    predict_mirror = model(image_tensor_mirror)
    
    landmarks_normal = ((predict_normal.view(-1,2)+0.5)).detach().numpy().tolist()
    landmarks_mirror = ((predict_mirror.view(-1,2)+0.5)).detach().numpy().tolist()
    
    mirror_align = []
    mirror_idx = (4,3,2,1,0,5,6,7,8,12,11,10,9,15,14,13,16,17,18,19,20,22,21,23,24,25,26)
    for idx in mirror_idx:
        mirror_align.append(landmarks_mirror[idx])
    
    landmarks_normal = np.array([(x*pil_image.width, y*pil_image.height) for (x, y) in landmarks_normal if 0 <= x and 0 <= y])
    landmarks_mirror = np.array([(pil_image.width-x*pil_image.width,y*pil_image.height) for (x, y) in mirror_align if 0 <= x and 0 <= y])
    
    total_landmarks = []
    for normal, mirror in zip(landmarks_normal, landmarks_mirror):
        total_landmarks.append(((normal[0]+mirror[0])/2,(normal[1]+mirror[1])/2))
    
    landmarks_json = json.load(open("/srv/deployments/backend/utils/template/landmarks.json", "r"))
    for idx, _ in enumerate(landmarks_json["landmarks"]):
        landmarks_json["landmarks"][idx]["x"] = total_landmarks[idx][0].item()
        landmarks_json["landmarks"][idx]["y"] = total_landmarks[idx][1].item()

    return {"label": landmarks_json["landmarks"]}