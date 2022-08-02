import io
import math
import numpy as np
from PIL import Image, ImageOps

from utils.transforms import transform_function
from utils.xception import xception_Net_54

def run_detect(image,
               cropped_img_byte,
               pretrained=None,
               rotation=False):
    pil_image = Image.open(io.BytesIO(cropped_img_byte))
    pil_image_mirror = ImageOps.mirror(pil_image)
    
    image_tensor_normal = transform_function(pil_image)
    image_tensor_mirror = transform_function(pil_image_mirror)
    
    model = xception_Net_54(pretrained=pretrained)
    
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
    
    p1 = np.array(total_landmarks[18])
    p2 = np.array(total_landmarks[5])
    
    angle = 90-(math.degrees(math.asin(abs(p2[1]-p1[1])/np.linalg.norm(p2-p1))))
    angle = 0 # Underconstruct
    if rotation:
        if angle > 30:
            if p1[0]<p2[0]:
                pil_image = pil_image.rotate(-angle)
                image = image.rotate(-angle)
                angle = -angle
            else:
                pil_image = pil_image.rotate(angle)
                image = image.rotate(angle)                
            
            new_image_tensor = transform_function(pil_image)
            model = xception_Net_54(pretrained=pretrained)
            
            new_predict = model(new_image_tensor)

            landmarks = ((new_predict.view(-1,2)+0.5)).detach().numpy().tolist()
            landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])
        else:
            angle = 0
    return image, pil_image, total_landmarks, angle