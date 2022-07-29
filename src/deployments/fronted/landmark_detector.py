import io
import math
import numpy as np
from PIL import Image

from utils.transforms import transform_function
from utils.xception import xception_Net_54

def run_detect(cropped_img_byte,
               pretrained=None,
               rotation=False):
    pil_image = Image.open(io.BytesIO(cropped_img_byte))
    image_tensor = transform_function(pil_image)
    
    model = xception_Net_54(pretrained=pretrained)
    predict = model(image_tensor)
    
    landmarks = ((predict.view(-1,2)+0.5)).detach().numpy().tolist()
    landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])
    
    if rotation:
        p1 = np.array(landmarks[18])
        p2 = np.array(landmarks[5])
        
        angle = 90-(math.degrees(math.asin(abs(p2[1]-p1[1])/np.linalg.norm(p2-p1))))
        
        if angle > 30:
            if p1[0]<p2[0]:
                pil_image = pil_image.rotate(-angle)
            else:
                pil_image = pil_image.rotate(angle)
            
            new_image_tensor = transform_function(pil_image)
            model = xception_Net_54(pretrained=pretrained)
            
            new_predict = model(new_image_tensor)

            landmarks = ((new_predict.view(-1,2)+0.5)).detach().numpy().tolist()
            landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])
            
    return pil_image, landmarks