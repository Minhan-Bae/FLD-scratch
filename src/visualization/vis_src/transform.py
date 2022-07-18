import warnings
warnings.filterwarnings("ignore")

import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN

def mtcnn(image, margin=400): #PIL
    numpy_image =  np.array(image)
    # numpy_image = numpy.
    mtcnn = MTCNN(
    image_size=512, margin=margin, min_face_size=100,
    thresholds=[0.1,0.1,0.1], factor=0.5, post_process=True,
    device='cpu', select_largest=True
    )
    
    bbox, _ = mtcnn.detect(numpy_image)
    crop_area = (bbox[0][0]-(bbox[0][2]-bbox[0][0])//2, # get bbox area with margin
                 bbox[0][1]-(bbox[0][3]-bbox[0][1])//2,
                 bbox[0][2]+(bbox[0][2]-bbox[0][0])//2,
                 bbox[0][3]+(bbox[0][3]-bbox[0][1])//2)
    
    pil_image = Image.fromarray(numpy_image)
    crop_img = pil_image.crop(crop_area)
    
    return crop_img, np.array(crop_img).shape[0], np.array(crop_img).shape[1]

def transform(image):
    crop_img, h, w = mtcnn(image)
    resize_img = TF.resize(crop_img, (112,112))
    image_tensor = TF.to_tensor(resize_img)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, crop_img, h, w