import numpy as np
from facenet_pytorch import MTCNN

def mtcnn(image, margin=20): #PIL
    numpy_image =  np.array(image)
    mtcnn = MTCNN(
    image_size=160, margin=margin, min_face_size=10,
    thresholds=[0.2, 0.2, 0.2], factor=0.709, post_process=True,
    device='cuda:0', select_largest=False
    )
    
    bbox, _ = mtcnn.detect(numpy_image)
    offset = ((bbox[0][2]-bbox[0][0])//10, (bbox[0][3]-bbox[0][1])//10)

    x,y = bbox[0][0]-offset[0], bbox[0][1]-offset[1]
    w,h = bbox[0][2]+offset[0] - x, bbox[0][3]+offset[1] - y
    box = {"left": x, "top":y, "width":w, "height":h}
    return box

def return_box(image, aspect_ratio=None):
    box = mtcnn(image)
    return box