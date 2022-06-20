import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def vis_keypoints(image, keypoints, color=(255,0,0), diameter=1):
    image = image.numpy().transpose(1,2,0).squeeze()
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)
        
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image, cmap=cm.gray)
    plt.show()