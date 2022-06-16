import cv2

def load_image(file):
    image = cv2.imread(file)
    
    if file.endswith(".png"):
        image = image[:,:,3]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image