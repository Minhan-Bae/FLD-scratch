import torchvision.transforms.functional as TF

def transform_function(image):
    if image is not None:
        resize_img = TF.resize(image, (128,128))
        gray_img = TF.to_grayscale(resize_img)
        image = TF.to_tensor(gray_img)
        image = image.unsqueeze(0)
        image = (image - image.min())/(image.max() - image.min()) # set image value between(0,1)
        image = (2 * image) - 1 # set image value between(-1, 1)
        return image
    
if __name__=="__main__":
    transform_function()