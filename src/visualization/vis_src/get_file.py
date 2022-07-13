import os

def get_image_list(dir_path, num_of_image=100):
    image_list = [os.path.join(dir_path, os.listdir(dir_path)[i]) for i in range(num_of_image)]
    return image_list
