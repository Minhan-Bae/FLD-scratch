import os
import natsort

def get_file_list(path, case):
    if case=="image":
        image_extension = [".jpg", ".png", ".jpeg"]
    else:
        image_extension = [".csv"]
    file_list = []
    for (root, _, files) in os.walk(path):
        if len(files) > 0:
            for file_name in files:
                if file_name[-4:] in image_extension:
                    file_list.append(os.path.join(root,file_name))
    return natsort.natsorted(file_list)