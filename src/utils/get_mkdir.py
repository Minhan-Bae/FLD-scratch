import os
from pathlib import Path

def get_mkdir(path, dir_type):
    dir_dict = {}
    try:
        for d in dir_type:
            dir_name = d+"_path"
            dir_list = os.path.join(path,d)
            dir_dict[d]=dir_list
            
            if not os.path.exists(dir_list):
                Path(dir_list).mkdir(parents=True, exist_ok=True)
                print(f"| {dir_name} was maked")
    except IndexError:
        pass