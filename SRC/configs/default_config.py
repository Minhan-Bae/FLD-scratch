import os
from datetime import datetime

# Experiment setup
DAY = f"{datetime.now().year}"+"-"+f"{datetime.now().month}".zfill(2)+"-"+f"{datetime.now().day}".zfill(2)
TIME = f"{datetime.now().hour}".zfill(2)+'-'+f"{datetime.now().minute}".zfill(2) # H_M 

DEVICE = '0,1,2,3'
MODEL = "xception"
EPOCH = 500
LR = 1e-4
SEED = 2022
BATCH_SIZE = 128 * len(DEVICE.split(',')) # number of gpu * 128(T4*4)
WORKERS = 4 * len(DEVICE.split(',')) # number of gpu * 4
EARLY_STOP_NUM = 200
VALID_TERM = 5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
WARM_UP = 50