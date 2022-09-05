#!/bin/sh

python /home/ubuntu/workspace/dv-22-komediclub/SRC/main.py \
    --epochs 500\
    --criterion wingloss\
    --image-dir /data/komedi/data\
    --train-csv-path /home/ubuntu/workspace/dv-22-komediclub/SRC/dataset/data/train_df.csv\
    --valid-csv-path /home/ubuntu/workspace/dv-22-komediclub/SRC/dataset/data/valid_df.csv\
    --resume /home/ubuntu/workspace/dv-22-komediclub/SRC/models/pretrained/2022-08-31-16-30-best.pt
