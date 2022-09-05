#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import warnings

warnings.filterwarnings("ignore")

import gc

gc.collect()

import argparse
import time
import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.empty_cache()

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from configs import default_config as C

from dataset.dataloader import *

from models.xception import *
from models.metric.nme import *
from models.loss.wing_loss import *

from utils.fix_seed import *
from utils.visualize import *
from utils.averagemeter import *
from utils.optimizer import *
from utils.str2bool import *


# global args (configuration)
args = None


def parse_args():
    parser = argparse.ArgumentParser(description="Facial Landmark Detection")

    # Define data path
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--train-csv-path", type=str)
    parser.add_argument("--valid-csv-path", type=str)

    parser.add_argument("--log-dir", default=f"./logs/{C.DAY}/{C.TIME}", type=str)

    parser.add_argument("--gpus", default=C.DEVICE, type=str)
    parser.add_argument("--epochs", default=C.EPOCH, type=int)
    parser.add_argument("--batch-size", default=C.BATCH_SIZE, type=int)
    parser.add_argument("--base-lr", default=C.LR, type=float)
    parser.add_argument("--workers", default=C.WORKERS, type=int)

    parser.add_argument("--criterion", default="wingloss", type=str)
    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--momentum",  default=C.MOMENTUM, type=float)
    parser.add_argument("--weight-decay", default=C.WEIGHT_DECAY, type=float)
    parser.add_argument("--valid-term", default=C.VALID_TERM, type=int)
    parser.add_argument("--early-stop", default=C.EARLY_STOP_NUM, type=int)
    parser.add_argument("--seed", default=C.SEED, type=int)
    parser.add_argument("--milestones", default="15,25,30", type=str)
    parser.add_argument("--warmup", default=C.WARM_UP, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--valid-initial", default="true", type=str2bool)

    global args
    args = parser.parse_args()

    # some other operations(optional)
    args.devices_id = [int(d) for d in args.gpus.split(",")]
    args.milestones = [int(m) for m in args.milestones.split(",")]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # set log&save dir
    os.makedirs(os.path.join(args.log_dir, "image_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "model_logs"), exist_ok=True)


def print_args(args):
    for arg in vars(args):
        s = arg + ": " + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename=""):
    torch.save(state, filename)
    logging.info(f"\nSave checkpoint to {filename}")


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    # global lr
    lr = args.base_lr * (0.2**n)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def validate(valid_loader, model, save=None):
    cum_loss = 0.0
    cum_nme = 0.0

    model.eval()

    with torch.no_grad():
        for features, labels in valid_loader:
            features = features.cuda()
            labels = labels.cuda()

            outputs = model(features).cuda()

            loss = wing_loss(outputs, labels)
            nme = NME(outputs, labels)

            cum_nme += nme.item()
            cum_loss += loss.item()
            break
    if save:
        visualize_batch(
            features[:16].cpu(),
            outputs[:16].cpu(),
            labels[:16].cpu(),
            shape=(4, 4),
            size=16,
            save=save,
        )

    return cum_loss / len(valid_loader), cum_nme / len(valid_loader)


def train(train_loader, model, optimizer, epoch, lr, scaler):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = 0.0

    model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (features, landmarks_gt) in pbar:
        features = features.cuda(non_blocking=True)
        landmarks_gt = landmarks_gt.cuda(non_blocking=True)

        with autocast(enabled=True):
            predicts = model(features)
            loss = wing_loss(predicts, landmarks_gt)

        data_time.update(time.time() - end)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        flag, _ = optimizer.step_handleNan()

        if flag:
            print(
                "Nan encounter! Backward gradient error. Not updating the associated gradients."
            )

        train_loss += loss.item()
        batch_time.update(time.time() - end)
        end = time.time()

    msg = (
        "Epoch: {}\t".format(str(epoch).zfill(len(str(C.EPOCH))))
        + "LR: {:.8f}\t".format(lr)
        + "Time: {:.3f} ({:.3f})\t".format(batch_time.val, batch_time.avg)
        + "Loss: {:.8f}\t".format(train_loss / len(train_loader))
    )
    logging.info(msg)


def main():
    """Main function for the run process"""
    parse_args()

    # initial step
    seed_everything(args.seed)

    # logging step
    logging.basicConfig(
        format="[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir,f"{C.DAY}-{C.TIME}.log"), mode='w'),
            logging.StreamHandler(),
        ],
    )

    print_args(args)  # print args

    # step 1. define the model structure
    model = XceptionNet(num_classes=27 * 2)
    torch.cuda.set_device(args.devices_id[0])

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()

    # step 2. optimization: loss and optimization method
    optimizer = SGD_NanHandler(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # step 2.1. resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info("=> Loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage
            )["state_dict"]
            model.load_state_dict(checkpoint, strict=False)

        else:
            logging.info("=> No checkpoint found at {}".format(args.resume))

    # step 3. load data and dataloader
    train_loader, valid_loader = Dataloader(args)

    # step 4. run
    cudnn.benchmark = True

    # step 4.1. init validation
    if args.valid_initial:
        logging.info("Validation from initial")
        init_val_loss, init_val_nme = validate(
            valid_loader,
            model,
            save=os.path.join(
                f"{args.log_dir}",
                f"image_logs",
                f'epoch({str(0).zfill(len(str(C.EPOCH)))}).jpg',
            ),
        )
        s = "\nInit Landmark Detection Validation(Loss, NME) Loss: {:6f}, NME: {:.3f}".format(
            init_val_loss, init_val_nme
        )
        logging.info(s)

    # step 4.2. train
    early_cnt = 0
    for epoch in range(1, args.epochs + 1):
        scaler = GradScaler()

        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, lr, scaler)

        file_name = os.path.join(f"{args.log_dir}",
                                 f"model_logs",
                                 f"{C.DAY}-epoch-{epoch}.pt"
                                 )
        # valid for save log
        if (epoch % args.valid_term == 0) or (epoch == args.epochs):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                },
                filename=file_name,
            )

            logging.info("\nValidation[{}]".format(epoch))
            val_loss, val_nme = validate(
                valid_loader,
                model,
                save=os.path.join(
                    f"{args.log_dir}",
                    f"image_logs",
                    f'epoch({str(epoch).zfill(len(str(C.EPOCH)))}).jpg',
                ),
            )
            s = "\tLoss: {:6f}, NME: {:.3f}".format(val_loss, val_nme)
            logging.info(s)

            if val_nme < init_val_nme:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                    },
                    filename=os.path.join(f"dv-22-komediclub/SRC/models/pretrained",
                                          f"{C.DAY}-{C.TIME}-best.pt"),
                )
                init_val_nme = val_nme
                early_cnt = 0
            else:
                early_cnt += 1
                s = "\tEarly stopping count: {}/{}".format(early_cnt, args.early_stop)
                logging.info(s)
                if early_cnt == args.early_stop:
                    s = "\tEarly stopping is activate"
                    logging.info(s)
                    break


if __name__ == "__main__":
    main()
