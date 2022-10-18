# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:config.py
@Time:2022/6/30 14:40

"""
import argparse
from datetime import datetime
import random
import os
import torch
import numpy as np
from core.log import *


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args_parser():

    parser = argparse.ArgumentParser(description='model_config')
    
    # general
    parser.add_argument("--save_path", type=str, default="", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="run_device")
    parser.add_argument("---num_class", type=int, default= 2, help="class_num")
    parser.add_argument(
        "--suffix", type=str, default="", help="suffix of save dir name"
    )

    # data
    # gatemlp
    parser.add_argument("--data_path", type=str, default="data/process_osa_v2.csv", help="path of dataset")
    # mlp
    # parser.add_argument("--data_path", type=str, default="data/process_osa_v2_36f.csv", help="path of dataset")
    parser.add_argument(
        "--n_threads",
        type=int,
        default="4",
        help="number of threads used for data loading",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help="date of experiment",
    )
    parser.add_argument("--submit_dir", type=str, default="data/" + datetime.today().strftime("%Y%m%d") + "_submit.csv",
                        help="submit_address")
    parser.add_argument("--is_train", type=bool, default=True, help="select train or valid")
    parser.add_argument("--val_prop", type=float, default=0.2, help="val_select")
    parser.add_argument(
        "--max_length", type=int, default=180, help="max length"
    )
    # optimization
    parser.add_argument("--model", type=str, default="8_gMLP", help="choice model")
    parser.add_argument("--select", type=int, default=1, help="choice model")
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum term")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--lr", type=float, default=5e-4
                        , help="initial learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0,
        help="minimum learning rate of cosine scheduler",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="number of training epochs"
    )
    parser.add_argument(
        "--step",
        type=lambda s: [int(item) for item in s.split(",")],
        default="1",
        help="multi-step for linear learning rate",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="multi_step"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=999, help="number of warmup epoch"
    )
    parser.add_argument(
        "--no_decay_keys",
        type=lambda s: [item for item in s.split(",")] if len(s) != 0 else "",
        default="",
        help="key name for apply weight decay",
    )

    parser.add_argument("--opt_type", type=str, default="AdamW", help="optimizer")

    args = parser.parse_args()

    return args


def create_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def set_save_path(args):
    if len(args.suffix) == 0:
        suffix = "log_model_{:s}_{}_bs{:d}_epoch{:d}_lr{:.5f}/".format(
            args.model,
            args.date,
            args.batch_size,
            args.epochs,
            args.lr,
        )
    else:
        suffix = args.suffix
    args.save_path = os.path.join("log", args.save_path, suffix)

    create_dir(args.save_path)


def write_settings(settings):
    """
    Save expriment settings to a file
    :param settings: the instance of option
    """

    with open(os.path.join(settings.save_path, "settings.log"), "w") as f:
        for k, v in settings.__dict__.items():
            f.write(str(k) + ": " + str(v) + "\n")