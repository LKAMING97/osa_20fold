# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:log.py
@Time:2022/6/28 9:41

"""
import logging
import os
import sys


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)  # 获取logger对象，如果不指定name则返回root对像
    logger.propagate = False
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    logger = get_logger("", "main")
