# utils/logger.py
import logging
import os
from logzero import setup_logger

def get_logger(log_dir: str | None = None, filename: str = "train.log"):
    """
    log_dir を指定すると、log_dir/filename にログを出す。
    指定しない場合は従来通り utils/log.txt に出す（互換用）。
    """
    if log_dir is None:
        log_path = "utils/log.txt"
        os.makedirs("utils", exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)

    return setup_logger(
        name="logger",
        level=logging.DEBUG,
        formatter=None,
        fileLoglevel=logging.DEBUG,
        disableStderrLogger=False,
        logfile=log_path,
    )

# 互換用：従来通り import logger でも動く
logger = get_logger()
