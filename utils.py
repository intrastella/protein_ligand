import logging
import logging.config
import toml

from box import Box
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

from torch import nn


def set_logger(name):
    cwd = Path().absolute()
    log_path = cwd / 'std.log'
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s - %(levelname)s - %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_path,
                'maxBytes': 1024,
                'backupCount': 3
            }
        },
        'loggers': {
            'default': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            }
        },
        'disable_existing_loggers': False
    })
    return logging.getLogger(name)


def create_exp_folder() -> Path:
    """
    Creates and returns a new experiment directory
    """
    cwd = Path().absolute()
    now = datetime.now()
    exp_id = now.strftime("%d-%m-%Y_%H-%M-%S")
    folder = cwd / f'experiments/{exp_id}'
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p
    
    
def get_config():
    """
    returns a toml dict as instance from box, i.e. dot notation 
    """
    cwd = Path().absolute()
    location = cwd / 'config.toml'
    data = Box(toml.load(location))
    return conf


def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)
