import getpass
import logging
import logging.config
import toml

from box import Box
from datetime import datetime
from pathlib import Path

import torch

from models.model_utils import get_device


'''class Password(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        values = getpass.getpass()
        setattr(namespace, self.dest, values)'''


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
    return Box(toml.load(location))


def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)
