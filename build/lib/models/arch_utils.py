from typing import Dict, Union, List

import torch

from enum import Enum
from pathlib import Path

from models.GAN.GAN import GAN
from models.Transformer.transformer import MultiHead, VAE
from utils import set_logger


logger = set_logger(__name__)


class Architecture(Enum):
    TRANSFORMER = "Transformer"
    GAN = "GAN"


def get_model(model_name: Architecture, model_conf: Dict, ckpt_path: Union[str, List[str]]=None):
    model = None
    if model_name == Architecture.GAN:
        if ckpt_path:
            if not isinstance(ckpt_path, list):
                raise ValueError('More than 1 checkpoints must be given in a list.')
            if len(ckpt_path) != 2:
                raise ValueError(f'1 checkpoint for genererator and 1 for discriminator is required. {len(ckpt_path)} were given.')
            
            model_conf = {}
            checkpoint = torch.load(ckpt_path[0])
            model_conf['batch_size'] = checkpoint['batch_size']
            model_conf['training_steps'] = checkpoint['training_steps']
            model_conf['n_epoch'] = checkpoint['n_epoch']
            model_conf['n_critic'] = checkpoint['n_critic']
            model_conf['lr'] = checkpoint['lr']
            model_conf['b1'] = checkpoint['b1']
            model_conf['b2'] = checkpoint['b2']
            model_conf['ckpt_path'] = ckpt_path
            
        model = GAN(**model_conf)
        model.setup()

    elif model_name == Architecture.TRANSFORMER:
        model = MultiHead(**model_conf)

    return model
