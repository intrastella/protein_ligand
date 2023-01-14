from typing import Dict

import torch
import yaml

from enum import Enum
from pathlib import Path

from models.GAN.GAN import GAN
from models.Transformer.transformer import MultiHead, VAE


class Architecture(Enum):
    TRANSFORMER = "Transformer"
    GAN = "GAN"


def get_model(model_name: Architecture, model_conf: Dict, ckpt_dir: Union[List[Path], Path] = None):
    model = None
    if model_name == Architecture.GAN:
        if not any(ckpt_dir.iterdir()):
            if not isinstance(ckpt_dir, list):
                raise ValueError('More than 1 checkpoints must be given in a list.')
            if len(ckpt_dir) != 2:
                raise ValueError(f'1 checkpoint for genererator and 1 for discriminator is required. {len(ckpt_dir)} were given.')
            
            model_dict = {}
            checkpoint = torch.load(ckpt_dir[0])
            model_dict[batch_size] = checkpoint['batch_size']
            model_dict[training_steps] = checkpoint['training_steps']
            model_dict[n_epoch] = checkpoint['n_epoch']
            model_dict[n_critic] = checkpoint['n_critic']
            model_dict[lr] = checkpoint['lr']
            model_dict[b1] = checkpoint['b1']
            model_dict[b2] = checkpoint['b2']
            
            model = GAN(**model_dict)
            
        else:
            model = GAN(**model_conf)

    elif model_name == Architecture.TRANSFORMER:
        model = MultiHead(**model_conf)

    return model


def best_ckpt():
    pass


def get_config(model: Architecture):
    conf = None
    cwd = Path().absolute()
    if model == Architecture.GAN:
        file = 'GAN/gan_config.yaml'
        conf = yaml.safe_load(Path(f'{cwd}/models/{file}').read_text())
    elif model == Architecture.TRANSFORMER:
        file = None
        conf = yaml.safe_load(Path('GAN/gan_config.yaml').read_text())
    return conf
