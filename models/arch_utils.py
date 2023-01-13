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


def get_model(model_name: Architecture, model_conf: Dict, weight_dir: Path = None):
    model = None
    if model_name == Architecture.GAN:
        if not any(weight_dir.iterdir()):
            model = GAN(**model_conf)
            model.load_state_dict(torch.load(weight_dir))
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
