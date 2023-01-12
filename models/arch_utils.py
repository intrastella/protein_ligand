from typing import Dict
import yaml

from enum import Enum
from pathlib import Path

from models.GAN.GAN import GAN
from models.Transformer.transformer import MultiHead, VAE


class Architecture(Enum):
    TRANSFORMER = "Transformer"
    GAN = "GAN"
    VAE = "VAE"


def get_model(model_name: Architecture, model_conf: Dict):
    model = None
    if model_name == Architecture.GAN:
        model = GAN(**model_conf)
    elif model_name == Architecture.VAE:
        model = VAE(**model_conf)
    elif model_name == Architecture.TRANSFORMER:
        model = MultiHead(**model_conf)

    return model


def get_config(model: Architecture):
    conf = None
    cwd = Path().absolute()
    if model == Architecture.GAN:
        file = 'GAN/gan_config.yaml'
        conf = yaml.safe_load(Path(f'{cwd}/models/{file}').read_text())
    elif model == Architecture.VAE:
        file = None
        conf = yaml.safe_load(Path('GAN/gan_config.yaml').read_text())
    elif model == Architecture.TRANSFORMER:
        file = None
        conf = yaml.safe_load(Path('GAN/gan_config.yaml').read_text())
    return conf

Path().absolute()
