from typing import Dict

import torch
import yaml

from enum import Enum
from pathlib import Path

from models.GAN.GAN import GAN
from models.Transformer.transformer import MultiHead, VAE


cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Architecture(Enum):
    TRANSFORMER = "Transformer"
    GAN = "GAN"


def get_model(model_name: Architecture, model_conf: Dict, ckpt_path: Union[List[Union[str, Path]], Union[str, Path]] = None):
    model = None
    if model_name == Architecture.GAN:
        if ckpt_path:
            if not isinstance(ckpt_path, list):
                raise ValueError('More than 1 checkpoints must be given in a list.')
            if len(ckpt_path) != 2:
                raise ValueError(f'1 checkpoint for genererator and 1 for discriminator is required. {len(ckpt_dir)} were given.')
            
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
