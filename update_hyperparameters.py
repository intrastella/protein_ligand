import argparse
import getpass
import logging
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from data.db_tools import SQL_Session
from models.GAN.data_loader import get_loader
from models.arch_utils import Architecture, get_model, get_config
from utils import create_exp_folder


__author__ = "Stella Muamba Ngufulu"
__contact__ = "stellamuambangufulu@gmail.com"
__copyright__ = "Copyright 2023, protein ligand project"
__date__ = "2023/01/13"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Dev"
__version__ = "0.0.1"


cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def main(args):
    cwd = Path().absolute()
    params = yaml.safe_load(Path(f'{cwd}/model/GAN/GAN/gan_config.yaml').read_text())


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model.", required=True)
    
    subparsers1 = parser.add_subparsers(dest='cmd')
    gan = subparsers1.add_parser('gan')
    gan.add_argument("--batch_size")
    gan.add_argument("--training_steps")
    gan.add_argument("--n_epoch")
    gan.add_argument("--n_critic")
    gan.add_argument("--lr")
    gan.add_argument("--b1")
    gan.add_argument("--b2")
    
    subparsers2 = parser.add_subparsers(dest='cmd')
    transformer = subparsers2.add_parser('transformer')
    transformer.add_argument("--batch_size")
    transformer.add_argument("--training_steps")
    transformergan.add_argument("--n_epoch")
    transformer.add_argument("--lr")
    transformer.add_argument("--b1")
    transformer.add_argument("--b2")
    
    
    # parsed_args = parser.parse_args()
    # args = vars(parsed_args)
    main(args)
