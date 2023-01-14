#!/usr/bin/env python3

""" 
    ------------------------------- *** DESCRIPTION *** ------------------------------
    This module creates an instance of an experiment of a DL model.
    So far it consits of a mixture of WGan and Transformer model.
    You can specify checkpoints from prior trainings or chooses the best one.
    ----------------------------------------------------------------------------------
    
    ---------------------------------- *** DATA *** ----------------------------------
    The data types for training and evalutation : smiles and sdf
    You can choose to store the data in a sql database.
    When executing this script it will ask for your credentials from you sql databse.
    ----------------------------------------------------------------------------------


    -------------------------------- *** LISCENCE *** --------------------------------
    This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.
    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. 
    -----------------------------------------------------------------------------------
"""


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


class Experiment:
    def __init__(self,
                 model: Architecture,
                 model_conf: Dict,
                 dataset: DataLoader,
                 exp_id: str = None, 
                 ckpt_path: str = None,
                 best_ckpt: bool = True):

        self.dataset = dataset

        if not exp_id:
            self.exp_dir = create_exp_folder()

        else:
            self.exp_dir = Path(f'{cwd}/exp_id')

        self.weight_dir = self.exp_dir / 'weights'
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.exp_dir / 'model_ckpts'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        if best_ckpt:
            ckpt_path = self.best_ckpt
            
        self.model = get_model(model, model_conf['Hyperparameter'], ckpt_path)

    def run(self):
        self.model.fit(self.dataset, self.exp_dir)
        # self.model.evaluate()

    @property
    def best_ckpt(self):
        pass


def main(opt):
    model_conf = get_config(Architecture(opt.model))
    cwd = Path().absolute()
    log_data = yaml.safe_load(Path(f'{cwd}/data/data_conf.yaml').read_text())
    log_data['Credentials']['DATABASE'] = model_conf['data']['db_name']
    session = SQL_Session(**log_data['Credentials'])

    if not session.df_exists('mol_rec'):
        if not opt.path:
            mol_type = model_conf['data']['mol_type']
            db_name = model_conf['data']['db_name']
            raise Exception(f"No {mol_type} found in {db_name} database and no path for dataloader ingestion was given.")

    logger.info(f"Creating experiment for model {opt.model}.")
    mol_feat = get_loader(path2data=opt.data_path, **model_conf['data'])
    if not opt.best_ckpt:
        exp = Experiment(Architecture(opt.model), model_conf, mol_feat, ckpt_path=opt.ckpt_path, best_ckpt=False)
    else:
        exp = Experiment(Architecture(opt.model), model_conf, mol_feat)
    exp.run()


if __name__ == '__main__':
    USER: 'root'
    PASSWORD:
  HOST: '127.0.0.1'
  PORT: 3306
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model.", required=True)
    parser.add_argument("--data_path", help="Path of dataset.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ckpt_path", help="Path of checkpoint.")
    group.add_argument("--best_ckpt", action='store_true', help="Use best trained model version.")

    # parsed_args = parser.parse_args()
    # args = vars(parsed_args)
    main(parser.parse_args())
