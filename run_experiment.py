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
from models.arch_utils import Architecture, get_model
from utils import create_exp_folder, get_config, update_conf, set_logger


__author__ = "Stella Muamba Ngufulu"
__contact__ = "stellamuambangufulu@gmail.com"
__copyright__ = "Copyright 2023, protein ligand project"
__date__ = "2023/01/13"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Dev"
__version__ = "0.0.1"


set_logger(__name__)


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


def main(args):
    conf = get_config()
    if args.SQL:
        
        if args.data_path:
            mol_feat = get_loader(path2data=args.data_path, **conf[Architecture(args.model)].data)
            conf.Credentials.HOST = args.HOST
            conf.Credentials.PORT = args.PORT
            
        else:
            db_name = conf[Architecture(args.model)].data.db_name
            mol_type = conf[Architecture(args.model)].data.mol_type
            if not db_name:
                raise Exception('No database for this project was created. Please give a path to a dataset.')
            
            user = input("User :")
            pwd = getpass.getpass("Password :")
            conf.Credentials.USER = user
            conf.Credentials.PASSWORD = pwd
            session = SQL_Session(**conf.Credentials)
            
            if not session.df_exists('mol_rec'):
                raise Exception(f"No {mol_type} found in {db_name} database and no path for dataloader ingestion was given.")
            
            mol_feat = get_loader(db_insertion=False, **conf[Architecture(args.model)].data)
    
    else:
        # model and path
        conf[Architecture(args.model)].data.db_name = args.db_name
        mol_feat = get_loader(path2data=args.data_path, db_insertion=False, **conf[Architecture(args.model)].data)

    logger.info(f"Creating experiment for model {args.model}.")
    
    if args.ckpt_path:
        best_ckpt = False
        
    exp = Experiment(Architecture(args.model), 
                    conf[Architecture(args.model)], 
                    mol_feat, 
                    ckpt_path=args.ckpt_path, 
                    best_ckpt=best_ckpt)
    exp.run()
    
    # remove username and password from config file
    conf.Credentials.USER = ''
    conf.Credentials.PASSWORD = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model.", required=True)
    parser.add_argument("--data_path", help="Path of dataset.")
    parser.add_argument("-sql", "--sql_info", dest='SQL', action='store_true')
    parser.add_argument("-ho", "--host", dest='HOST', help="Host for sql API.")
    parser.add_argument("-po", "--port", dest='PORT', help="Port for sql API.")
    parser.add_argument("-db", "--database", dest='db_name', help="Database name to use.")
    # parser.add_argument('-p', '--password', action=Password, nargs='?', dest='password')
    parser.add_argument("--ckpt_path", help="Path of checkpoint.")

    args = parser.parse_args()
    
    if args.data_path and args.SQL:
        if not (args.HOST or args.PORT or args.db_name):
            parser.error('Please specify [-ho HOST], [-po PORT] and [-db DATABASE].')
    # parsed_args = parser.parse_args()
    # args = vars(parsed_args)
    main(args)
