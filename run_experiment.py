import argparse
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
                 best_ckpt = True):

        self.dataset = dataset

        if not exp_id:
            self.exp_dir = create_exp_folder()

        else:
            self.exp_dir = Path(f'{cwd}/exp_id')

        self.weight_dir = self.exp_dir / 'weights'
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.exp_dir / 'model_ckpts'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_path = ckpt_path

        self.model = get_model(model, model_conf['Hyperparameter'], self.ckpt_path)

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
    if opt.ckpt_path:
        exp = Experiment(Architecture(opt.model), model_conf, mol_feat, ckpt_path=opt.ckpt_path, best_ckpt=False)
    else:
        exp = Experiment(Architecture(opt.model), model_conf, mol_feat)
    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model.", required=True)
    parser.add_argument("--data_path", help="Path of dataset.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ckpt_path", help="Path of checkpoint.")
    group.add_argument("--best_ckpt", action='store_false', help="Use best trained model version.")

    main(parser.parse_args())
