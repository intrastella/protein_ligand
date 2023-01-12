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

cwd = Path().absolute()
logging.basicConfig(filename=f"{cwd}/std.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


cuda = True if torch.cuda.is_available() else False


class Experiment:
    def __init__(self,
                 model: Architecture,
                 model_conf: Dict,
                 dataset: DataLoader):

        self.model = get_model(model, model_conf['Hyperparameter'])
        self.dataset = dataset

    def run(self):
        self.model.fit(self.dataset)
        # images = self.model.test()


def main(opt):
    model_conf = get_config(Architecture(opt.model))

    log_data = yaml.safe_load(Path('data/data_conf.yaml').read_text())
    log_data['Credentials']['DATABASE'] = model_conf['data']['db_name']
    session = SQL_Session(**log_data['Credentials'])

    if not session.df_exists('mol_rec'):
        if not opt.path:
            mol_type = model_conf['data']['mol_type']
            db_name = model_conf['data']['db_name']
            raise Exception(f"No {mol_type} found in {db_name} database and no path for data ingestion was given.")

    logger.info(f"Creating experiment for model {opt.model}.")
    mol_feat = get_loader(path2data=opt.path, **model_conf['data'])
    exp = Experiment(Architecture(opt.model), model_conf, mol_feat)
    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model.", required=True)
    parser.add_argument("--path", help="Path of dataset.")
    # group = parser.add_mutually_exclusive_group()

    main(parser.parse_args())
