import os
import logging
from pathlib import Path
from typing import List, Union

import yaml
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Draw


logger = logging.getLogger(__name__)


def save_mol_img(mol: str, name: str):
    """

    :param mol: smile format
    :param name: name of mol
    :return: png image saved in ./data/mol_imgs
    """
    cwd = Path().absolute()
    folder = f'{cwd}/data/mol_imgs/{name}'
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    m = Chem.MolFromSmiles(mol)
    Draw.MolToFile(m, p)


class Prot_Lig_Dataset(Dataset):
    def __init__(self, protein: torch.Tensor, ligand: torch.Tensor):

        self.data = ligand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, ...]

    def get_split(self, index):
        return self.data[:index, ...]
    