import os
import logging
from pathlib import Path
from typing import List, Union

import yaml
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from atom3d.datasets import LMDBDataset

from data.db_tools import SQL_Session
from data.molecular_matrix import Smile2Mat


logger = logging.getLogger(__name__)


class Prot_Lig_Dataset(Dataset):
    def __init__(self, protein: torch.Tensor, ligand: torch.Tensor):

        self.data = ligand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, ...]

    def get_split(self, index):
        return self.data[:index, ...]
    