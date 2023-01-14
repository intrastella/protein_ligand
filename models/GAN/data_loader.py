import logging
from pathlib import Path
from typing import List
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data.MolEnums import MolType
from data.MolEnums import MolDataStruct
from data.molecular_matrix import Smile2Mat

cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def get_loader(db_name: str,
               mol_struct: str,
               mol_type: str,
               batch_size: int,
               table_names: Union[str, List[str]] = None,
               amount: int = None,
               path2data: str = None,
               db_insertion: bool = True):

    if mol_struct == MolDataStruct.SMILE.value:
        converter = Smile2Mat(path2mol=path2data)
        mol_feat = converter.get_smiles_dataset(db_name, MolType(mol_type), table_names, amount, db_insertion=True)
        train_data = SMILE_Dataset(mol_feat)
        trainer = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        return trainer

    else:
        pass


class SMILE_Dataset(Dataset):
    def __init__(self, mol_tensor: torch.Tensor):

        self.data = mol_tensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, ...]
