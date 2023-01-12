import datetime
import logging
from pathlib import Path
from typing import List, Union
import numpy as np

import torch
import yaml
from atom3d.datasets import LMDBDataset

from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from tqdm import tqdm

from data.MolEnums import Degree, FormalCharge, Hybridization, Chiral, TotalNumHs, Stereo, InRing, IsAromatic, \
    IsConjugated, FeatTables, MolType
from data.db_tools import SQL_Session
from utils import one_hot_enc, pad_tensor, pad_batch


cwd = Path().absolute()
logging.basicConfig(filename=f"{cwd}/std.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Smile2Mat:

    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                                   'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                                   'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    def __init__(self, smiles: list = None, path2mol: str = None):
        self.mol = None
        self.smiles = smiles
        self.path2mol = path2mol

    def get_atom_features(self, atom,
                          use_chirality=True,
                          hydrogens_implicit=True):

        if not hydrogens_implicit:
            self.permitted_list_of_atoms += ['H']

        atom_type_enc = one_hot_enc(str(atom.GetSymbol()), self.permitted_list_of_atoms)
        n_heavy_neighbors_enc = one_hot_enc(int(atom.GetDegree()), Degree.values())
        formal_charge_enc = one_hot_enc(int(atom.GetFormalCharge()), FormalCharge.values())
        hybridisation_type_enc = one_hot_enc(str(atom.GetHybridization()), Hybridization.values())
        in_ring_enc = one_hot_enc(atom.IsInRing(), InRing.values())
        is_aromatic_enc = one_hot_enc(atom.GetIsAromatic(), IsAromatic.values())

        atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

        atom_features_vector = atom_type_enc + \
                               n_heavy_neighbors_enc + \
                               formal_charge_enc + \
                               hybridisation_type_enc + \
                               in_ring_enc + \
                               is_aromatic_enc + \
                               atomic_mass_scaled + \
                               vdw_radius_scaled + \
                               covalent_radius_scaled

        if use_chirality:
            chirality_type_enc = one_hot_enc(str(atom.GetChiralTag()), Chiral.values())
            atom_features_vector += chirality_type_enc

        if hydrogens_implicit:
            n_hydrogens_enc = one_hot_enc(int(atom.GetTotalNumHs()), TotalNumHs.values())
            atom_features_vector += n_hydrogens_enc

        return np.array(atom_features_vector)

    def get_bond_features(self, bond, use_stereochemistry=True):

        bond_type_enc = one_hot_enc(bond.GetBondType(), self.permitted_list_of_bond_types)
        bond_is_conj_enc = one_hot_enc(bond.GetIsConjugated(), IsConjugated.values())
        bond_is_in_ring_enc = one_hot_enc(bond.IsInRing(), InRing.values())

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry:
            stereo_type_enc = one_hot_enc(str(bond.GetStereo()), Stereo.values())
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)

    def smiles2data(self, smiles: list, db_name: str, mol_type: MolType, db_insertion: bool = False) -> torch.Tensor:
        """

        :param smiles:
        :param db_name:
        :param mol_type:
        :return:
        """

        data_list = []

        for smile in smiles:
            self.mol = Chem.MolFromSmiles(smile)

            (n_nodes, n_edges, n_node_features, n_edge_features) = self.get_sizes()
            mol_nodes = self.get_mol_nodes(n_nodes, n_node_features)
            mol_edges = self.get_mol_edges(n_edges, n_edge_features)

            adj_mat = GetAdjacencyMatrix(self.mol)
            adj_mat = torch.from_numpy(adj_mat)

            padded = pad_tensor([adj_mat, mol_nodes, mol_edges])
            data_list.append(padded)

        dataset = pad_batch(data_list)
        if db_insertion:
            self.db_ingestion(dataset, mol_type, db_name)
        return torch.permute(dataset, (0, 3, 1, 2))

    def get_sizes(self):
        n_nodes = self.mol.GetNumAtoms()
        n_edges = 2 * self.mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        return n_nodes, n_edges, n_node_features, n_edge_features

    def get_mol_nodes(self, n_nodes: int, n_node_features: int) -> torch.tensor:
        mol_nodes = np.zeros((n_nodes, n_node_features))
        for atom in self.mol.GetAtoms():
            mol_nodes[atom.GetIdx(), :] = self.get_atom_features(atom)
        return torch.from_numpy(mol_nodes)

    def get_mol_edges(self, n_edges: int, n_edge_features: int) -> torch.Tensor:
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(self.mol))
        mol_edges = np.zeros((n_edges, n_edge_features))
        for (k, (i, j)) in enumerate(zip(rows, cols)):
            mol_edges[k] = self.get_bond_features(self.mol.GetBondBetweenAtoms(int(i), int(j)))
        return torch.from_numpy(mol_edges)

    def db_ingestion(self, dataset: torch.Tensor, mol_type: MolType, db_name: str):
        """

        :param dataset:
        :param mol_type:
        :param db_name:
        :return:
        """
        session = self.create_connection(db_name)
        rec_count = 0
        if session.df_exists('mol_rec'):
            rec_count = session.get_record_count(db_name, 'mol_rec')
        else:
            session.set_up_mol_table()

        mol_id_index = torch.cat([torch.ones(1, dataset.shape[1], 1, dataset.shape[3]) * (rec_count + i) for i in range(1, dataset.shape[0]+1)], dim=0)
        dataset = torch.cat([dataset, mol_id_index], dim=2)

        features = [dataset[..., i] for i in range(3)]

        atom_bond = self.get_col_names()
        bond_col = atom_bond[1] + [f'fill{i}' for i in range(len(atom_bond[0]) - len(atom_bond[1]))] + ['mol_id']
        atom_col = atom_bond[0] + ['mol_id']
        adj_col = [i for i in range(len(atom_bond[0]))] + ['mol_id']
        col_names = [adj_col, atom_col, bond_col]

        for _ in range(dataset.shape[0]):
            session.insert_in2_mol_rec(mol_type, datetime.date.today(), True, True, True)
        logger.info('Insertion into molecule records completed.')

        for table, feat, col_name in zip(FeatTables.values(), features, col_names):
            if not session.df_exists(table):
                session.create_feat_tables(table, col_name)
            session.quick_insert(table, feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2]), col_name)
        logger.info(f"Insertion into feature tables completed.")
        session.close()

    def create_connection(self, db_name: str):
        log_data = yaml.safe_load(Path('data/data_conf.yaml').read_text())
        log_data['Credentials']['DATABASE'] = db_name
        return SQL_Session(**log_data['Credentials'])

    def get_smiles_dataset(self,
                           db_name: str = None,
                           mol_type: MolType = None,
                           table_names: Union[str, List[str]] = None,
                           amount: int = None,
                           db_insertion: bool = False):
        """

        :param db_name:
        :param mol_type:
        :param table_names:
        :param amount:
        :param path_LMDB:
        :return:
        """

        if self.path2mol or self.smiles:
            logger.info("Creating dataset from smiles format.")
            if self.path2mol:
                # needs to be changed
                dataset = LMDBDataset(self.path2mol)
                if not self.smiles:
                    self.smiles = []
                if not amount:
                    amount = len(dataset)
                for i in tqdm(range(amount)):
                    self.smiles.append(dataset[i]['smiles'])
                logger.info(f"{len(dataset)} smiles have been loaded from path.")

            mol_feat = self.smiles2data(self.smiles, db_name, mol_type=mol_type, db_insertion=db_insertion)

        else:
            logger.info("Fetching dataset from SQL.")
            session = self.create_connection(db_name)

            mol_feat = []
            condition = f"mol_id IN (SELECT mol_id FROM {db_name}.mol_rec WHERE mol_type = '{mol_type}')"
            if table_names and (table_names != 'None'):
                for name in table_names:
                    mol_feat.append(session.get_data_from_db(db_name, name, amount, condition))
            else:
                for name in FeatTables.values():
                    mol_feat.append(session.get_data_from_db(db_name, name, amount, condition))
            mol_feat = torch.cat(mol_feat, dim=3)
            mol_feat = torch.permute(mol_feat, (0, 3, 1, 2))
            session.close()

        return mol_feat

    def get_col_names(self):
         atom_feat = self.permitted_list_of_atoms + \
            [f'{deg}_deg' for deg in Degree.values()] + \
            [f'{charge}_charge' for charge in FormalCharge.values()] + \
            [f'{name}_hyb' for name in Hybridization.values()] + \
            [f'{str(InRing.values()[i])}_ring' for i in range(2)] + \
            [f'{str(IsAromatic.values()[i])}_aromatic' for i in range(2)] + \
            ['atomic_mass_scaled', 'vdw_radius_scaled', 'covalent_radius_scaled'] + \
            Chiral.values() + \
            [f'{numH}_Hs' for numH in TotalNumHs.values()]

         bond_feat = ['single', 'double', 'tripple', 'aromatic'] + \
            [f'{str(IsConjugated.values()[i])}_conjug' for i in range(2)] + \
            [f'{str(InRing.values()[i])}_ring' for i in range(2)] + \
            Stereo.values()

         return atom_feat, bond_feat
