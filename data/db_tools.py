import datetime
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Date
from sqlalchemy import Float
from sqlalchemy import INT
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker
import torch

from data.MolEnums import MolType
from utils import df_to_tensor


cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class SQL_Session:

    def __init__(self, USER: str,
                     PASSWORD: str,
                     HOST: str,
                     PORT: int,
                     DATABASE: str):

        connection_string = "mysql+pymysql://%s:%s@%s:%s/%s" % (USER, PASSWORD, HOST, PORT, DATABASE)
        self.sqlEngine = create_engine(connection_string)
        self.dbConnection = self.sqlEngine.connect()
        self.META_DATA = MetaData(bind=self.sqlEngine)

    def create_feat_tables(self, table_name: str, columns: list):
        meta = MetaData()
        Table(
            table_name, meta, Column(f'{table_name}_id', INT, primary_key=True, autoincrement=True), *[Column(str(name), Float) for name in columns]
        )

        meta.create_all(self.sqlEngine)
        logger.info(f"New table for {table_name} created.")

    def insert_feat_tables(self, table_name: str, rec_vals: Dict):
        Session = sessionmaker(bind=self.sqlEngine)
        session = Session()
        feat_table = Table(table_name, self.META_DATA, schema="ligand_protein", autoload=True, autoload_with=self.sqlEngine)

        stmt = insert(feat_table).values(**rec_vals)
        session.execute(stmt)
        session.commit()

    def quick_insert(self, table_name: str, in_tensor: torch.Tensor, columns: list):
        for i in range(len(in_tensor[:, 0])):
            dict_data = {str(columns[j]): in_tensor[i, j].item() for j in range(len(in_tensor[0, :]))}
            self.insert_feat_tables(table_name, dict_data)

    def get_data_from_db(self, db_name: str, table_name: str, amount: int, condition: str) -> torch.Tensor:
        cmd = f"SELECT * FROM {db_name}.{table_name}"

        if condition and (condition != 'None'):
            cmd += f" WHERE {condition} "
        if amount and (amount != 'None'):
            cmd += f" LIMIT {amount}"

        frame = pd.read_sql(cmd, self.dbConnection)
        frame.drop([f'{table_name}_id', 'mol_id'], axis=1, inplace=True)
        num = pd.read_sql(f'SELECT MAX(DISTINCT mol_id) FROM {db_name}.mol_rec', self.dbConnection)
        total_rec_num = 3 # num.to_numpy()[0][0]

        df_tensor = df_to_tensor(frame)
        df_tensor = df_tensor.reshape(total_rec_num, int(df_tensor.shape[0] / total_rec_num), df_tensor.shape[1])
        return df_tensor.unsqueeze(3)

    def insert_in2_mol_rec(self,
                           mol_type: MolType,
                           date: datetime.date.today(),
                           adj_mat: bool,
                           atom_feat: bool,
                           bond_feat: bool):

        Session = sessionmaker(bind=self.sqlEngine)
        session = Session()
        mol_rec = Table('mol_rec', self.META_DATA, schema="ligand_protein", autoload=True, autoload_with=self.sqlEngine)

        stmt = insert(mol_rec).values(mol_type=mol_type.value, date=date, adj_mat=adj_mat, atom_feat=atom_feat, bond_feat=bond_feat)
        session.execute(stmt)
        session.commit()

    def panda_in2_db(self, df, table_name):
        df.to_sql(table_name, self.dbConnection, schema=None, if_exists='append')

    def get_record_count(self, db_name: str, table_name: str):
        Session = sessionmaker(bind=self.sqlEngine)
        session = Session()
        table = Table(table_name, self.META_DATA, schema=db_name, autoload=True, autoload_with=self.sqlEngine)
        count = session.query(table).count()
        return count

    def df_exists(self, table: str) -> bool:
        return sqlalchemy.inspect(self.sqlEngine).has_table(table)

    def close(self):
        self.dbConnection.close()

    def set_up_mol_table(self):

        meta = MetaData()
        Table(
            'mol_rec', meta,
            Column('mol_id', Float, primary_key=True, autoincrement=True),
            Column('mol_type', String(100)),
            Column('date', Date),
            Column('adj_mat', String(10)),
            Column('atom_feat', String(10)),
            Column('bond_feat', String(10))
        )
        meta.create_all(self.sqlEngine)
        logger.info(f"New table for molecule records created.")
