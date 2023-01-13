import re
from pathlib import Path
from typing import Union

from rdkit.Chem import PandasTools
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertTokenizer

def get_tokens(file: Union[str, Path]):
    """
    sdf file from BindingDB is used
    """
    
    df = PandasTools.LoadSDF(file, embedProps=True, molColName=None, smilesName='smiles')
    
    seuences = df['BindingDB Target Chain Sequence'].to_list()
    max_len = max([len(list(seq.replace(' ', '')) for seq in sequences])
    tensor_list = []
    
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    
    for seq in sequences:
        sequence = re.sub(r"[UZOB]", "X", seq)
        encoded_input = tokenizer(sequence, return_tensors='pt', padding=True, max_length=max_len)
        tensor_list.append(model(**encoded_input))
        
    protein_dataset = torch.stack(tensor_list)
    return protein_dataset





    


