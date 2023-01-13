from pathlib import Path
from typing import Union

from rdkit.Chem import PandasTools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


def get_tokens(file: Union[str, Path]):
    # file = '/home/stella/Downloads/BindingDB_BindingDB_2D.sdf'
    df = PandasTools.LoadSDF(file, embedProps=True, molColName=None, smilesName='smiles')

    x = df['BindingDB Target Chain Sequence'].iloc[0]

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(x)
    train_sequences = tokenizer.texts_to_sequences(x)
    maxlen = max([len(x) for x in train_sequences])
    train_padded = pad_sequences(train_sequences, maxlen=len(list(y)))


    tz = BertTokenizer.from_pretrained("bert-base-cased")
    sent = x

    # Encode the sentence
    encoded = tz.encode_plus(
        text=sent,
        add_special_tokens=True,
        max_length=len(list(y)),
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']


