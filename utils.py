from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn


def create_exp_folder():
    cwd = Path().absolute()
    now = datetime.now()
    exp_id = now.strftime("%d-%m-%Y_%H-%M-%S")
    folder = f'{cwd}/exp_{exp_id}'
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def one_hot_enc(elem: Union[str, int, float, bool], permitted: List[str] = None) -> List[Union[int, float]]:
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if elem not in permitted:
        elem = permitted[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: elem == s, permitted))]
    return binary_encoding


def pad_n_mask(mat: List[list]) -> torch.Tensor:
    mask_info = torch.zeros((len(mat), max(set([len(row) for row in mat]))))

    padded_mat = tf.keras.preprocessing.sequence.pad_sequences(
        mat, padding="post", value=-1
    )

    padded_mat = torch.tensor(padded_mat).to(torch.float32)
    mask = padded_mat.ge(-.5)
    padded_mat[padded_mat == -1.] = 0
    mask_info.masked_fill_(mask, 1.)

    return torch.stack((padded_mat, mask_info), dim=2)


def pad_tensor(tensors: List[torch.Tensor]):
    padded = []
    xaxis_sizes = [f_tensor.shape[0] for f_tensor in tensors]
    yaxis_sizes = [f_tensor.shape[1] for f_tensor in tensors]

    max_x = max(xaxis_sizes)
    max_y = max(yaxis_sizes)

    for f_tensor, xs, ys in zip(tensors, xaxis_sizes, yaxis_sizes):
        pad = (0, max_y-ys, 0, max_x-xs)
        padded.append(F.pad(f_tensor, pad, "constant", 0))
    return torch.stack(padded, dim=2)


def pad_batch(tensors: List[torch.Tensor]):
    padded = []
    axis_sizes = [f_tensor.shape[0] for f_tensor in tensors]
    max_s = max(axis_sizes)

    for f_tensor, a in zip(tensors, axis_sizes):
        pad = (0, 0, 0, 0, 0, max_s-a)
        padded.append(F.pad(f_tensor, pad, "constant", 0))
    return torch.stack(padded, dim=0)


def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)
