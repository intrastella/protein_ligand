from typing import List, Union

import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn


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





def in_size(out_tensor: tuple, kernel: tuple, pad: tuple = (0, 0), dilation: tuple = (1, 1), stride: tuple = (1, 1)):
    h = lambda x: stride[0] * (x - 1) - 2 * pad[0] + dilation[0] * (kernel[0] - 1) + 1
    w = lambda x: stride[1] * (x - 1) - 2 * pad[1] + dilation[1] * (kernel[1] - 1) + 1

    h_in = h(out_tensor[0])
    w_in = w(out_tensor[1])

    return h_in, w_in


def kernel_size(out_tensor: int, stride: int = 1, pad: int = 0, dilation: int = 1):
    return 1 + (out_tensor * (1 - stride) + stride + 2 * pad - 1) / dilation


def test():
    size = in_size((124, 85), (5, 5), stride=(2, 2))
    k = kernel_size(120)

    stride_size = lambda in_s, out_s, d, k: (in_s - 1 - d * (k - 1)) / (out_s - 1)
    k_size = lambda in_s, d, s: 1 - ((- in_s + 1) / d)

    # torch.Size([4, 3, 120, 81])
    # (3, 120, 81) - torch.Size([1, 12, 78, 61])
    input1 = torch.zeros(1, 12, 78, 61)
    m1 = nn.Conv2d(3, 12, (6, 5), dilation=(8, 5))
    t1 = nn.ConvTranspose2d(12, 3, (6, 5), dilation=(8, 5))

    # (12, 100, 61) -> torch.Size([1, 24, 32, 28])
    input2 = torch.zeros(1, 24, 32, 28)
    m2 = nn.Conv2d(12, 24, (3, 3), stride=(2, 1), dilation=(6, 2))
    t2 = nn.ConvTranspose2d(24, 12, (3, 3), stride=(2, 2), dilation=(6, 3))

    # (24, 32, 28) - (48, 12, 12)
    input3 = torch.zeros(1, 48, 12, 12)
    m3 = nn.Conv2d(24, 48, (4, 3), stride=(1, 1), dilation=(6, 2))
    t3 = nn.ConvTranspose2d(24, 48, (4, 3), dilation=(6, 3))

    # (48, 12, 12) - (96, 1, 1)
    input3 = torch.zeros(1, 100, 1, 1)
    m3 = nn.Conv2d(48, 100, (12, 12), dilation=(1, 1), stride=(12, 12))
    t3 = nn.ConvTranspose2d(48, 100, (12, 12), dilation=(1, 1), stride=(12, 12))
