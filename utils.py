from typing import List, Union

import numpy as np
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


def convTrans(out_tensor: tuple, kernel: tuple, pad: tuple = (0, 0), dilation: tuple = (1, 1), stride: tuple = (1, 1)):
    h = lambda x: stride[0] * (x - 1) - 2 * pad[0] + dilation[0] * (kernel[0] - 1) + 1
    w = lambda x: stride[1] * (x - 1) - 2 * pad[1] + dilation[1] * (kernel[1] - 1) + 1

    h_in = h(out_tensor[0])
    w_in = w(out_tensor[1])

    return h_in, w_in


def test(in_tensor: torch.Tensor, in_C=None, out_C=None, k=None, d=None, s=None) -> torch.Tensor:
    res = fast(in_tensor, in_C, out_C, k, d, s)
    return res


def fast(x, in_C=None, out_C=None, k=None, d=None, s=None):
    t1 = nn.Conv2d(3, 3, (1, 4), dilation=(1, 6), stride=(1, 1))
    t2 = nn.Conv2d(3, 3, (1, 4), dilation=(1, 1), stride=(1, 2))
    m = nn.AvgPool2d((4, 4), padding=2, stride=1)
    t = nn.Conv2d(in_C, out_C, (k, k), dilation=(d, d), stride=(s, s))

    x = t1(x)
    x = m(x)

    x = t2(x)
    x = m(x)

    x = t(x)
    x = m(x)
    return x


def stride_up_conv(in_T, out_T, k, d):
    return ((out_T - 1) - d * (k - 1)) / (in_T - 1)


def stride_conv(in_T, out_T, k, d):
    if out_T == 1:
        res = in_T - d * (k - 1)
    else:
        res = ((in_T - 1) - d * (k - 1)) / (out_T - 1)
    return res


def d_up_conv(in_T, out_T, k):
    return int(((out_T - 1) - (in_T - 1)) / (k - 1))


def d_conv(in_T, out_T, k):
    return ((in_T - 1) - (out_T - 1)) / (k - 1)


def find_val(in_T, out_T, in_C, out_C, side):
    sides = {'left': 2, 'right': 3}
    possible_parameters = []

    for k in np.arange(2, 10, 1, dtype=int):
        d_max = d_conv(in_T, out_T, k)
        d_limit = min(d_max, 8)
        for d in np.arange(1, d_limit + 1, 1, dtype=int):
            s = stride_conv(in_T, out_T, k, d)
            s = int(s)

            res = test(torch.ones(5, 3, 81, 81), in_C, out_C, k, d, s)
            txt = "SHAPE: {shape}, in_T: {in_T}, res: {res}, k: {k}, d:{d}, s:{s}".format(shape=res.shape, in_T=in_T,
                                                                                          res=res.shape[sides[side]], k=k, d=d,
                                                                                          s=s)
            print(txt)

            if out_T == res.shape[sides[side]]:
                print(f' \n SOLUTION: {(k, d, s)}')
                possible_parameters.append((k, d, s))

    if len(possible_parameters) == 0:
        print('No solution.')
    else:
        print(possible_parameters)


def main():
    find_val(32, 1, 3, 3, 'right')
    # out = fast(torch.ones(5, 3, 120, 81))
    # print(out.shape)


if __name__ == '__main__':
    main()
