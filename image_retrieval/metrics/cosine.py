from typing import TypeVar

import torch
from torchtyping import TensorType

X = TypeVar("X")
Y = TypeVar("Y")
D = TypeVar("D")


def cosine_sim(
    a: TensorType["X", "D"], b: TensorType["Y", "D"], eps=1e-8
) -> TensorType["X", "Y"]:
    """
    Compute the cosine similarity of two matrix

    :param a:  (X, D)
    :param b:  (Y, D)
    :param eps: to avoid div by zero
    :return: (X,Y)
    """
    a_norm = a / torch.clamp(a.norm(dim=1).view(-1, 1), min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1).view(-1, 1), min=eps)
    sim_mt = a_norm @ b_norm.T
    return sim_mt
