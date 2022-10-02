import torch

from image_retrieval.metrics import cosine_sim


def test_cosine_sim():

    a = torch.rand(2, 6)
    b = torch.rand(4, 6)

    sim = cosine_sim(a, b)

    assert sim.shape == (2, 4)
