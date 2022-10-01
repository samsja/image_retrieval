import torch

from image_retrieval.models import ConvNext


def test_convnext():

    model = ConvNext(pretrained=False)

    assert model(torch.zeros(10, 3, 224, 224)).shape == (10, 1000)
    assert model.forward_features(torch.zeros(10, 3, 224, 224)).shape == (10, 640)
