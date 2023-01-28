import torch

from image_retrieval.models import ConvNext


def test_convnext():

    with torch.no_grad():

        model = ConvNext(pretrained=False)

        inputs = torch.zeros(10, 3, 224, 224)

        features = model(inputs)

        assert features.shape == (10, 640)
