import torch

from image_retrieval.models import ConvNext


def test_convnext():

    with torch.no_grad():

        model = ConvNext(pretrained=False)

        inputs = torch.zeros(10, 3, 224, 224)

        features = model.forward_features(inputs)
        outputs = model(inputs)

        assert outputs.shape == (10, 1000)

        assert features.shape == (10, 640)
        assert (model.forward_from_features(features) == outputs).all()
