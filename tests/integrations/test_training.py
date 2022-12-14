import pytest

from image_retrieval.script.train import train


@pytest.mark.parametrize("module", ["ArcFaceModule", "SoftMaxModule", "MetricLearningModule"])
def test_full_training(module, data_path):
    train(epoch=1, module=module, data_path=data_path, debug=True)


@pytest.mark.parametrize("module", ["SimSiamModule"])
def test_full_training_ssl(module, data_path):
    train(epoch=1, ssl=True, module=module, data_path=data_path, debug=True)
