import pytest

from image_retrieval.script.train import train


@pytest.mark.parametrize(
    "module", ["ArcFaceModule", "SoftMaxModule", "MetricLearningModule"]
)
def test_full_training(module, data_path):
    train(epoch=1, module=module, data_path=data_path, debug=True)


@pytest.mark.parametrize(
    "module,augmentation",
    [
        ("SimSiamModule", "SSLAugmentation"),
        ("SimSiamModule", "SSLAugmentation2"),
        ("FastSiamModule", "FastSiamSSL"),
    ],
)
def test_full_training_ssl(module, augmentation, data_path):
    train(epoch=1, aug=augmentation, module=module, data_path=data_path, debug=True)
