import pytorch_lightning as pl

from image_retrieval.data import CIFAR100
from image_retrieval.modules import ClassificationModule


def test_full_training(data_path):

    data = CIFAR100(root_path=data_path, debug=True)

    model = ClassificationModule(debug=True)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        precision=16,
    )

    trainer.fit(model, data)
