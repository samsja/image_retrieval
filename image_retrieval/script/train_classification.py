import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from typer import Typer

from image_retrieval.data import CIFAR100
from image_retrieval.models import ConvNext
from image_retrieval.modules import ClassificationModule

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    epoch: int,
    batch_size: int = 32,
    num_workers: int = 4,
    data_path: str = "data_trash",
    checkpoint_path: str = "checkpoints",
    convnext_size: str = "nano",
    patience: int = 10,
    debug: bool = False,
):

    data = CIFAR100(root_path=data_path, batch_size=batch_size, num_workers=num_workers, debug=debug)

    model = ConvNext(pretrained=not (debug), size=convnext_size)
    module = ClassificationModule(model, data, debug=debug)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=patience, strict=False),
        ModelCheckpoint(dirpath=checkpoint_path),
    ]

    trainer_args = {
        "gpus": 1,
        "max_epochs": epoch,
        "precision": 16,
        "callbacks": callbacks,
    }

    if debug:
        trainer_args["log_every_n_steps"] = 1

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(module, data)


if __name__ == "__main__":
    app()
