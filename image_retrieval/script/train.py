from importlib import import_module

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from typer import Typer

from image_retrieval.data import CIFAR100
from image_retrieval.data.utils_ssl import get_transforms as get_transforms_ssl
from image_retrieval.models import ConvNext

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    epoch: int,
    batch_size: int = 32,
    num_workers: int = 4,
    module: str = "SoftMaxModule",
    data_path: str = "data_trash",
    checkpoint_path: str = "checkpoints",
    lr: float = 1e-3,
    convnext_size: str = "nano",
    patience: int = 10,
    ssl: bool = False,
    debug: bool = False,
):

    data = CIFAR100(
        root_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        transform=get_transforms_ssl if ssl else None,
    )

    model = ConvNext(pretrained=not (debug), size=convnext_size)

    module_class = getattr(import_module("image_retrieval.modules"), module)
    module = module_class(model, data, lr, debug=debug)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=patience, strict=False),
        ModelCheckpoint(dirpath=checkpoint_path),
    ]

    if debug:
        wandb.init(mode="disabled")

    wandb_logger = WandbLogger(project="image_retrieval", save_dir="lightning_logs")

    wandb_logger.experiment.config["lr"] = lr
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["convnext_size"] = convnext_size
    wandb_logger.experiment.config["module"] = module

    trainer_args = {
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": epoch,
        "precision": 16,
        "callbacks": callbacks,
        "logger": wandb_logger,
    }

    if debug:
        trainer_args["log_every_n_steps"] = 1

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(module, data)


if __name__ == "__main__":
    app()
