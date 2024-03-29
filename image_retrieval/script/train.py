from importlib import import_module

import pytorch_lightning as pl
import torch
import wandb as wandb_lib
from pytorch_lightning.callbacks import ModelCheckpoint

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from typer import Typer

app = Typer(pretty_exceptions_enable=False)
torch.set_float32_matmul_precision("medium")


@app.command()
def train(
    epoch: int,
    batch_size: int = 32,
    num_workers: int = 4,
    module: str = "MetricLearningModule",
    aug: str = "BasicAugmentation",
    backbone: str = "ConvNextNano",
    data_path: str = "data_trash",
    dataset: str = "CIFAR100",
    checkpoint_path: str = "checkpoints",
    lr: float = 1e-3,
    # patience: int = 10,
    pretrained: bool = False,
    gpus: int = 1,
    debug: bool = False,
    project_name: str = "image_retrieval",
    no_wandb: bool = False,
):

    augmentation = getattr(import_module("image_retrieval.augmentation"), aug)

    data_cls = getattr(import_module("image_retrieval.data"), dataset)
    data = data_cls(
        root_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        transform=augmentation,
    )

    model_cls = getattr(import_module("image_retrieval.models"), backbone)
    model = model_cls(pretrained=not debug and pretrained)

    module_class = getattr(import_module("image_retrieval.modules"), module)

    module_ = module_class(model, data, lr, debug=debug)

    callbacks = [
        # EarlyStopping(monitor="val_loss", mode="min",
        # patience=patience, strict=False),
        ModelCheckpoint(dirpath=checkpoint_path),
    ]

    if debug or no_wandb:
        wandb_lib.init(mode="disabled")

    wandb_logger = WandbLogger(project=project_name, save_dir="lightning_logs")

    wandb_logger.experiment.config["lr"] = lr
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["backbone"] = backbone
    wandb_logger.experiment.config["module"] = module
    wandb_logger.experiment.config["augmentation"] = aug
    wandb_logger.experiment.config["dataset"] = dataset
    wandb_logger.experiment.config["gpus"] = gpus

    trainer_args = {
        "accelerator": "gpu",
        "devices": gpus,
        "max_epochs": epoch,
        "precision": 16,
        "callbacks": callbacks,
        "logger": wandb_logger,
    }

    if debug:
        trainer_args["log_every_n_steps"] = 1

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(module_, data)


if __name__ == "__main__":
    app()
