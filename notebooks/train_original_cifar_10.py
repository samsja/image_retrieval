# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from image_retrieval.data import CIFAR10
from image_retrieval.data.utils_ssl import get_transforms as get_transforms_ssl
from image_retrieval.models import ConvNext
from image_retrieval.modules import SimSiamModule
from image_retrieval.modules.ssl.simsiam2.module import SimSiamModule as SimSiamModule2
from image_retrieval.modules.ssl.simsiam2.simsiam import SimSiam
from image_retrieval.utils.plot import imshow
from image_retrieval.data.utils_ssl import MEAN, STD
from image_retrieval.metrics import cosine_sim
# -

import matplotlib.pyplot as plt

epoch: int = 900
batch_size: int = 512
num_workers: int = 8
data_path: str = "data_trash"
checkpoint_path: str = "checkpoints"
lr: float = 0.06
convnext_size: str = "nano"
patience: int = 900
debug: bool = False

data = CIFAR10(
    root_path=data_path,
    batch_size=batch_size,
    num_workers=num_workers,
    debug=debug,
    transform=get_transforms_ssl
)

# +
#model = ConvNext(pretrained= False, size=convnext_size)
#module = SimSiamModule(model, data, lr, debug=debug)
# -

model = SimSiam()
module = SimSiamModule(model, data, lr)

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=patience, strict=False),
    ModelCheckpoint(dirpath=checkpoint_path),
]

trainer_args = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": epoch,
    "precision": 16,
    "callbacks": callbacks,
}

wandb_logger = WandbLogger(project="image_retrieval", save_dir="lightning_logs")
trainer_args["logger"] = wandb_logger,

# + tags=[]
trainer = pl.Trainer(**trainer_args)

trainer.fit(module, data)

# -


