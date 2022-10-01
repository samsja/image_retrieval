from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100 as CIFAR100torch

from image_retrieval.data.utils import get_transforms

_IMAGE_SHAPE_VAL = (32, 32)
_IMAGE_SHAPE = (32, 32)


class CIFAR100(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size if not debug else 8
        self.num_workers = num_workers
        self.data_path = root_path
        self.debug = debug

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = CIFAR100torch(
            self.data_path,
            transform=get_transforms(_IMAGE_SHAPE),
            train=True,
            download=True,
        )

        self.test_dataset = CIFAR100torch(
            self.data_path,
            transform=get_transforms(_IMAGE_SHAPE_VAL, augmentation=False),
            train=False,
        )

        if self.debug:
            for dataset in [self.train_dataset, self.test_dataset]:
                dataset.data = dataset.data[0 : self.batch_size]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
