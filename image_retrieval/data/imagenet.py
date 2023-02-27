from typing import Optional, Type

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet as ImageNetDataset

from image_retrieval.augmentation import AbstractAugmentation

_IMAGE_SHAPE = (224, 224)


class ImageNet(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        transform: Type[AbstractAugmentation],
        batch_size: int = 32,
        num_workers: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size if not debug else 8
        self.num_workers = num_workers if not debug else 0
        self.data_path = root_path
        self.debug = debug
        self.transform = transform(image_shape=_IMAGE_SHAPE)

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = ImageNetDataset(
            self.data_path, transform=self.transform.get_transform(), split="train"
        )

        self.test_dataset = ImageNetDataset(
            self.data_path, transform=self.transform.get_transform_val(), split="val"
        )
        self.test_dataset, self.query_dataset = torch.utils.data.random_split(
            self.test_dataset,
            [
                len(self.test_dataset) - 100,
                100,
            ],
            generator=torch.Generator().manual_seed(42),
        )

        if self.debug:
            for dataset in [self.train_dataset, self.test_dataset]:
                dataset.__class__ = _debug_dataset(self.batch_size, dataset.__class__)

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

    def query_dataloader(self):
        return DataLoader(
            self.query_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        return 1000


def _debug_dataset(max_len: int, origin_class):
    class _DebugDataset(origin_class):
        def __len__(self):
            return max_len

    return _DebugDataset
