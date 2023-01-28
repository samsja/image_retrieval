from typing import TypeVar

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchtyping import TensorType

from image_retrieval.models import AbstractModel
from image_retrieval.modules.base_module import BaseRetrievalMixin

batch_size = TypeVar("batch_size")


class SoftMaxModule(BaseRetrievalMixin):
    def __init__(
        self,
        model: AbstractModel,
        data: pl.LightningDataModule,
        lr=1e-3,
        debug=False,
    ):
        super().__init__(data, debug)
        self.model = model
        self.lr = lr
        self.fc = nn.Linear(model.output_size, data.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x: TensorType["batch_size":...]) -> TensorType["batch_size":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        logits = self.fc(features)
        loss = self.loss_fn(logits, y)
        acc = self.acc_fn(logits, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model(x)
        output = self.fc(features)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        acc = self.acc_fn(output, y)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
