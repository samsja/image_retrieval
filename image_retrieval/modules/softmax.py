from typing import TypeVar

import pytorch_lightning as pl
import torch
import torchmetrics
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule

batch = TypeVar("batch")


class SoftMaxModule(BaseRetrievalModule):
    def __init__(
        self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False
    ):
        super().__init__(model, data, lr, debug)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        output = self.model.forward_from_features(features)
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
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
