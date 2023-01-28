from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalMixin

batch = TypeVar("batch")


class MetricLearningModule(BaseRetrievalMixin):
    def __init__(
        self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False
    ):
        super().__init__(data, debug)
        self.model = model
        self.loss_fn = TripletMarginLoss()
        self.lr = lr

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model(x)
        loss = self.loss_fn(features, y)

        self.log("val_loss", loss)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
