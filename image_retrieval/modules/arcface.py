from itertools import chain
from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule

batch = TypeVar("batch")


class ArcFaceModule(BaseRetrievalModule):
    def __init__(
        self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False
    ):
        super().__init__(model, data, lr, debug)
        self.loss_fn = ArcFaceLoss(data.num_classes, model.output_size)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)

        self.log("train_loss", loss)

        return loss

    def on_validation_start(self) -> None:
        # here we need to normalize the query as well
        super().on_validation_start()
        self.retrieval_metrics.query_embeddings = torch.nn.functional.normalize(
            self.retrieval_metrics.query_embeddings
        )  #

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model(x)
        loss = self.loss_fn(features, y)

        self.log("val_loss", loss)

        features = torch.nn.functional.normalize(features)

        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            chain(self.model.parameters(), self.loss_fn.parameters()), lr=self.lr
        )
