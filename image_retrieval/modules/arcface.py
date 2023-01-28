from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torchtyping import TensorType

from image_retrieval.models import AbstractModel
from image_retrieval.modules.base_module import BaseRetrievalMixin

batch = TypeVar("batch")


class ArcFaceModule(BaseRetrievalMixin):
    def __init__(
        self, model: AbstractModel, data: pl.LightningDataModule, lr=1e-3, debug=False
    ):
        super().__init__(data, debug)
        self.lr = lr
        self.model = model
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
        features = self(x)
        loss = self.loss_fn(features, y)

        self.log("val_loss", loss)

        features = torch.nn.functional.normalize(features)

        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
