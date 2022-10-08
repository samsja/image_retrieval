from itertools import chain

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule


class ArcFaceModule(BaseRetrievalModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__(model, data, lr, debug)

        self.loss_fn = ArcFaceLoss(data.num_classes, model.embedding_size)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model.forward_features(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        loss = self.loss_fn(features, y)

        self.log("val_loss", loss)

        features = torch.nn.functional.normalize(features)

        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(chain(self.model.parameters(), self.loss_fn.parameters()), lr=self.lr)
