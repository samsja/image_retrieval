from typing import TypeVar

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalMixin

batch_size = TypeVar("batch_size")


class SimpleCluster(BaseRetrievalMixin):
    """
    SimSiam implementation (self supervised learning):
    https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: pl.LightningDataModule,
        lr=1e-3,
        dim=1000,
        temp_target=0.025,
        temp_anchor=0.1,
        debug=False,
    ):
        super().__init__(data, debug)

        self.lr = lr
        self.temp_target = temp_target
        self.temp_anchor = temp_anchor
        self.dim = dim

        self.model = model

        prev_dim = model.output_size
        self.fc = nn.Linear(prev_dim, dim + 1)
        self.loss_fn = torch.nn.KLDivLoss(log_target=True)

    def forward(self, x: TensorType["batch_size":...]) -> TensorType["batch_size":...]:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        target, anchor = x[0], x[1]
        self.log("fc_std", self.fc.weight.std(), prog_bar=True)
        output_target = F.log_softmax(
            self.fc(self.model(target)) / self.temp_target, dim=1
        )
        output_anchor = F.log_softmax(
            self.fc(self.model(anchor)) / self.temp_anchor, dim=1
        )

        loss = self.loss_fn(output_anchor, output_target)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
