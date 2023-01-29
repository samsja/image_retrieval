from typing import TypeVar

import pytorch_lightning as pl
import torch
from torch import nn
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalMixin

batch = TypeVar("batch")


class SimSiamModule(BaseRetrievalMixin):
    """
    SimSiam implementation (self supervised learning):
    https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: pl.LightningDataModule,
        lr=1e-3,
        dim=2048,
        momentum=0.9,
        weight_decay=5e-4,
        debug=False,
    ):
        super().__init__(data, debug)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        prev_dim = model.output_size
        self.head = nn.Sequential(
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer

        self.model = model

        hidden_dim = int(dim / 4)
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(hidden_dim, dim),
        )  # output layer

        self.loss_fn = nn.CosineSimilarity(dim=1)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        x = self.model(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x1, x2 = x[0], x[1]
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = (
            -(
                self.loss_fn(p1, z2.detach()).mean()
                + self.loss_fn(p2, z1.detach()).mean()
            )
            * 0.5
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
