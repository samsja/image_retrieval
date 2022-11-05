from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule


class SimSiamModule(BaseRetrievalModule):
    """
    SimSiam implementation (self supervised learning):
    https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: pl.LightningDataModule,
        lr=1e-3,
        pred_dim=512,
        dim=1024,
        debug=False,
    ):
        super().__init__(model, data, lr, debug)

        prev_dim = self.model.embedding_size

        self.encoder_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, dim),
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

        self.loss_fn = nn.CosineSimilarity(dim=1)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        x = self.model.forward_features(x)
        return self.encoder_head(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x1, x2 = x[0], x[1]

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = -(self.loss_fn(p1, z2.detach()).mean() + self.loss_fn(p2, z1.detach()).mean()) * 0.5
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            chain(self.model.parameters(), self.encoder_head.parameters(), self.predictor.parameters()),
            lr=self.lr,
        )
