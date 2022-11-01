from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule


class Encoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=out_dim, out_features=out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)

        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.linear3 = nn.Linear(in_features=out_dim, out_features=out_dim)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.linear3(x))
        return x


class SimSiamModule(BaseRetrievalModule):
    """
    SimSiam implementation (self supervised learning):
    https://arxiv.org/abs/2011.10566
    """

    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__(model, data, lr, debug)

        self.encoder = Encoder(self.model.embedding_size)
        self.loss_fn = nn.CosineSimilarity(dim=1)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model.forward_features(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[0], x[1]

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        p1 = self.encoder(z1).detach()
        p2 = self.encoder(z2).detach()

        loss = -(self.loss_fn(p1, z2).mean() + self.loss_fn(p2, z1).mean()) * 0.5
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(chain(self.model.parameters(), self.encoder.parameters()), lr=self.lr)
