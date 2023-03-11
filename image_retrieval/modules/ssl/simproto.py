from typing import TypeVar

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalMixin

batch_size = TypeVar("batch_size")


class SimProto(BaseRetrievalMixin):
    def __init__(
        self,
        model: torch.nn.Module,
        data: pl.LightningDataModule,
        lr=1e-3,
        dim=20,
        temp_target=0.025,
        temp_anchor=0.1,
        lambda_=0.1,
        lr_fc=0.01,
        debug=False,
    ):
        super().__init__(data, debug)

        self.lr = lr
        self.lr_fc = lr_fc
        self.temp_target = temp_target
        self.temp_anchor = temp_anchor
        self.lambda_ = lambda_
        self.dim = dim

        self.model = model

        prev_dim = model.output_size
        self.fc = nn.Linear(prev_dim, dim)
        self.loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    def forward(self, x: TensorType["batch_size":...]) -> TensorType["batch_size":...]:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        target, anchor = x[0], x[1]

        self.log("fc_std", self.fc.weight.std(), prog_bar=True)

        logits_target = F.log_softmax(
            self.fc(self.model(target)) / self.temp_target, dim=1
        ).detach()

        logits_anchor = F.log_softmax(
            self.fc(self.model(anchor)) / self.temp_anchor, dim=1
        )

        class_std = logits_anchor.argmax(dim=1).float().std()
        self.log("class_std", class_std)

        me_max = (
            self.lambda_ * logits_anchor.mean()
        )  # regularize to avoid prototypes to collapse
        self.log("me_max", me_max)

        kl_div = self.loss_fn(logits_anchor, logits_target)
        self.log("kl_div", kl_div)

        loss = kl_div - me_max
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.lr},
                {"params": self.fc.parameters(), "lr": self.lr_fc},
            ]
        )
