import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from image_retrieval.modules.base_module import BaseRetrievalModule


class NormalizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, False, device, dtype)

    def forward(self, x: TensorType["batch", "in_features"]) -> TensorType["batch", "out_features"]:
        return F.linear(F.normalize(x), F.normalize(self.weight).T, self.bias)


class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64, margin=28.6) -> None:
        super().__init__()
        self.scale = scale
        self.margin = np.radians(margin)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(
        self, logits: TensorType["batch", "classes"], labels: TensorType["batches"]
    ) -> TensorType["batch", "classes"]:
        index_to_add_margin = (torch.arange(len(labels)).long().to(labels.device), labels.long())
        value_to_add_margin = logits[index_to_add_margin]

        logits[index_to_add_margin] = torch.cos(torch.acos(value_to_add_margin) + self.margin).half()
        logits *= self.scale

        print(logits.mean())

        return self.cross_entropy(logits, labels)


class ArcFace2Module(BaseRetrievalModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__(model, data, lr, debug)

        self.loss_fn = ArcFaceLoss(data.num_classes, model.embedding_size)
        self.model.model.head.fc = NormalizedLinear(data.num_classes, model.embedding_size)
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        x = self.model.forward_features(x)
        return self.model.model.head.fc(x)

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
        # loss = self.loss_fn(output, y)
        # acc = self.acc_fn(output, y)
        #
        # self.log("val_loss", loss)
        # self.log("val_acc", acc)

        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
