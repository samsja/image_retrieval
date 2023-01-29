from typing import NamedTuple, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy as accuracy_fn
from torchmetrics.functional import retrieval_average_precision

from image_retrieval.metrics import cosine_sim


class BaseRetrievalMixin(pl.LightningModule):
    def __init__(self, data: pl.LightningDataModule, debug=False):
        super().__init__()
        self.debug = debug
        self.data = data

        self.retrieval_metrics = RetrievalHelper()

    def on_validation_start(self) -> None:
        self.retrieval_metrics.on_validation_start(
            self.data.query_dataloader(), self.device, self
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.retrieval_metrics.on_validation_epoch_end()
        if metrics.accuracy:
            self.log("val_accuracy", metrics.accuracy, prog_bar=True)

        if metrics.map:
            self.log("val_map", metrics.map, prog_bar=True)


class RetrievalMetrics(NamedTuple):
    map: Optional[torch.Tensor] = None
    accuracy: Optional[torch.Tensor] = None


class RetrievalHelper:
    def __init__(self):
        self.query_embeddings = None
        self.query_labels = None

    def on_validation_start(
        self, query_dataloader: DataLoader, device: torch.device, model: torch.nn.Module
    ) -> None:
        query_embeddings = []
        query_labels = []
        for images, labels in query_dataloader:
            images = images.to(device)
            query_embeddings.append(model(images))
            query_labels.append(labels)

        self.query_embeddings = torch.concat(query_embeddings)
        self.query_labels = torch.concat(query_labels)
        self.val_sim = []
        self.val_labels = []

    def validation_add_features(self, features: torch.Tensor, labels: torch.Tensor):
        self.val_sim.append(cosine_sim(self.query_embeddings, features))
        self.val_labels.append(labels)

    def on_validation_epoch_end(self) -> RetrievalMetrics:
        self.val_sim = torch.concat(self.val_sim, dim=1)
        self.val_labels = torch.concat(self.val_labels)

        preds = torch.stack([label == self.val_labels for label in self.query_labels])
        map = retrieval_average_precision(self.val_sim, preds)

        _, top_index = self.val_sim.topk(1, dim=1, largest=True, sorted=True)
        top_class = self.val_labels[top_index]
        accuracy = accuracy_fn(top_class, self.query_labels.to(top_class.device))

        del self.query_embeddings
        del self.query_labels
        self.query_embeddings = None
        self.query_labels = None

        return RetrievalMetrics(map, accuracy)
