import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import retrieval_average_precision

from image_retrieval.metrics import cosine_sim


class BaseRetrievalModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__()
        self.lr = lr
        self.model = model
        self.debug = debug
        self.data = data

        self.retrieval_metrics = RetrievalHelper()

    def on_validation_start(self) -> None:
        self.retrieval_metrics.on_validation_start(self.data.query_dataloader(), self.device, self.model)

    def on_validation_epoch_end(self) -> None:
        self.log("val_map", self.retrieval_metrics.on_validation_epoch_end())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


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
            query_embeddings.append(model.forward_features(images))
            query_labels.append(labels)

        self.query_embeddings = torch.concat(query_embeddings)
        self.query_labels = torch.concat(query_labels)
        self.val_sim = []
        self.val_labels = []

    def validation_add_features(self, features: torch.Tensor, labels: torch.Tensor):
        self.val_sim.append(cosine_sim(self.query_embeddings, features))
        self.val_labels.append(labels)

    def on_validation_epoch_end(self) -> float:
        self.val_sim = torch.concat(self.val_sim, dim=1)
        self.val_labels = torch.concat(self.val_labels)

        preds = torch.stack([label == self.val_labels for label in self.query_labels])
        map = retrieval_average_precision(self.val_sim, preds)

        del self.query_embeddings
        del self.query_labels
        self.query_embeddings = None
        self.query_labels = None

        return map
