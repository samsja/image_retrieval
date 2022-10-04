import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from torchmetrics.functional import retrieval_average_precision
from torchtyping import TensorType

from image_retrieval.metrics import cosine_sim


class MetricLearningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__()
        self.lr = lr
        self.loss_fn = TripletMarginLoss()
        self.model = model
        self.debug = debug
        self.data = data

        self.query_embeddings = None
        self.query_labels = None

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model.forward_features(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)

        return loss

    def on_validation_start(self) -> None:
        query_embeddings = []
        query_labels = []
        for images, labels in self.data.query_dataloader():
            images = images.to(self.device)
            query_embeddings.append(self.model.forward_features(images))
            query_labels.append(labels)

        self.query_embeddings = torch.concat(query_embeddings)
        self.query_labels = torch.concat(query_labels)
        self.val_sim = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        output = self.model.forward_from_features(features)
        loss = self.loss_fn(output, y)

        self.log("val_loss", loss)

        self.val_sim.append(cosine_sim(self.query_embeddings, features))
        self.val_labels.append(y)

    def on_validation_epoch_end(self) -> None:
        self.val_sim = torch.concat(self.val_sim, dim=1)
        self.val_labels = torch.concat(self.val_labels)

        preds = torch.stack([label == self.val_labels for label in self.query_labels])
        map = retrieval_average_precision(self.val_sim, preds)
        self.log("val_map", map)

    def on_validation_end(self) -> None:
        del self.query_embeddings
        del self.query_labels
        self.query_embeddings = None
        self.query_labels = None

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
