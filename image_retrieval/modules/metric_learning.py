import pytorch_lightning as pl
import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from torchtyping import TensorType

from image_retrieval.modules.helper import RetrievalHelper


class MetricLearningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__()
        self.lr = lr
        self.loss_fn = TripletMarginLoss()
        self.model = model
        self.debug = debug
        self.data = data

        self.retrieval_metrics = RetrievalHelper()

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model.forward_features(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)

        return loss

    def on_validation_start(self) -> None:
        self.retrieval_metrics.on_validation_start(self.data.query_dataloader(), self.device, self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        loss = self.loss_fn(features, y)

        self.log("val_loss", loss)
        self.retrieval_metrics.validation_add_features(features, y)

    def on_validation_epoch_end(self) -> None:
        self.log("val_map", self.retrieval_metrics.on_validation_epoch_end())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
