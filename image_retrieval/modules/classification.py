import pytorch_lightning as pl
import torch
import torchmetrics
from torchtyping import TensorType

from image_retrieval.modules.helper import RetrievalHelper


class ClassificationModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, data: pl.LightningDataModule, lr=1e-3, debug=False):
        super().__init__()
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()
        self.model = model
        self.debug = debug
        self.data = data

        self.retrieval_metrics = RetrievalHelper()

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def on_validation_start(self) -> None:
        self.retrieval_metrics.on_validation_start(self.data.query_dataloader(), self.device, self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        output = self.model.forward_from_features(features)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        self.retrieval_metrics.validation_add_features(features, y)

    def on_validation_epoch_end(self) -> None:
        self.log("val_map", self.retrieval_metrics.on_validation_epoch_end())

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        acc = self.acc_fn(output, y)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
