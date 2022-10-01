import pytorch_lightning as pl
import torch
import torchmetrics
from torchtyping import TensorType


class ClassificationModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr=1e-3, debug=False):
        super().__init__()
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()
        self.model = model
        self.debug = debug

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        acc = self.acc_fn(output, y)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
