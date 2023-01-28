
import pytorch_lightning as pl
import torch
from torchtyping import TensorType

from .criterion import SimSiamLoss
from ...base_module import BaseRetrievalModule


class SimSiamModule(BaseRetrievalModule):
    """
    SimSiam implementation (self supervised learning):
    https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: pl.LightningDataModule,
        lr=0.06,
        debug=False,
    ):
        super().__init__(model, data, lr, debug)
        self.loss = SimSiamLoss()

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x1, x2 = x[0], x[1]
        outs = self.model(im_aug1=x1, im_aug2=x2)
        loss = self.loss(outs['z1'], outs['z2'], outs['p1'], outs['p2'])
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.forward_features(x)
        self.retrieval_metrics.validation_add_features(features, y)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

