from timm import create_model
from torch import nn
from torchtyping import TensorType


class ConvNext(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = create_model("convnext_nano", *args, **kwargs)

    def forward(self, x: TensorType["batch":...]) -> TensorType["batch":...]:
        return self.model(x)
