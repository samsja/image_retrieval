from typing import TypeVar

from timm import create_model
from torch import nn
from torchtyping import TensorType

from .abstract_model import AbstractModel

batch = TypeVar("batch")
H = TypeVar("H")
W = TypeVar("W")
L = TypeVar("L")
F = TypeVar("F")


class ConvNext(nn.Module, AbstractModel):
    def __init__(self, size="nano", pretrained=False, *args, **kwargs):
        super().__init__()
        self.model = create_model(
            f"convnext_{size}", num_classes=0, *args, pretrained=pretrained, **kwargs
        )

    def forward(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "F"]:
        return self.model(x)

    @property
    def output_size(self) -> int:
        return self.model.num_features
