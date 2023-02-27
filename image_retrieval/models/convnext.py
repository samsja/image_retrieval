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
    def __init__(self, name="convnext_nano", pretrained=False, *args, **kwargs):
        super().__init__()
        self.model = create_model(
            name, num_classes=0, *args, pretrained=pretrained, **kwargs
        )

    def forward(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "F"]:
        return self.model(x)

    @property
    def output_size(self) -> int:
        return self.model.num_features


class ConvNextNano(ConvNext):
    def __init__(self, pretrained=False, *args, **kwargs):
        super().__init__("convnext_nano", pretrained, *args, **kwargs)


class ConvNextBase(ConvNext):
    def __init__(self, pretrained=False, *args, **kwargs):
        super().__init__("convnext_base", pretrained, *args, **kwargs)
