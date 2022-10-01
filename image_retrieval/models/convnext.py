from timm import create_model
from torch import nn
from torchtyping import TensorType


class ConvNext(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = create_model("convnext_nano", *args, **kwargs)

    def forward(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "L"]:
        return self.model(x)

    def forward_features(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "F"]:
        x = self.model.forward_features(x)
        x = self.model.head.global_pool(x)
        x = self.model.head.norm(x)
        return self.model.head.flatten(x)
