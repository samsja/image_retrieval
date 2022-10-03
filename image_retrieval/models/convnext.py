from timm import create_model
from torch import nn
from torchtyping import TensorType


class ConvNext(nn.Module):
    def __init__(self, size="nano", *args, **kwargs):
        super().__init__()
        self.model = create_model(f"convnext_{size}", *args, **kwargs)

    def forward(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "L"]:
        return self.model(x)

    def forward_features(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "F"]:
        x = self.model.forward_features(x)
        x = self.model.head.global_pool(x)
        x = self.model.head.norm(x)
        return self.model.head.flatten(x)

    def forward_from_features(self, x: TensorType["batch", "F"]) -> TensorType["batch", "L"]:
        x = self.model.head.drop(x)
        return self.model.head.fc(x)
