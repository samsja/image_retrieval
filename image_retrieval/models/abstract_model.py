from abc import ABC, abstractmethod
from typing import TypeVar

from torchtyping import TensorType

batch = TypeVar("batch")
H = TypeVar("H")
W = TypeVar("W")
E = TypeVar("E")


class AbstractModel(ABC):
    @abstractmethod
    def forward(self, x: TensorType["batch", 3, "H", "W"]) -> TensorType["batch", "E"]:
        ...

    @property
    @abstractmethod
    def output_size(self) -> int:
        ...
