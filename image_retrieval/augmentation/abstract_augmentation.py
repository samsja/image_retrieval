from abc import ABC, abstractmethod
from typing import Callable, Tuple


class AbstractAugmentation(ABC):
    def __init__(self, image_shape: Tuple[int, int]):
        self.image_shape = image_shape

    @abstractmethod
    def get_transform(self) -> Callable:
        ...

    @abstractmethod
    def get_transform_val(self) -> Callable:
        ...
