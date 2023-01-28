from typing import Callable

import torchvision
from timm.data import RandomResizedCropAndInterpolation, rand_augment_transform

from .abstract_augmentation import AbstractAugmentation

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class BasicAugmentation(AbstractAugmentation):
    def get_transform(self) -> Callable:
        resize = RandomResizedCropAndInterpolation(size=self.image_shape)
        tfm = rand_augment_transform(
            config_str="rand-m9-mstd0.5",
            hparams={},
        )

        transform = torchvision.transforms.Lambda(lambda x: tfm(resize(x)))

        all_transform = [
            transform,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
        ]

        return torchvision.transforms.Compose(all_transform)

    def get_transform_val(self) -> Callable:

        all_transform = [
            torchvision.transforms.Resize(self.image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
        ]

        return torchvision.transforms.Compose(all_transform)
