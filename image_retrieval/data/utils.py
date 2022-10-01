from typing import Callable, Tuple

import torchvision
from timm.data import RandomResizedCropAndInterpolation, rand_augment_transform

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_transforms(image_shape: Tuple[int, int], augmentation=True) -> Callable:
    """get basic transformation
    :param image_shape: shape of the image when resizing
    :param augmentation: include augmentation
    """

    if augmentation:
        resize = RandomResizedCropAndInterpolation(size=image_shape)
        tfm = rand_augment_transform(
            config_str="rand-m9-mstd0.5",
            hparams={},
        )

        transform = torchvision.transforms.Lambda(lambda x: tfm(resize(x)))
    else:
        transform = torchvision.transforms.Resize(image_shape)

    all_transform = [
        transform,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]

    return torchvision.transforms.Compose(all_transform)
