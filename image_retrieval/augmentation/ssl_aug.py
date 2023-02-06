# inspired by https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
import random
from typing import Callable

from PIL import ImageFilter
from timm.data import RandomResizedCropAndInterpolation
from torchvision import transforms

from image_retrieval.augmentation.abstract_augmentation import AbstractAugmentation


class KCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, k=2):
        self.base_transform = base_transform
        self.k = k

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.k)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SSLAugmentation(AbstractAugmentation):
    def get_transform(self) -> Callable:
        normalize = transforms.Normalize(mean=MEAN, std=STD)

        augmentation_transform = [
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),  # not strengthened
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]

        all_transform = transforms.Compose(
            [
                RandomResizedCropAndInterpolation(size=self.image_shape),
                *augmentation_transform,
                transforms.ToTensor(),
                normalize,
            ]
        )

        return KCropsTransform(all_transform)

    def get_transform_val(self) -> Callable:
        normalize = transforms.Normalize(mean=MEAN, std=STD)

        all_transform = transforms.Compose(
            [
                transforms.Resize(self.image_shape),
                transforms.ToTensor(),
                normalize,
            ]
        )

        return all_transform


class SSLAugmentation2(AbstractAugmentation):
    def get_transform(self) -> Callable:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_shape[0], scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        return KCropsTransform(train_transforms)

    def get_transform_val(self) -> Callable:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )


class FastSiamSSL(AbstractAugmentation):
    """
    https://link.springer.com/chapter/10.1007/978-3-031-16788-1_4
    """

    def get_transform(self) -> Callable:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_shape[0], scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        return KCropsTransform(train_transforms, k=4)

    def get_transform_val(self) -> Callable:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
