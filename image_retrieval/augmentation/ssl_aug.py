# inspired by https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
import random
from typing import Callable

from PIL import ImageFilter
from timm.data import RandomResizedCropAndInterpolation
from torchvision import transforms

from image_retrieval.augmentation.abstract_augmentation import AbstractAugmentation


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


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

        return TwoCropsTransform(all_transform)

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
