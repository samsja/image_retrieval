# inspired by https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
import random
from typing import Callable, Tuple

from PIL import ImageFilter
from timm.data import RandomResizedCropAndInterpolation
from torchvision import transforms


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


def get_transforms(image_shape: Tuple[int, int], augmentation=True) -> Callable:
    """get basic transformation
    :param image_shape: shape of the image when resizing
    :param augmentation: include augmentation
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resize = (
        RandomResizedCropAndInterpolation(size=image_shape)
        if augmentation
        else transforms.Resize(image_shape)
    )

    augmentation_transform = (
        [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        if augmentation
        else []
    )

    all_transform = transforms.Compose(
        [
            resize,
            *augmentation_transform,
            transforms.ToTensor(),
            normalize,
        ]
    )

    return TwoCropsTransform(all_transform) if augmentation else all_transform
