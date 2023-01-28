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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(image_shape: Tuple[int, int], augmentation=True) -> Callable:
    """get basic transformation
    :param image_shape: shape of the image when resizing
    :param augmentation: include augmentation
    """
    if augmentation:
        return TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(image_shape[0], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])