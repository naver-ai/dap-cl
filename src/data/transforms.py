#!/usr/bin/env python3
"""
image transformations
"""

import torchvision as tv

def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    resize_dim = 256
    crop_dim = 224
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform
