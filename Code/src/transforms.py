"""Image preprocessing. Counterfactuals are applied in pixel space, before normalization."""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(image, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(image)  # [0, 1]
    return tensor


def maybe_hflip(image: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and torch.rand(1).item() < 0.5:
        return torch.flip(image, dims=[2])
    return image


def photometric_style_view(image: torch.Tensor) -> torch.Tensor:
    """DG-PLDR augmentation A(x): mild color jitter plus a small blur.

    This is the published style perturbation used to build relation pairs.
    It is separate from the disease-preserving counterfactual generator.
    """
    image = TF.adjust_brightness(image, 1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    image = TF.adjust_contrast(image, 1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    image = TF.adjust_saturation(image, 1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    if torch.rand(1).item() < 0.5:
        image = TF.gaussian_blur(image, kernel_size=5, sigma=0.8)
    return image.clamp(0, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    if images.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif images.dim() == 5:
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    return (images - mean) / std
