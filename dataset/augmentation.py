import random
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor


class RandomHorizontalFlip:
    """Randomly flip an image horizontally with a given probability.

    This class is designed to work with the GSN dataset, where images are
    28x28 grayscale tensors of shape (1, H, W). Horizontal flip does not
    change the shape *counts*, so targets are passed through unchanged.

    Usage
    -----
    >>> aug = RandomHorizontalFlip(p=0.5)
    >>> img, counts = aug(img, counts)
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if random.random() < self.p:
            image = torch.flip(image, dims=[-1])
            if counts is not None:
                # counts: [sq, circ, up, right, down, left]
                sq, ci, up, right, down, left = counts.tolist()
                right, left = left, right
                counts = torch.tensor(
                    [sq, ci, up, right, down, left],
                    dtype=counts.dtype,
                    device=counts.device,
                )
        return image, counts


class RandomVerticalFlip:
    """Randomly flip an image vertically and update orientation labels.

    Vertical flip swaps ``up`` and ``down`` triangle counts, other counts stay
    unchanged.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if random.random() < self.p:
            # Flip along height dimension (second-to-last dim for CHW)
            image = torch.flip(image, dims=[-2])
            if counts is not None:
                # counts: [sq, circ, up, right, down, left]
                sq, ci, up, right, down, left = counts.tolist()
                up, down = down, up
                counts = torch.tensor(
                    [sq, ci, up, right, down, left],
                    dtype=counts.dtype,
                    device=counts.device,
                )
        return image, counts


class RandomRotate90:
    """Randomly rotate the image by k*90 degrees and rotate orientation labels.

    Rotation is clockwise, k in {1, 2, 3}. Squares and circles are unchanged.
    Oriented triangles are cyclically permuted:

    * 90° cw:    up->right->down->left->up
    * 180° cw:   up<->down, right<->left
    * 270° cw:   up->left->down->right->up
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if random.random() >= self.p:
            return image, counts

        k = random.randint(1, 3)
        # Rotate in H,W plane (CHW -> rotate last two dims)
        image = torch.rot90(image, k=-k, dims=[-2, -1])

        if counts is not None:
            sq, ci, up, right, down, left = counts.tolist()
            if k == 1:  # 90° cw
                up, right, down, left = left, up, right, down
            elif k == 2:  # 180°
                up, down = down, up
                right, left = left, right
            elif k == 3:  # 270° cw
                up, right, down, left = right, down, left, up
            counts = torch.tensor(
                [sq, ci, up, right, down, left],
                dtype=counts.dtype,
                device=counts.device,
            )
        return image, counts


class RandomBrightnessContrast:
    """Perform mild brightness/contrast adjustment while keeping labels intact."""

    def __init__(self, p: float = 0.5, brightness_delta: float = 0.1, contrast_delta: float = 0.1):
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        assert brightness_delta >= 0.0 and contrast_delta >= 0.0
        self.p = p
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if random.random() >= self.p:
            return image, counts

        brightness = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
        contrast = 1.0 + random.uniform(-self.contrast_delta, self.contrast_delta)
        # Apply brightness
        image = torch.clamp(image * brightness, 0.0, 1.0)
        # Apply contrast around mean per channel
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = torch.clamp((image - mean) * contrast + mean, 0.0, 1.0)
        return image, counts


class RandomGaussianNoise:
    """Add mild Gaussian noise; labels stay unchanged."""

    def __init__(self, p: float = 0.5, std: float = 0.03):
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        assert std >= 0.0
        self.p = p
        self.std = std

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if random.random() >= self.p:
            return image, counts

        noise = torch.randn_like(image) * self.std
        image = torch.clamp(image + noise, 0.0, 1.0)
        return image, counts


class RandomAugmentation:
    """Apply multiple augmentations with independent probabilities."""

    def __init__(
        self,
        *,
        p_hflip: float = 0.0,
        p_vflip: float = 0.0,
        p_rotate90: float = 0.0,
        p_brightness: float = 0.0,
        p_noise: float = 0.0,
        brightness_delta: float = 0.05,
        contrast_delta: float = 0.05,
        noise_std: float = 0.02,
        augmentations: Optional[Dict[str, Callable[..., Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]]]] = None,
    ):
        available = augmentations or {
            "hflip": RandomHorizontalFlip,
            "vflip": RandomVerticalFlip,
            "rotate90": RandomRotate90,
            "brightness": RandomBrightnessContrast,
            "noise": RandomGaussianNoise,
        }
        self.transforms: list[Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]] = []

        if p_hflip > 0:
            cls = available.get("hflip", RandomHorizontalFlip)
            self.transforms.append(cls(p=p_hflip))
        if p_vflip > 0:
            cls = available.get("vflip", RandomVerticalFlip)
            self.transforms.append(cls(p=p_vflip))
        if p_rotate90 > 0:
            cls = available.get("rotate90", RandomRotate90)
            self.transforms.append(cls(p=p_rotate90))
        if p_brightness > 0:
            cls = available.get("brightness", RandomBrightnessContrast)
            self.transforms.append(
                cls(p=p_brightness, brightness_delta=brightness_delta, contrast_delta=contrast_delta)
            )
        if p_noise > 0:
            cls = available.get("noise", RandomGaussianNoise)
            self.transforms.append(cls(p=p_noise, std=noise_std))

    def __call__(self, image: Tensor, counts: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        order = list(self.transforms)
        random.shuffle(order)
        for t in order:
            image, counts = t(image, counts)
        return image, counts


def apply_augmentations(
    image: Tensor,
    counts: Optional[Tensor],
    transform: Optional[Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    if transform is None:
        return image, counts
    return transform(image, counts)
