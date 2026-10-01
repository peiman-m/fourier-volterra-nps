from collections.abc import Callable
from typing import Any

import torch
import torchvision
from torchvision import transforms

from .dataset import ImageDataset


class CIFARDataProcessor(torch.utils.data.Dataset):
    """CIFAR10 dataset wrapper that returns only images.

    Supports ``split="val"`` to carve a validation set out of the training
    data when no official validation split is available. The paired
    ``split="train"`` and ``split="val"`` instances must use the same
    ``val_fraction`` and ``random_state`` to get complementary slices.

    Args:
        dirpath: Root directory for the dataset.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
            ``"val"`` uses training data with a random subset determined by
            ``val_fraction`` and ``random_state``.
        transform: Optional image transform.
        download: Download the dataset if not already present.
        val_fraction: Fraction of training data reserved for validation.
            Only used when ``split`` is ``"train"`` or ``"val"``.
        random_state: Seed for the train/val permutation. Must match between
            the paired train and val processor instances.
    """

    IMAGE_SHAPE = (32, 32)

    def __init__(
        self,
        dirpath: str,
        *,
        split: str = "train",
        transform: Callable | None = None,
        download: bool = False,
        val_fraction: float = 0.1,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.subset = split
        self.dataset = torchvision.datasets.CIFAR10(
            root=dirpath,
            train=(split != "test"),
            transform=transform,
            download=download,
            **kwargs,
        )

        if split in ("train", "val"):
            n = len(self.dataset)
            perm = torch.randperm(
                n, generator=torch.Generator().manual_seed(random_state)
            ).tolist()
            cut = int(n * (1 - val_fraction))
            self._indices = perm[:cut] if split == "train" else perm[cut:]
        else:
            self._indices = None

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self._indices[idx] if self._indices is not None else idx
        image, _ = self.dataset[real_idx]
        return image


class CIFARDataset(ImageDataset):
    """DataLoader for CIFAR image datasets."""

    def __init__(self, *, processor: CIFARDataProcessor, **kwargs: Any) -> None:
        if not isinstance(processor, CIFARDataProcessor):
            raise TypeError(
                f"Expected CIFARDataProcessor, got {type(processor).__name__}"
            )

        super().__init__(
            processor=processor,
            image_shape=processor.IMAGE_SHAPE,
            **kwargs,
        )
