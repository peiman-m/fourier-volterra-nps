from collections.abc import Callable
from typing import Any

import torch
import torchvision
from torchvision import transforms

from .dataset import ImageDataset


class DTDDataProcessor(torch.utils.data.Dataset):
    """Describable Textures Dataset (DTD) processor.

    This wraps the torchvision DTD dataset and can be used standalone for
    quick exploration. For training with neural processes, use DTDDataset
    which provides context/query splitting and efficient batching.

    Usage:
        # For exploration:
        processor = DTDDataProcessor(dirpath='./data', split='train', download=True)
        image = processor[0]  # Get a single image

        # For training (recommended):
        processor = DTDDataProcessor(dirpath='./data', split='train')
        dataset = DTDDataset(
            processor=processor,
            min_nc=200, max_nc=800,
            min_nq=200, max_nq=800
        )
        loader = DataLoader(dataset, batch_size=16, num_workers=4)

    Note:
        DTD images vary in size (231-778 height, 271-900 width). This processor
        applies random cropping and resizing to normalize them to 48x48 pixels.
        The processor wraps torchvision datasets which load images on-demand
        from disk, so they don't cause memory issues with DataLoader workers.
    """

    #     Height range: 231-778
    #     Width range: 271-900
    #     Channels: {3}
    IMAGE_SHAPE = (48, 48)
    CROP_SIZE = tuple(3 * s for s in IMAGE_SHAPE)

    def __init__(
        self,
        dirpath: str,
        *,
        split: str = "train",
        transform: Callable | None = None,
        download: bool = False,
        **kwargs,
    ) -> None:

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(size=self.CROP_SIZE),
                    transforms.Resize(size=self.IMAGE_SHAPE),
                ]
            )

        self.subset = split
        self.dataset = torchvision.datasets.DTD(
            root=dirpath,
            split=split,
            transform=transform,
            download=download,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.dataset[idx]  # Discard label
        return image


class DTDDataset(ImageDataset):
    """DataLoader for DTD image datasets."""

    def __init__(self, *, processor: DTDDataProcessor, **kwargs: Any) -> None:
        if not isinstance(processor, DTDDataProcessor):
            raise TypeError(f"Expected DTD processor, got {type(processor).__name__}")

        super().__init__(
            processor=processor,
            image_shape=processor.IMAGE_SHAPE,
            **kwargs,
        )
