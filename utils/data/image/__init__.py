# CIFAR datasets
from .cifar import CIFARDataProcessor, CIFARDataset

# DTD dataset
from .dtd import DTDDataProcessor, DTDDataset

# Image dataset and batch
from .dataset import ImageBatch, ImageDataset

# SVHN dataset
from .svhn import SVHNDataProcessor, SVHNDataset

__all__ = [
    # CIFAR
    "CIFARDataProcessor",
    "CIFARDataset",
    # DTD
    "DTDDataProcessor",
    "DTDDataset",
    # Base image components
    "ImageBatch",
    "ImageDataset",
    # SVHN
    "SVHNDataProcessor",
    "SVHNDataset",
]
