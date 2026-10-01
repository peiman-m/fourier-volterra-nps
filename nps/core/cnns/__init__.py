from .base import BaseCNN
from .cnn import ConvNet
from .fno import FNO
from .sfcnn import SetFourierConvNet
from .unet import UNet

__all__ = [
    "BaseCNN",
    "ConvNet",
    "FNO",
    "UNet",
    "SetFourierConvNet",
]
