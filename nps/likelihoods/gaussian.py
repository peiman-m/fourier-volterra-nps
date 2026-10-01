import torch
import torch.nn as nn
from torch.distributions import Normal

from .base import BaseLikelihood, TransformConfig


class HeteroscedasticNormalLikelihood(BaseLikelihood):
    """
    Gaussian likelihood with input-dependent (heteroscedastic) noise.

    Parameters
    ----------
    min_noise : float
        Minimum noise value to ensure numerical stability
    location_transform : TransformConfig, optional
        Transformation(s) to apply to mean parameter. Can be:
        - str: transformation name
            ('identity', 'sigmoid', 'softplus', 'clamp', 'add', 'multiply', etc.)
        - dict: {'transform_name': {'param1': value1, 'param2': value2}} for
            parameterized transforms
        - list/ListConfig: sequence of transformations applied in order, e.g.
            ['softplus', {'multiply': {'value': 2.0}}, {'clamp': {'min': 0.0, 'max': 1.0}}]
    scale_transform : TransformConfig, optional
        Transformation(s) to apply to standard deviation parameter. Can be:
        - str: transformation name ('softplus', 'exp', 'clamp', 'add', 'multiply', etc.)
        - dict: {'transform_name': {'param1': value1, 'param2': value2}} for
            parameterized transforms
        - list/ListConfig: sequence of transformations applied in order, e.g.
            ['softplus', {'multiply': {'value': 2.0}}, {'clamp': {'min': 0.0, 'max': 1.0}}]
    """

    def __init__(
        self,
        min_noise: float = 1e-6,
        location_transform: TransformConfig = "identity",
        scale_transform: TransformConfig = "softplus",
    ) -> None:
        super().__init__()
        self.min_noise = min_noise
        self.location_transform = location_transform
        self.scale_transform = scale_transform

    def forward(self, x: torch.Tensor) -> Normal:
        """
        Create a Normal distribution with input-dependent mean and standard deviation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing both mean and log variance concatenated along last dimension

        Returns
        -------
        Normal
            Normal distribution with input-dependent mean and standard deviation
        """
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"Last dimension must be even, got {x.shape[-1]}")

        loc, log_scale = torch.chunk(
            x, 2, dim=-1
        )  # Splitting tensor into two equal parts

        # Apply transformations
        loc = self._apply_transform(loc, self.location_transform)
        scale = (
            self._apply_transform(log_scale, self.scale_transform)
            + self.min_noise
        )

        return Normal(loc, scale)


class HomoscedasticNormalLikelihood(BaseLikelihood):
    """
    Gaussian likelihood with homoscedastic noise and learnable variance.

    Parameters
    ----------
    scale : float or torch.Tensor
        Initial scale value (standard deviation)
    min_noise : float
        Minimum noise value to ensure numerical stability
    location_transform : str, dict, or sequence, optional
        Transformation(s) to apply to mean parameter. Can be:
        - str: transformation name ('identity', 'sigmoid', 'softplus', 'clamp', 'add', 'multiply', etc.)
        - dict: {'transform_name': {'param1': value1, 'param2': value2}} for parameterized transforms
        - list/ListConfig: sequence of transformations applied in order, e.g.
            ['softplus', {'multiply': {'value': 2.0}}, {'clamp': {'min': 0.0, 'max': 1.0}}]
    learnable_scale : bool
        Whether to make the scale a learnable parameter
    """

    def __init__(
        self,
        scale: float | torch.Tensor = 1.0,
        min_noise: float = 1e-6,
        location_transform: TransformConfig = "identity",
        learnable_scale: bool = False,
    ) -> None:
        super().__init__()
        self.min_noise = min_noise
        self.location_transform = location_transform

        # Make log_scale a parameter if it's trainable
        if learnable_scale:
            self.log_scale = nn.Parameter(
                torch.tensor(scale, dtype=torch.float32)
                .clamp(min=min_noise)
                .log()
            )
        else:
            self.register_buffer(
                "log_scale",
                torch.tensor(scale, dtype=torch.float32)
                .clamp(min=min_noise)
                .log(),
            )

    def forward(self, x: torch.Tensor) -> Normal:
        """
        Create a Normal distribution with mean x and constant standard deviation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor representing the mean

        Returns
        -------
        Normal
            Normal distribution with mean x and constant standard deviation
        """
        # Apply transformation to mean
        loc = self._apply_transform(x, self.location_transform)

        # Broadcast log_scale to match x's shape
        log_scale = (
            self.log_scale.expand_as(x)
            if self.log_scale.dim() == 0
            else self.log_scale
        )
        scale = log_scale.exp() + self.min_noise

        return Normal(loc, scale)
