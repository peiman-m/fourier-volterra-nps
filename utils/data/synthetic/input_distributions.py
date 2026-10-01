from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import einops
import torch
from omegaconf import OmegaConf


@dataclass
class BaseRandomParameterDistributionSampleConfig:
    """Configuration for sampling operations."""

    batch_shape: torch.Size
    dim: int
    n: int
    values_range: torch.Tensor


class BaseRandomParameterDistribution(ABC):
    """Abstract base class for different probability distributions."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample_params(
        self, batch_shape: torch.Size, dim: int
    ) -> dict[str, torch.Tensor]:
        """Sample distribution parameters."""
        pass

    @abstractmethod
    def sample_values(
        self, config: BaseRandomParameterDistributionSampleConfig, **params
    ) -> torch.Tensor:
        """Sample values from the distribution given parameters."""
        pass

    @staticmethod
    def _shift_scale(tensor: torch.Tensor, range_: torch.Tensor) -> torch.Tensor:
        """Shift and scale values from [0, 1] to target range."""
        return tensor * (range_[..., 1] - range_[..., 0]) + range_[..., 0]

    @staticmethod
    def _to_tensor(
        range_: (
            tuple[tuple[tuple[float, float], ...], ...]  # mode * dim * range
            | tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
        ),
    ) -> torch.Tensor:
        """Convert range tuple to tensor if needed."""
        return (
            torch.as_tensor(range_, dtype=torch.float)
            if not isinstance(range_, torch.Tensor)
            else range_
        )


class UniformSampler(BaseRandomParameterDistribution):
    """Uniform distribution sampler."""

    def sample_params(
        self, batch_shape: torch.Size, dim: int
    ) -> dict[str, torch.Tensor]:
        """No parameters needed for uniform distribution."""
        return {}

    def sample_values(
        self, config: BaseRandomParameterDistributionSampleConfig, **params
    ) -> torch.Tensor:
        """Sample from uniform distribution."""
        return self._shift_scale(
            torch.rand((*config.batch_shape, config.n, config.dim)), config.values_range
        )


class RandomOffsetSampler(BaseRandomParameterDistribution):
    """Sampler for distribution with random offsets."""

    def __init__(
        self,
        offset_range: (
            tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
        ),
    ) -> None:
        super().__init__()
        # Resolve nested ListConfig objects to native Python types
        if OmegaConf.is_config(offset_range):
            offset_range = cast(Any, OmegaConf.to_container(offset_range, resolve=True))
        self.offset_range = self._to_tensor(offset_range)

    def sample_params(
        self, batch_shape: torch.Size, dim: int
    ) -> dict[str, torch.Tensor]:
        """Sample offset parameter."""
        offset = self._shift_scale(torch.rand(*batch_shape, 1, dim), self.offset_range)
        return {"offset": offset}

    def sample_values(
        self,
        config: BaseRandomParameterDistributionSampleConfig,
        *,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Sample values with offset."""
        uniform_samples = super().sample_values(config)
        return uniform_samples + offset


class MixtureBetaSampler(BaseRandomParameterDistribution):
    """Mixture Beta distribution sampler."""

    def __init__(
        self,
        alpha_range: (
            tuple[tuple[tuple[float, float], ...], ...]  # mode * dim * range
            | tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
        ),
        beta_range: (
            tuple[tuple[tuple[float, float], ...], ...]  # mode * dim * range
            | tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
        ),
        num_modes: int = 1,
    ) -> None:
        super().__init__()
        # Resolve nested ListConfig objects to native Python types
        if OmegaConf.is_config(alpha_range):
            alpha_range = cast(Any, OmegaConf.to_container(alpha_range, resolve=True))
        if OmegaConf.is_config(beta_range):
            beta_range = cast(Any, OmegaConf.to_container(beta_range, resolve=True))
        self.alpha_range = self._to_tensor(alpha_range)
        self.beta_range = self._to_tensor(beta_range)
        self.num_modes = num_modes

    def sample_params(
        self, batch_shape: torch.Size, dim: int
    ) -> dict[str, torch.Tensor]:
        """Sample alpha and beta parameters."""

        def _sample_param(param_range: torch.Tensor) -> torch.Tensor:
            return self._shift_scale(
                torch.rand((*batch_shape, 1, self.num_modes, dim)), param_range
            )

        return {
            "alpha": _sample_param(self.alpha_range),
            "beta": _sample_param(self.beta_range),
        }

    def sample_values(
        self,
        config: BaseRandomParameterDistributionSampleConfig,
        *,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from beta distribution."""

        def _extend(tensor: torch.Tensor, n: int) -> torch.Tensor:
            return einops.repeat(tensor, "... 1 m d -> ... n m d", n=n)

        def _select_mode(tensor: torch.Tensor) -> torch.Tensor:
            *b, n, m, d = tensor.shape
            mode_indices = torch.randint(0, m, size=(*b, n, 1, d))
            result = torch.gather(tensor, dim=2, index=mode_indices)
            result = result.squeeze(2)
            return result

        beta_dist = torch.distributions.beta.Beta(
            _extend(alpha, config.n), _extend(beta, config.n)
        )
        samples = beta_dist.sample()
        samples = _select_mode(samples)
        return self._shift_scale(samples, config.values_range)
