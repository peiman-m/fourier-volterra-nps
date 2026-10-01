from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from .input_distributions import (
    BaseRandomParameterDistribution,
    BaseRandomParameterDistributionSampleConfig,
)


@dataclass
class SyntheticInputGeneratorSampleConfig:
    """Configuration for synthetic input generation."""

    batch_shape: torch.Size
    dim: int
    nc: int  # Number of context points
    nq: int | None = None  # Number of query points


class SyntheticInputGenerator:
    """Generator for synthetic input data."""

    def __init__(
        self,
        *,
        context_sampler: BaseRandomParameterDistribution,
        context_range: (
            tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
        ),
        query_sampler: BaseRandomParameterDistribution | None = None,
        query_range: (
            tuple[tuple[float, float], ...]  # dim * range
            | tuple[float, float]  # range
            | torch.Tensor
            | None
        ) = None,
        share_params: bool = False,
    ) -> None:
        """
        Initialize the synthetic input generator.

        Args:
            context_sampler: Distribution for sampling context points
            context_range: Range for context values
            query_sampler: Distribution for sampling query points
            query_range: Range for query values
            share_params: Whether to share parameters between context and query
        """
        self.context_sampler = context_sampler

        # Resolve nested ListConfig objects to native Python types
        if OmegaConf.is_config(context_range):
            context_range = cast(Any, OmegaConf.to_container(context_range, resolve=True))
        self.context_range = BaseRandomParameterDistribution._to_tensor(context_range)

        self.query_sampler = query_sampler or context_sampler

        # Resolve nested ListConfig objects for query_range
        if query_range is not None and OmegaConf.is_config(query_range):
            query_range = cast(Any, OmegaConf.to_container(query_range, resolve=True))
        self.query_range = (
            BaseRandomParameterDistribution._to_tensor(query_range)
            if query_range is not None
            else self.context_range
        )
        self.share_params = (
            share_params
            and (query_sampler is None)
            and type(query_sampler) == type(context_sampler)
        )

    def sample(self, config: SyntheticInputGeneratorSampleConfig) -> torch.Tensor:
        """
        Sample synthetic input data.

        Args:
            config: Sampling configuration

        Returns:
            Tensor of sampled points
        """
        # Sample context points
        context_params = self.context_sampler.sample_params(
            config.batch_shape, config.dim
        )
        xc = self.context_sampler.sample_values(
            BaseRandomParameterDistributionSampleConfig(
                batch_shape=config.batch_shape,
                dim=config.dim,
                n=config.nc,
                values_range=self.context_range,
            ),
            **context_params,
        )

        if config.nq is not None:
            # Sample query points
            query_params = (
                context_params
                if self.share_params
                else self.query_sampler.sample_params(config.batch_shape, config.dim)
            )

            xq = self.query_sampler.sample_values(
                BaseRandomParameterDistributionSampleConfig(
                    batch_shape=config.batch_shape,
                    dim=config.dim,
                    n=config.nq,
                    values_range=self.query_range,
                ),
                **query_params,
            )
            return torch.cat([xc, xq], dim=1)

        return xc
