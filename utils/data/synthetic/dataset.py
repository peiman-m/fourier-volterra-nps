import random
from abc import ABC
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from ..base import BaseIterableDataset, Batch, GroundTruthPredictor
from .synthetic_input import (
    SyntheticInputGenerator,
    SyntheticInputGeneratorSampleConfig,
)
from .synthetic_output import BaseSyntheticOutputGenerator


@dataclass
class SyntheticBatch(Batch):
    gt_mean: torch.Tensor | None = None
    gt_std: torch.Tensor | None = None
    gt_loglik: torch.Tensor | None = None
    gt_pred: GroundTruthPredictor | None = None


class SyntheticDataset(BaseIterableDataset, ABC):
    """Base class for generating synthetic datasets."""

    def __init__(
        self,
        *,
        dim: int,
        min_nc: int,
        max_nc: int,
        min_nq: int,
        max_nq: int,
        input_generator: SyntheticInputGenerator,
        output_generator: BaseSyntheticOutputGenerator,
        **kwargs,
    ) -> None:
        """Initialize synthetic data generator.

        Args:
            dim: Dimensionality of input features
            min_nc: Minimum number of context points
            max_nc: Maximum number of context points
            min_nq: Minimum number of query points
            max_nq: Maximum number of query points
            input_generator: Generator for input data
            output_generator: Generator for output data given inputs
            **kwargs: Additional arguments passed to BaseIterableDataset
        """
        super().__init__(**kwargs)

        # Set synthetic generator parameters
        self.dim = dim
        self.min_nc = min_nc
        self.max_nc = max_nc
        self.min_nq = min_nq
        self.max_nq = max_nq
        self.input_generator = input_generator
        self.output_generator = output_generator

        print(
            f'[{type(self).__name__}] [{type(output_generator).__name__}] '
            f'nc=[{min_nc}, {max_nc}], nq=[{min_nq}, {max_nq}], dim={dim}'
        )

    def _sample_point_counts(self) -> tuple[int, int]:
        """Sample number of context and query points uniformly.

        Returns:
            Tuple of (num_context, num_query)
        """
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nq = torch.randint(low=self.min_nq, high=self.max_nq + 1, size=())
        return int(nc.item()), int(nq.item())

    @staticmethod
    def _split_samples(
        points: torch.Tensor, nc: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split points into context and query sets.

        Args:
            points: Tensor of shape (batch_size, num_points, dim)
            nc: Number of context points

        Returns:
            Tuple of (context_points, query_points)
        """
        if points.size(1) < nc:
            raise ValueError(f"Not enough points to split: {points.size(1)} < {nc}")

        return points[:, :nc, :], points[:, nc:, :]

    def _sample_batch(
        self,
        nc: int,
        nq: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        """Sample a complete batch of data.

        Args:
            nc: Number of context points
            nq: Number of query points
            batch_shape: Shape of the batch

        Returns:
            SyntheticBatch object with sampled data
        """
        x = self.input_generator.sample(
            SyntheticInputGeneratorSampleConfig(
                nc=nc, nq=nq, dim=self.dim, batch_shape=batch_shape
            )
        )
        y, gt_pred = self.output_generator.sample(x=x)

        # Split into context and query (held-out)
        xc, xq = self._split_samples(x, nc)
        yc, yq = self._split_samples(y, nc)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xq=xq,
            yq=yq,
            gt_pred=gt_pred,
        )

    def generate_batch(self) -> Batch:
        """Generate batch of synthetic data.

        Returns:
            SyntheticBatch containing context and query data with optional ground truth.
        """
        # Sample number of context and query points.
        nc, nq = self._sample_point_counts()

        # Sample entire batch (context and query points).
        batch = self._sample_batch(
            nc=nc,
            nq=nq,
            batch_shape=torch.Size((self.batch_size,)),
        )

        return batch


class MixtureSyntheticDataset(SyntheticDataset):
    def __init__(
        self,
        *,
        generators: tuple[SyntheticDataset, ...],
        mixture_probs: tuple[float, ...],
        mix_samples: bool = False,
        **kwargs,
    ) -> None:
        # Hydra's ``_convert_="all"`` hands YAML sequences in as ``list``;
        # promote to ``tuple`` for the stored attributes.
        if isinstance(generators, list):
            generators = tuple(generators)
        if isinstance(mixture_probs, list):
            mixture_probs = tuple(mixture_probs)

        assert len(generators) == len(
            mixture_probs
        ), "Must be a mixture prob for each generator."
        assert sum(mixture_probs) == 1, "Sum of mixture_probs must be 1."
        assert all(
            prob > 0 for prob in mixture_probs
        ), "All elements of mixture_probs must be positive."

        super().__init__(**kwargs)

        # Whether or not to sample mixture for each sample in batch.
        self.mix_samples = mix_samples
        self.generators = generators
        self.mixture_probs = mixture_probs

        # Ensure samples per epoch of generators are infinite, so does not stop sampling.
        # num_batches is declared int; np.inf is a deliberate "never stop" sentinel.
        for generator in self.generators:
            generator.num_batches = cast(int, np.inf)

    def generate_batch(self) -> Batch:
        # Sample generator.
        gen = random.choices(self.generators, weights=self.mixture_probs, k=1)[0]

        # Sample number of context and query points.
        nc, nq = self._sample_point_counts()

        return gen._sample_batch(
            nc=nc, nq=nq, batch_shape=torch.Size((self.batch_size,))
        )
