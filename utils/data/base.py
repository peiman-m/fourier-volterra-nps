import functools
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch


@dataclass
class BaseBatch(ABC):
    """Base class for all batch types.

    Declares the universal neural-process fields shared by every concrete
    batch (``x``/``y`` and the context/query split). Subclasses redeclare
    them (with shape-specific comments) and add their own fields; all
    batches are constructed by keyword, so the inherited field order is
    not observable.
    """

    x: torch.Tensor  # Input features
    y: torch.Tensor  # Output values

    xc: torch.Tensor  # Context input features
    yc: torch.Tensor  # Context output values

    xq: torch.Tensor  # Query input features
    yq: torch.Tensor  # Query output values

    def __hash__(self):
        return id(self)

    def to(self, device: torch.device) -> "BaseBatch":
        """Move all tensors in the batch to the specified device.

        Args:
            device: The device to move tensors to

        Returns:
            Self with tensors moved to device
        """
        # Move all tensors to the specified device
        for field in self.__dataclass_fields__:
            tensor = getattr(self, field)
            if isinstance(tensor, torch.Tensor):
                setattr(self, field, tensor.to(device))

        return self


EvalOn = Literal["query", "context"]


@dataclass
class Batch(BaseBatch):
    """Standard batch containing training and context data."""

    x: torch.Tensor  # Input features
    y: torch.Tensor  # output values

    xq: torch.Tensor  # Query input features (the points the model predicts at)
    yq: torch.Tensor  # Query output values

    xc: torch.Tensor  # Context input features
    yc: torch.Tensor  # Context output values


@functools.singledispatch
def as_batch(batch: BaseBatch, *, eval_on: EvalOn = "query") -> BaseBatch:
    """Return a batch with the query slot populated according to ``eval_on``.

    ``eval_on="query"`` is the identity (the held-out slot already lives in
    the query fields). ``eval_on="context"`` returns a copy with the context
    slot swapped into the query slot. Grid-bearing subclasses (``ImageBatch``,
    ``ERA5Batch``, ``KolmogorovBatch``) register their own handlers near their
    dataclass definitions to swap their ``xq``/``yq`` fields and ``mq_grid``.
    """
    raise NotImplementedError(
        f"as_batch is not registered for batch type {type(batch).__name__}"
    )


@as_batch.register(Batch)
def _as_batch_batch(batch: Batch, *, eval_on: EvalOn = "query") -> Batch:
    if eval_on == "query":
        return batch
    if eval_on == "context":
        return replace(batch, xq=batch.xc, yq=batch.yc)
    raise ValueError(f"Unsupported eval_on: {eval_on!r}")


class BaseIterableDataset(torch.utils.data.IterableDataset, ABC):
    """Base data generator for dynamically generated data (e.g. Synthetic, ERA5).

    Each call to generate_batch() produces fresh random data. In deterministic mode,
    batches are cached on first iteration for reproducible validation/test sets.
    """

    # Set by ``worker_init_fn`` (setup.py) in deterministic DDP mode; declared
    # here so the cache-slicing logic in ``__iter__`` typechecks.
    _ddp_global_worker_id: int | None
    _ddp_total_workers: int

    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        deterministic: bool = False,
        deterministic_seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        """Initialize the data generator.

        Args:
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
            deterministic: If True, generates deterministic batches.
            deterministic_seed: Seed to use for deterministic generation.
            drop_last: If True, drops the last batch if it's smaller than batch_size.

        Raises:
            ValueError: If samples_per_epoch or batch_size are not positive, or if
                batch_size exceeds samples_per_epoch.
        """
        super().__init__()

        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > samples_per_epoch:
            raise ValueError("batch_size cannot exceed samples_per_epoch")

        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed
        self.drop_last = drop_last

        # Calculate batch-related attributes
        self._calculate_batch_properties()
        self._global_num_batches = self.num_batches

        # Initialize state variables
        self._random_states: dict[str, Any] | None = None
        self._cached_batches: list[BaseBatch] | None = None
        self._batch_counter: int = 0

    def _get_ddp_info(self) -> tuple[int, int]:
        """Return (rank, world_size) from env vars set by torchrun/Lightning."""
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return rank, world_size

    def _calculate_batch_properties(self) -> None:
        """Calculate and set batch-related properties."""
        self.num_batches = self.samples_per_epoch // self.batch_size
        remainder = self.samples_per_epoch % self.batch_size

        if not self.drop_last and remainder:
            self.num_batches += 1
            self.final_batch_size = remainder
        else:
            self.final_batch_size = self.batch_size

    def _save_random_states(self) -> None:
        """Save current random states."""
        self._random_states = {
            "torch": torch.get_rng_state().clone(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def _restore_random_states(self) -> None:
        """Restore previously saved random states."""
        if self._random_states is None:
            return

        torch.set_rng_state(self._random_states["torch"])
        np.random.set_state(self._random_states["numpy"])
        random.setstate(self._random_states["random"])

    @contextmanager
    def _random_state_context(self) -> Iterator[None]:
        """Context manager for handling random states."""
        if self.deterministic:
            self._save_random_states()
            try:
                # Use seed_everything with workers=False to avoid affecting other processes
                pl.seed_everything(self.deterministic_seed, workers=False)
            except TypeError:
                # Fallback for older versions of PyTorch Lightning
                pl.seed_everything(self.deterministic_seed)
        try:
            yield
        finally:
            if self.deterministic:
                self._restore_random_states()

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        raise TypeError("length is undefined for IterableDataset")

    def __iter__(self) -> Iterator[BaseBatch]:
        """Initialize iteration state and return iterator."""
        if self.deterministic and self._cached_batches is None:
            with self._random_state_context():
                all_batches = self._generate_all_batches()
            global_worker_id = getattr(self, '_ddp_global_worker_id', None)
            if global_worker_id is not None:
                # worker_init_fn ran: split across all global workers (ranks × workers)
                total_workers = self._ddp_total_workers
                self._cached_batches = all_batches[global_worker_id::total_workers]
            else:
                # num_eval_workers=0: no worker_init_fn, split by rank only
                rank, world_size = self._get_ddp_info()
                self._cached_batches = all_batches[rank::world_size]
            # Update num_batches so __next__ stops at the correct index
            self.num_batches = len(self._cached_batches)
        self._batch_counter = 0
        return self

    def _generate_all_batches(self) -> list[BaseBatch]:
        """Generate all batches for deterministic mode.

        Returns:
            List of all batches for the epoch.
        """
        original_batch_counter = self._batch_counter
        self._batch_counter = 0
        batches = []
        for i in range(self._global_num_batches):
            is_final_batch = i == self._global_num_batches - 1
            original_batch_size = self.batch_size

            if is_final_batch and not self.drop_last:
                self.batch_size = self.final_batch_size

            try:
                batches.append(self.generate_batch())
            finally:
                if is_final_batch and not self.drop_last:
                    self.batch_size = original_batch_size

            self._batch_counter += 1
        self._batch_counter = original_batch_counter
        return batches

    def __next__(self) -> BaseBatch:
        """Generate next batch of data."""
        if self._batch_counter >= self.num_batches:
            raise StopIteration

        if self.deterministic and self._cached_batches is not None:
            batch = self._cached_batches[self._batch_counter]
        else:
            worker_info = torch.utils.data.get_worker_info()
            if not self.deterministic and worker_info is None:
                # No workers: split by rank directly
                rank, world_size = self._get_ddp_info()
                per_rank_batches = self._global_num_batches // world_size
                if self._batch_counter >= per_rank_batches:
                    raise StopIteration
                is_final_batch = self._batch_counter == per_rank_batches - 1
            else:
                is_final_batch = self._batch_counter == self.num_batches - 1

            original_batch_size = self.batch_size

            if is_final_batch and not self.drop_last:
                self.batch_size = self.final_batch_size

            try:
                batch = self.generate_batch()
            finally:
                if is_final_batch and not self.drop_last:
                    self.batch_size = original_batch_size

        self._batch_counter += 1
        return batch

    @abstractmethod
    def generate_batch(self) -> BaseBatch:
        """Generate a single batch of data.

        Returns:
            A batch object containing the generated data.

        This method must be implemented by derived classes.
        """
        pass


class BaseMapDataset(torch.utils.data.Dataset, ABC):
    """Base data loader for fixed/cached datasets (e.g. Image, Kolmogorov).

    Uses map-style indexing so that PyTorch DataLoader handles multi-worker
    and distributed sampling correctly. The collate_fn performs per-batch
    context/query splitting.
    """

    def __init__(
        self,
        *,
        samples_per_epoch: int,
    ) -> None:
        """Initialize the map-style data loader.

        Args:
            samples_per_epoch: Number of samples per epoch.
        """
        super().__init__()

        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")

        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    @abstractmethod
    def _sample_point_counts(self, n_max: int) -> tuple[int, int]:
        """Sample number of context and query points.

        Args:
            n_max: Maximum total number of points.

        Returns:
            Tuple of (context_points, query_points).
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict | torch.Tensor:
        """Return raw data for a single sample."""
        ...

    @abstractmethod
    def collate_fn(self, samples: list) -> BaseBatch:
        """Collate samples into a batch with context/query splitting."""
        ...


class GroundTruthPredictor(ABC):
    """Abstract base class for ground truth prediction models."""

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xq: torch.Tensor,
        yq: torch.Tensor | None = None,
    ) -> Any:
        """Predict outputs for query inputs given context.

        Args:
            xc: Context input features
            yc: Context output values
            xq: Query input features
            yq: Optional query output values

        Returns:
            Predictions for query inputs
        """
        raise NotImplementedError

    @abstractmethod
    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Sample outputs for given inputs.

        Args:
            x: Input features

        Returns:
            Sampled outputs
        """
        raise NotImplementedError
