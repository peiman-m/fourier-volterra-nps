import random
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import gpytorch
import torch

from ..base import GroundTruthPredictor
from .gp import GPGroundTruthPredictor
from .kernel_func import RandomHyperparameterKernel


class BaseSyntheticOutputGenerator(ABC):
    """Samples synthetic outputs ``y`` given inputs ``x``.

    ``sample`` returns ``(y, gt_pred)``. Generators without a closed-form
    ground truth (e.g. sawtooth, square wave) implement only the
    ``_sample_outputs`` hook and inherit the base ``sample``, which pairs the
    sampled ``y`` with ``gt_pred=None``. GP-based generators are the lone
    exception: they override ``sample`` to attach a
    :class:`GroundTruthPredictor`, which downstream is read by the
    ``gt_log_likelihood`` metric and the synthetic plot helper.
    """

    def __init__(self) -> None:
        super().__init__()

    def sample(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, GroundTruthPredictor | None]:
        """Sample outputs for the inputs ``x``.

        Args:
            x: Tensor of shape ``(batch_size, nc + nq, dim)`` holding the
                context and query inputs.

        Returns:
            Tuple ``(y, gt_pred)`` where ``y`` has shape
            ``(batch_size, nc + nq, 1)`` and ``gt_pred`` is ``None`` (no
            ground-truth predictor) unless the subclass overrides ``sample``.
        """
        return self._sample_outputs(x), None

    @abstractmethod
    def _sample_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return outputs ``y`` of shape ``(batch_size, nc + nq, 1)`` for ``x``."""


class SawtoothWaveGenerator(BaseSyntheticOutputGenerator):
    def __init__(
        self,
        min_freq: float = 1.0,
        max_freq: float = 1.0,
        min_amp: float = 1.0,
        max_amp: float = 1.0,
        noise_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.noise_std = noise_std

    def _sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Sample a frequency.
        batch_size, dim = x.shape[0], x.shape[-1]

        freq = self._sample_freq(batch_size)
        amp = self._sample_amp(batch_size)

        # Sample a direction.
        direction = (2 * torch.randint(0, 2, (batch_size, dim)) - 1).float()

        # Sample a uniformly distributed (conditional on frequency) offset.
        sample = torch.rand((batch_size,))
        offset = sample / freq

        # Construct the sawtooth and add noise.
        f = (
            freq[:, None, None] * (x @ direction[:, :, None] - offset[:, None, None])
        ) % 1

        # Scale by amplitude and center around 0
        f = amp[:, None, None] * (2 * f - 1)

        # Add noise
        y = f + self.noise_std * torch.randn_like(f)

        return y

    def _sample_freq(self, batch_size: int) -> torch.Tensor:
        # Sample frequency.
        freq = (
            torch.rand((batch_size,)) * (self.max_freq - self.min_freq) + self.min_freq
        )
        return freq

    def _sample_amp(self, batch_size: int) -> torch.Tensor:
        # Sample amplitude.
        amp = torch.rand((batch_size,)) * (self.max_amp - self.min_amp) + self.min_amp
        return amp


class SquareWaveGenerator(BaseSyntheticOutputGenerator):
    def __init__(
        self,
        min_freq: float = 1.0,
        max_freq: float = 1.0,
        min_amp: float = 1.0,
        max_amp: float = 1.0,
        min_duty_cycle: float = 0.25,
        max_duty_cycle: float = 0.75,
        noise_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.min_duty_cycle = min_duty_cycle
        self.max_duty_cycle = max_duty_cycle
        self.noise_std = noise_std

    def _sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Sample a frequency.
        batch_size, dim = x.shape[0], x.shape[-1]

        freq = self._sample_freq(batch_size)
        amp = self._sample_amp(batch_size)
        duty_cycle = self._sample_duty_cycle(batch_size, dim)

        # Sample a uniformly distributed (conditional on frequency) offset.
        sample = torch.rand((batch_size,))
        offset = sample / freq

        # Construct the wave and add noise.
        f = (freq[:, None, None] * (x - offset[:, None, None])) % 1

        # Convert to square wave by thresholding
        f = (f < duty_cycle[:, None, :]).float()

        # Scale by amplitude and center around 0
        f = amp[:, None, None] * (2 * f - 1)

        # Add noise
        y = f + self.noise_std * torch.randn_like(f)

        return y

    def _sample_freq(self, batch_size: int) -> torch.Tensor:
        # Sample frequency.
        freq = (
            torch.rand((batch_size,)) * (self.max_freq - self.min_freq) + self.min_freq
        )
        return freq

    def _sample_amp(self, batch_size: int) -> torch.Tensor:
        # Sample amplitude.
        amp = torch.rand((batch_size,)) * (self.max_amp - self.min_amp) + self.min_amp
        return amp

    def _sample_duty_cycle(self, batch_size: int, dim: int) -> torch.Tensor:
        # Sample duty cycle.
        duty_cycle = (
            torch.rand((batch_size, dim)) * (self.max_duty_cycle - self.min_duty_cycle)
            + self.min_duty_cycle
        )
        return duty_cycle


class GPSyntheticOutputGenerator(BaseSyntheticOutputGenerator):
    def __init__(
        self,
        kernel: (
            Callable[[], RandomHyperparameterKernel]
            | tuple[Callable[[], RandomHyperparameterKernel], ...]
        ),
        noise_std: float,
    ) -> None:
        super().__init__()

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = noise_std

    def _set_up_gp(self) -> GPGroundTruthPredictor:
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std**2.0

        return GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)

    def sample(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, GroundTruthPredictor]:
        # The only generator with a closed-form ground truth: build a fresh
        # predictor per call, sample ``y`` from it, and return both so the
        # predictor can be attached to the batch.
        gt_pred = self._set_up_gp()
        y = gt_pred.sample_outputs(x)
        return y, gt_pred

    def _sample_outputs(self, x: torch.Tensor) -> torch.Tensor:
        # GP overrides ``sample`` directly because its outputs are coupled to a
        # per-call predictor, so the y-only hook is never the path taken here.
        del x
        raise NotImplementedError(
            "GPSyntheticOutputGenerator produces outputs via `sample`, which "
            "also returns the ground-truth predictor."
        )
