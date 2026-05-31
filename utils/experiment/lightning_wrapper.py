from collections import defaultdict
from collections.abc import Callable
from dataclasses import fields
from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import BaseBatch
from ..data.base import EvalOn, as_batch

# Import forward wrappers
from .forward_wrappers import get_forward_wrapper
from .metrics.accumulators import BaseAccumulator
from .metrics.spec import MetricSpec


class PhaseConfig:
    """Per-phase config: a list of ``MetricSpec`` plus an optional ``loss_fn``.

    The Lightning loop reaches forward wrappers via the registry on every
    step; ``loss_fn`` is the sole backprop entry point on training phases.

    Args:
        metric_specs: Specs to evaluate every step (validation/test) or as
            train-time auxiliary telemetry. Empty list = no metrics.
        loss_fn: ``(model, batch) -> scalar Tensor`` callable used by
            ``training_step``. Required on the train phase; ignored on
            validation/test. ``None`` on validation/test by convention.
        name: Optional prefix for logged keys. With ``cfg.name = "query"``
            and ``spec.name = "loglik"``, the epoch metric logs as
            ``f"{phase}_query_loglik_epoch"``.
    """

    def __init__(
        self,
        metric_specs: list[MetricSpec] | None = None,
        loss_fn: Callable[[nn.Module, BaseBatch], Tensor] | None = None,
        name: str | None = None,
    ) -> None:
        self.metric_specs: list[MetricSpec] = list(metric_specs or [])
        self.loss_fn = loss_fn
        self.name = name


def _safe_key(name: str) -> str:
    """Munge a metric name into a valid ``nn.ModuleDict`` key (no dots, no slashes)."""
    return name.replace("/", "__").replace(".", "_")


class LitWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for Neural Process models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        train_config: PhaseConfig | list[PhaseConfig] | None = None,
        validation_config: PhaseConfig | list[PhaseConfig] | None = None,
        test_config: PhaseConfig | list[PhaseConfig] | None = None,
        save_hyperparams: bool = True,
        scheduler_config=None,
    ) -> None:
        """Initialize the Lightning wrapper.

        Args:
            model: Neural Process model to wrap.
            optimizer: Optimizer for model training.
            train_config: Configuration(s) for training phase. Can be a single PhaseConfig or list.
            validation_config: Configuration(s) for validation phase. Can be a single PhaseConfig or list.
            test_config: Configuration(s) for test phase. Can be a single PhaseConfig or list.
            save_hyperparams: Whether to save hyperparameters for tracking.
            scheduler_config: LR scheduler config dict. Must include 't_max' when enabled.
                Example: {enabled: true, t_max: 100, eta_min: 1e-6}
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.train_config = self._normalize_config(train_config)
        self.validation_config = self._normalize_config(validation_config)
        self.test_config = self._normalize_config(test_config)

        # Accumulators flattened into one ModuleDict so .to(device) reaches
        # their state buffers and torchmetrics handles DDP all-reduce. Flat
        # key = f"{phase}__{_safe_key(full_name)}" — nested ModuleDicts by
        # phase would collide with nn.Module attribute names like train().
        self._accumulators = nn.ModuleDict()
        self._name_to_key: dict[str, dict[str, str]] = {}
        for phase, configs in (
            ("train", self.train_config),
            ("validation", self.validation_config),
            ("test", self.test_config),
        ):
            key_map: dict[str, str] = {}
            for cfg in configs:
                for spec in cfg.metric_specs:
                    full_name = (
                        f"{cfg.name}_{spec.name}" if cfg.name else spec.name
                    )
                    flat_key = f"{phase}__{_safe_key(full_name)}"
                    if flat_key in self._accumulators:
                        raise ValueError(
                            f"Duplicate metric {full_name!r} in phase {phase!r} "
                            f"(flat key {flat_key!r} already registered)."
                        )
                    self._accumulators[flat_key] = spec.accumulator
                    key_map[full_name] = flat_key
            self._name_to_key[phase] = key_map

        # Save hyperparameters for tracking
        if save_hyperparams:
            self.save_hyperparameters(ignore=["model", "optimizer", "validation_config", "test_config", "scheduler_config"])

    def _normalize_config(
        self, config: PhaseConfig | list[PhaseConfig] | None
    ) -> list[PhaseConfig]:
        """Normalize config input to always be a list.

        Args:
            config: Configuration(s) for a phase.

        Returns:
            List of PhaseConfig objects.
        """
        if config is None:
            return []
        elif isinstance(config, PhaseConfig):
            return [config]
        else:
            return config

    def _run_step(
        self, phase: str, batch: BaseBatch, batch_idx: int
    ) -> Tensor | None:
        """Run forward + metrics + (training only) loss for one step.

        Groups specs by ``eval_on``, calls the forward wrapper once per
        group, runs each spec's ``metric_fn`` against the resulting
        likelihood, and updates per-spec ``torchmetrics`` accumulators.
        Returns the loss tensor for ``training_step`` to ``.backward()``,
        or ``None`` for validation/test.
        """
        configs = {
            "train": self.train_config,
            "validation": self.validation_config,
            "test": self.test_config,
        }[phase]

        loss_value: Tensor | None = None
        for cfg in configs:
            if phase == "train" and cfg.loss_fn is not None:
                loss_value = cfg.loss_fn(self.model, batch)
                # Train epoch-loss goes through Lightning's
                # mean-of-batch-means via sync_dist=True. Acceptable here;
                # use an explicit SampleMeanAccumulator if exact sample-mean
                # is needed.
                self.log(
                    "train_loss",
                    loss_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

            if not cfg.metric_specs:
                continue

            by_eval_on: dict[EvalOn, list[MetricSpec]] = defaultdict(list)
            for spec in cfg.metric_specs:
                by_eval_on[spec.eval_on].append(spec)

            for eval_on, group in by_eval_on.items():
                b = as_batch(batch, eval_on=eval_on)
                fw = get_forward_wrapper(self.model, b)
                if fw is None:
                    raise ValueError(
                        f"No forward wrapper registered for "
                        f"({type(self.model).__name__}, {type(b).__name__})."
                    )
                likelihood = fw(self.model, b)
                for spec in group:
                    raw = spec.metric_fn(likelihood, b)

                    full_name = (
                        f"{cfg.name}_{spec.name}" if cfg.name else spec.name
                    )
                    if spec.step_reducer is not None:
                        self.log(
                            f"{phase}_{full_name}",
                            spec.step_reducer(raw),
                            on_step=True,
                            on_epoch=False,
                            prog_bar=spec.prog_bar,
                            sync_dist=spec.sync_dist,
                        )

                    acc = cast(
                        BaseAccumulator,
                        self._accumulators[self._name_to_key[phase][full_name]],
                    )
                    acc.update(raw)
                    self.log(
                        f"{phase}_{full_name}_epoch",
                        acc,                # Lightning auto-calls .compute()/.reset()
                        on_step=False,
                        on_epoch=True,
                        # NO sync_dist — accumulator handles its own DDP reduction.
                    )

        return loss_value

    def training_step(self, batch: BaseBatch, batch_idx: int) -> Tensor | None:
        """Perform a training step. Returns the loss for backprop."""
        return self._run_step("train", batch, batch_idx)

    def validation_step(self, batch: BaseBatch, batch_idx: int) -> None:
        """Perform a validation step."""
        self._run_step("validation", batch, batch_idx)

    def test_step(self, batch: BaseBatch, batch_idx: int) -> None:
        """Perform a test step."""
        self._run_step("test", batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizers for training."""
        torch.autograd.set_detect_anomaly(True)
        if self.optimizer is None:
            raise ValueError(f"Optimizer is not initizlied for training!")
        if self.scheduler_config and self.scheduler_config.get("enabled"):
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_config.get("t_max"),
                eta_min=self.scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return self.optimizer


class TaggedModelCheckpoint(ModelCheckpoint):
    """
    A subclass of ModelCheckpoint that appends a suffix/tag to the state_key
    to avoid conflicts when multiple ModelCheckpoint instances are used.
    """

    def __init__(self, *args, tag: str = "", **kwargs):
        """
        Initialize TaggedModelCheckpoint with an optional tag.

        Args:
            tag: A suffix to append to the state_key to make it unique
            *args, **kwargs: Arguments passed to parent ModelCheckpoint
        """
        super().__init__(*args, **kwargs)
        self._tag = tag

    @property
    def state_key(self) -> str:
        """
        Generate a unique state_key by appending the tag to the parent's state_key.
        """
        parent_state_key = super().state_key
        if self._tag:
            return f"{parent_state_key}_{self._tag}"
        return parent_state_key


def _batch_to_cpu(batch: BaseBatch):
    batch_kwargs = {
        field.name: (
            getattr(batch, field.name).cpu()
            if isinstance(getattr(batch, field.name), torch.Tensor)
            else getattr(batch, field.name)
        )
        for field in fields(batch)
    }
    return type(batch)(**batch_kwargs)
