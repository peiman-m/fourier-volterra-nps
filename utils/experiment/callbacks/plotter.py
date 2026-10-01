"""PlotterCallback — Lightning Callback that wraps a BaseNeuralProcessPlotter.

Owns the every-N-validation-epochs gate, validation-batch caching, eval-mode
toggling, and rank-0 gating. Plotter implementations subclassing
``BaseNeuralProcessPlotter`` handle forward, metric calls, and figure output
internally (via ``_get_forward_wrapper`` and ``_handle_figure_output``); this
callback just hands them a list of cached batches at the right cadence.
"""

from __future__ import annotations

from typing import Any, cast

import lightning.pytorch as pl
import torch

from ...data import BaseBatch
from ...plot_fn.base import BaseNeuralProcessPlotter


class PlotterCallback(pl.Callback):
    """Caches N validation batches and invokes a ``BaseNeuralProcessPlotter`` every K validation epochs.

    Args:
        plotter: A ``BaseNeuralProcessPlotter`` instance. Receives
            ``(model, batches, name)`` once per scheduled epoch.
        every_n_val_epochs: Plot every K validation epochs. ``<= 0`` disables.
        num_batches: Cache up to this many validation batches per epoch
            for the plotter to iterate over.
    """

    def __init__(
        self,
        plotter: BaseNeuralProcessPlotter,
        *,
        every_n_val_epochs: int = 1,
        num_batches: int = 5,
    ) -> None:
        super().__init__()
        self.plotter = plotter
        self.every_n_val_epochs = every_n_val_epochs
        self.num_batches = num_batches
        self._cached: list[BaseBatch] = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if len(self._cached) < self.num_batches:
            self._cached.append(batch)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            self._cached.clear()
            return
        if self.every_n_val_epochs <= 0 or not self._cached:
            self._cached.clear()
            return
        if trainer.current_epoch % self.every_n_val_epochs != 0:
            self._cached.clear()
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                self.plotter(
                    cast(Any, pl_module).model,
                    self._cached,
                    f"validation-epoch-{trainer.current_epoch:04d}",
                )
        finally:
            pl_module.train(was_training)
            self._cached.clear()
