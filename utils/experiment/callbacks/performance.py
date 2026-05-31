"""LogPerformanceCallback — wall-clock + GPU-memory telemetry."""

from __future__ import annotations

import time
from typing import Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT


class LogPerformanceCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.between_step_time = 0.0

        # Forward pass timing for different phases
        self.train_forward_start_time = 0.0
        self.validation_forward_start_time = 0.0
        self.test_forward_start_time = 0.0

        # Accumulators for end-of-training/test mean/std summary
        # Note: val accumulators skip sanity-check batches (see on_validation_batch_end)
        self.all_train_forward_times: list[float] = []
        self.all_val_forward_times: list[float] = []
        self.all_test_forward_times: list[float] = []
        self.all_train_gpu_memory_GB: list[float] = []
        self.all_val_gpu_memory_GB: list[float] = []
        self.all_test_gpu_memory_GB: list[float] = []

        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(
                0
            ).total_memory  # Get total GPU memory

    @rank_zero_only
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "hardware/performance_between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
        )
        self.train_forward_start_time = time.time()

    @rank_zero_only
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.train_forward_start_time
        self.all_train_forward_times.append(forward_time)
        pl_module.log(
            "hardware/performance_train_forward_time",
            forward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # Log GPU memory after forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_train_gpu_memory_GB.append(gpu_mem_gb)
            pl_module.log(
                "hardware/gpu_memory_used_after_train_forward_GB",
                gpu_mem_gb,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            pl_module.log(
                "hardware/gpu_memory_total_after_forward_GB",
                self.total_memory / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

        self.backward_start_time = time.time()

    @rank_zero_only
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_after_backward(trainer, pl_module)
        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "hardware/performance_backward_time",
            backward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # Log GPU memory after gradient calculation
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()

            pl_module.log(
                "hardware/gpu_memory_used_after_gradients_GB",
                memory_allocated / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            pl_module.log(
                "hardware/gpu_memory_total_after_gradients_GB",
                self.total_memory / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

    @rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.update_count += 1

        # Calculate total elapsed time
        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        # Calculate updates per second
        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        # Log updates per second to wandb using pl_module.log
        pl_module.log(
            "hardware/performance_average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        pl_module.log(
            "hardware/performance_last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # Reset peak memory tracking for next step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.between_step_time = time.time()

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        t = torch.tensor(values)
        # correction=0: population std (correct for observed batches; also avoids
        # NaN when len==1 that Bessel's correction would produce)
        return t.mean().item(), t.std(correction=0).item()

    @staticmethod
    def _log_summary(
        summary: dict[str, float],
        trainer: pl.Trainer,
    ) -> None:
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            experiment = cast(Any, trainer.logger).experiment
            if hasattr(experiment, "summary"):
                for k, v in summary.items():
                    experiment.summary[k] = v

    @rank_zero_only
    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_train_end(trainer, pl_module)

        summary: dict[str, float] = {}

        if self.all_train_forward_times:
            mean, std = self._mean_std(self.all_train_forward_times)
            summary["hardware/train_forward_time_mean_s"] = mean
            summary["hardware/train_forward_time_std_s"] = std

        if self.all_val_forward_times:
            mean, std = self._mean_std(self.all_val_forward_times)
            summary["hardware/val_forward_time_mean_s"] = mean
            summary["hardware/val_forward_time_std_s"] = std

        if self.all_train_gpu_memory_GB:
            mean, std = self._mean_std(self.all_train_gpu_memory_GB)
            summary["hardware/train_gpu_memory_mean_GB"] = mean
            summary["hardware/train_gpu_memory_std_GB"] = std

        if self.all_val_gpu_memory_GB:
            mean, std = self._mean_std(self.all_val_gpu_memory_GB)
            summary["hardware/val_gpu_memory_mean_GB"] = mean
            summary["hardware/val_gpu_memory_std_GB"] = std

        if summary:
            self._log_summary(summary, trainer)

    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx)
        self.validation_forward_start_time = time.time()

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        forward_time = time.time() - self.validation_forward_start_time

        # Skip accumulation and logging during Lightning's sanity check
        if trainer.sanity_checking:
            return

        self.all_val_forward_times.append(forward_time)
        pl_module.log(
            "hardware/performance_validation_forward_time",
            forward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # Log GPU memory after validation forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_val_gpu_memory_GB.append(gpu_mem_gb)
            pl_module.log(
                "hardware/gpu_memory_used_after_val_forward_GB",
                gpu_mem_gb,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

    @rank_zero_only
    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx)
        self.test_forward_start_time = time.time()

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        forward_time = time.time() - self.test_forward_start_time
        self.all_test_forward_times.append(forward_time)

        # Only log to pl_module when a real logger is present.
        # With logger=False (eval.py), Lightning still accumulates pl_module.log
        # values into the trainer.test() return dict, which would pollute
        # results["metrics"] with hardware keys.
        if trainer.logger is not None:
            pl_module.log(
                "hardware/performance_test_forward_time",
                forward_time,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

        # Log GPU memory after test forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_test_gpu_memory_GB.append(gpu_mem_gb)
            if trainer.logger is not None:
                pl_module.log(
                    "hardware/gpu_memory_used_after_test_forward_GB",
                    gpu_mem_gb,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )

    def get_test_summary(self) -> dict[str, float]:
        """Return mean/std of test forward time and GPU memory over all batches."""
        summary: dict[str, float] = {}
        if self.all_test_forward_times:
            mean, std = self._mean_std(self.all_test_forward_times)
            summary["test_forward_time_mean_s"] = mean
            summary["test_forward_time_std_s"] = std
        if self.all_test_gpu_memory_GB:
            mean, std = self._mean_std(self.all_test_gpu_memory_GB)
            summary["test_gpu_memory_mean_GB"] = mean
            summary["test_gpu_memory_std_GB"] = std
        return summary

    @rank_zero_only
    def on_test_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_test_end(trainer, pl_module)

        summary: dict[str, float] = {}

        if self.all_test_forward_times:
            mean, std = self._mean_std(self.all_test_forward_times)
            summary["hardware/test_forward_time_mean_s"] = mean
            summary["hardware/test_forward_time_std_s"] = std

        if self.all_test_gpu_memory_GB:
            mean, std = self._mean_std(self.all_test_gpu_memory_GB)
            summary["hardware/test_gpu_memory_mean_GB"] = mean
            summary["hardware/test_gpu_memory_std_GB"] = std

        # Only print/log when running under a real logger (i.e. during training).
        # In eval.py the trainer uses logger=False; eval.py prints the summary itself.
        if summary and trainer.logger is not None:
            self._log_summary(summary, trainer)
