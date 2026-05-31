import os
import platform
import psutil
from collections.abc import Callable
from typing import Any, cast

import lightning.pytorch as pl
import torch
import torch.version
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from ..data import BaseIterableDataset, BaseMapDataset
from .callbacks import LogPerformanceCallback, PlotterCallback
from .lightning_wrapper import TaggedModelCheckpoint
from .utils import ensure_directory_exists


class _AttrDict(dict):
    """Thin dict subclass that exposes keys as attributes.

    ``instantiate(cfg, _convert_="all")`` returns a plain dict at the
    top level. This shim preserves ``experiment.misc.seed``-style
    attribute access at the aggregator level while keeping nested
    torch modules as-is (their constructors received plain Python
    kwargs, which is the point of the ``_convert_`` setting).

    Nested dicts are wrapped lazily on access, so
    ``experiment.misc.checkpointing.local`` works. Leaf non-dict
    values pass through unchanged.
    """

    def __getattr__(self, name) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, _AttrDict):
            value = _AttrDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value


def initialize_experiment(cfg: DictConfig) -> DictConfig:
    """Initialize an experiment from a pre-composed Hydra config."""
    pl.seed_everything(cfg.misc.seed)
    # ``_convert_="all"`` hands native Python dicts/lists to nested
    # ``_target_`` constructors so no ``ListConfig`` / ``DictConfig``
    # leaks into torch modules.
    experiment = instantiate(cfg, _convert_="all")
    # Re-expose the top-level container as an attribute-access shim
    # so existing ``experiment.misc.seed``-style call sites keep
    # working. Nested torch modules pass through unchanged.
    if isinstance(experiment, dict):
        experiment = _AttrDict(experiment)
    pl.seed_everything(experiment.misc.seed)

    # ``misc.metrics_dirpath`` is always present (interpolated from
    # ``misc.artifacts_dir`` in ``conf/misc/base.yaml``).
    ensure_directory_exists(experiment.misc.metrics_dirpath)

    # Plots dirpath only exists on experiments that wire in a plot
    # function. The guard is the presence of ``plot_fn``, not the
    # dirpath field itself (which is always interpolated).
    if getattr(experiment.misc, "plot_fn", None) is not None:
        ensure_directory_exists(experiment.misc.plots_dirpath)

    # The _AttrDict shim is consumed everywhere as a DictConfig-like object
    # (attribute access); downstream functions annotate it as DictConfig.
    return cast(DictConfig, experiment)


def setup_checkpointing(
    cfg: DictConfig, experiment: DictConfig
) -> list[Callback]:
    """Own all checkpoint wiring — directory creation and callback
    construction — in one place.

    Returns the list of checkpoint callbacks (may be empty if
    checkpointing is disabled).
    """
    callbacks: list[Callback] = []

    ckpt = getattr(experiment.misc, "checkpointing", {})
    if not ckpt.get("local", False):
        return callbacks

    checkpoint_dirpath = experiment.misc.checkpoint_dirpath
    ensure_directory_exists(checkpoint_dirpath)

    save_weights_only = ckpt.get("save_weights_only", True)
    version = ckpt.get("enable_version_counter", False)

    if ckpt.get("save_last", True):
        callbacks.append(
            TaggedModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="last",
                save_top_k=1,
                every_n_epochs=1,
                enable_version_counter=False,
                verbose=False,
                save_weights_only=save_weights_only,
                tag="last",
            )
        )

    best_config = ckpt.get("best", {})
    best_monitor = best_config.get("monitor") if best_config else None
    if best_monitor:
        callbacks.append(
            TaggedModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="best",
                monitor=best_monitor,
                mode=best_config.get("mode", "max"),
                save_top_k=best_config.get("save_top_k", 1),
                enable_version_counter=False,
                verbose=False,
                save_weights_only=save_weights_only,
                tag="best",
            )
        )

    every_n_epochs = ckpt.get("save_periodic_every_n_epochs", None)
    if every_n_epochs is not None and every_n_epochs > 0:
        callbacks.append(
            TaggedModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="{epoch}",
                every_n_epochs=every_n_epochs,
                enable_version_counter=version,
                save_top_k=-1,
                verbose=False,
                save_weights_only=save_weights_only,
                tag="periodic",
            )
        )

    return callbacks


def initialize_callbacks(
    cfg: DictConfig, experiment: DictConfig
) -> list[Callback]:
    """Return the Lightning callback list for an experiment.

    Delegates checkpoint-callback construction to ``setup_checkpointing``
    so the checkpointing config is read in exactly one place. Adds the
    optional hardware-performance callback on top, and wraps any
    configured ``misc.plot_fn`` (a ``BaseNeuralProcessPlotter``) in a
    ``PlotterCallback`` driven by ``misc.plot_interval`` and
    ``misc.num_plots``.
    """
    callbacks: list[Callback] = []

    callbacks.extend(setup_checkpointing(cfg, experiment))

    if getattr(experiment.misc, "hardware_callback", False):
        callbacks.append(LogPerformanceCallback())

    plot_fn = getattr(experiment.misc, "plot_fn", None)
    if plot_fn is not None:
        callbacks.append(
            PlotterCallback(
                plotter=plot_fn,
                every_n_val_epochs=getattr(experiment.misc, "plot_interval", 1),
                num_batches=getattr(experiment.misc, "num_plots", 5),
            )
        )

    return callbacks


def initialize_logger(cfg: DictConfig) -> Logger | bool:
    """Initialize the W&B logger (or fall back to Lightning's default).

    Takes ``cfg`` only — no instantiated ``experiment`` — so it can be
    called *before* ``initialize_experiment``. This lets stdout from
    dataset construction, model ``__init__``, and the rest of the
    instantiate walk be captured by wandb's console-log redirect.
    Everything it reads (``misc.wandb_logging_enabled``, ``misc.project``,
    ``misc.name``, ``misc.checkpointing.wandb``, ``misc.wandb_settings``,
    ``misc.default_lightning_logger``) already exists on the raw cfg
    before instantiate runs.
    """
    if not cfg.misc.wandb_logging_enabled:
        # Respect the user's static ``default_lightning_logger`` config
        # (``true`` = Lightning's default CSV logger, ``false`` = no
        # logger at all). Previously ``initialize_callbacks`` mutated
        # this field at runtime based on the callback list, but that
        # mutation is order-dependent and prevents moving the logger
        # init before the callback construction.
        return bool(cfg.misc.default_lightning_logger)

    checkpointing_config = cfg.misc.get("checkpointing", {})
    log_model = bool(checkpointing_config.get("wandb", False))

    wandb_options = {
        "project": cfg.misc.project,
        "name": cfg.misc.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "log_model": log_model,
    }

    if settings := cfg.misc.get("wandb_settings", None):
        wandb_options["settings"] = settings

    return WandbLogger(**wandb_options)


def create_dataloader(
    dataset: BaseIterableDataset | BaseMapDataset,
    num_workers: int,
    worker_init_fn: Callable | None = None,
    pin_memory: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """Creates a DataLoader with common configurations.

    Dispatches between map-style (fixed datasets) and iterable (dynamic generators).

    Args:
        dataset: The dataset/generator to create a DataLoader for.
        num_workers: Number of worker processes for data loading.
        worker_init_fn: Function to initialize each worker (iterable datasets only).
        pin_memory: Whether to pin memory for faster GPU transfer.
        batch_size: Batch size (map-style datasets only).
        shuffle: Whether to shuffle samples (map-style datasets only).
        drop_last: Whether to drop the last incomplete batch (map-style datasets only).
    """
    if isinstance(dataset, BaseMapDataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )
    else:
        # Existing IterableDataset path (Synthetic, ERA5)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )


def adjust_num_batches(worker_id: int | None = None) -> int:
    """
    Adjust the number of batches for a worker in a
    distributed data loading scenario.

    This function reduces the number of batches processed
    by each worker to ensure even distribution of work across
    all available workers.

    Args:
        worker_id (int, optional): ID of the worker.
            Primarily used for logging. If None, will
            be derived from worker info.

    Returns:
        int: The adjusted number of batches for this worker.

    Raises:
        RuntimeError: If called outside of a DataLoader worker context.
    """
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        raise RuntimeError(
            "This function must be called from within a DataLoader worker."
        )

    # Use worker_info.id if worker_id was not provided
    worker_id = worker_info.id if worker_id is None else worker_id

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    num_workers = worker_info.num_workers
    global_worker_id = rank * num_workers + worker_id

    # WorkerInfo.dataset is typed as the generic torch Dataset; the eval
    # dataloaders are always BaseIterableDataset here.
    worker_dataset = cast(BaseIterableDataset, worker_info.dataset)

    if worker_dataset.deterministic:
        # Store position for __iter__ to slice the full cache by global worker id.
        # Do NOT modify num_batches here — _generate_all_batches() needs the global count.
        worker_dataset._ddp_global_worker_id = global_worker_id
        worker_dataset._ddp_total_workers = world_size * num_workers
        total_workers = world_size * num_workers
        return worker_dataset.num_batches // total_workers

    num_batches = worker_dataset.num_batches  # global total
    total_processes = world_size * num_workers     # total workers across all GPUs

    base_batches = num_batches // total_processes
    remainder = num_batches % total_processes
    adjusted_num_batches = base_batches + (1 if global_worker_id < remainder else 0)

    print(
        f"Worker {global_worker_id} (rank={rank}, local={worker_id}): "
        f"num_batches={adjusted_num_batches} of {num_batches} global"
        + (" [+1 remainder]" if global_worker_id < remainder else "")
    )

    # Update the dataset's batch count so __next__ stops at the correct index
    worker_dataset.num_batches = adjusted_num_batches

    return adjusted_num_batches


def print_hardware_info() -> None:
    """Log available hardware information and the hardware being used for training."""
    print("\n" + "="*50)
    print("HARDWARE INFORMATION")
    print("="*50)

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

    memory = psutil.virtual_memory()
    print(f"Total RAM: {round(memory.total / (1024**3), 1)} GB")
    print(f"Available RAM: {round(memory.available / (1024**3), 1)} GB")

    if torch.cuda.is_available():
        print("CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {round(props.total_memory / (1024**3), 1)} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA Available: No")

    if torch.backends.mps.is_available():
        print("MPS Available: Yes (Apple Silicon)")
    else:
        print("MPS Available: No")

    print("="*50)


def print_training_config(
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader | None,
    epochs: int,
    train_batches_per_rank: int | None = None,
    val_batches_per_rank: int | None = None,
) -> None:
    """Log training configuration information."""
    if train_batches_per_rank is None:
        try:
            train_batches_per_rank = len(train_dataloader)
        except TypeError:
            train_batches_per_rank = cast(
                BaseIterableDataset, train_dataloader.dataset
            ).num_batches

    print("Training Configuration:")
    print(f"  Accelerator: {trainer.accelerator.__class__.__name__}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Max Epochs: {epochs}")
    print(f"  Train Batches per Epoch (per rank): {train_batches_per_rank}")
    if validation_dataloader is not None:
        if val_batches_per_rank is None:
            try:
                val_batches_per_rank = len(validation_dataloader)
            except TypeError:
                val_batches_per_rank = cast(
                    BaseIterableDataset, validation_dataloader.dataset
                ).num_batches
        print(f"  Validation Batches per Epoch (per rank): {val_batches_per_rank}")
    print("="*50 + "\n")
