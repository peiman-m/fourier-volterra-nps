import os
from pathlib import Path
import sys
from typing import Any, cast

# Ensure the repo root is on sys.path so ``from conf import _resolvers``
# resolves whether the script is run via ``python train.py`` or imported
# as a module.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
import torch
import wandb

from conf import _resolvers  # noqa: F401  (registers eval/product/range)
from utils.experiment import (
    LitWrapper,
    adjust_num_batches,
    create_dataloader,
    initialize_callbacks,
    initialize_experiment,
    initialize_logger,
    print_hardware_info,
    print_training_config,
)


def run_training(cfg: DictConfig) -> None:
    """Full training body. Importable so tests can call it directly
    with a pre-composed ``DictConfig`` without going through
    ``@hydra.main``."""
    # Initialize the logger BEFORE instantiate so stdout from dataset
    # construction, model ``__init__``, and the rest of the Hydra
    # instantiate walk is captured by wandb's console-log redirect.
    # ``initialize_logger(cfg)`` reads only scalar ``cfg.misc.*`` fields
    # that exist on the raw cfg, so it doesn't need ``experiment``.
    logger = initialize_logger(cfg)
    if hasattr(logger, "experiment"):
        _ = cast(Any, logger).experiment  # fires ``wandb.init()``

    experiment = initialize_experiment(cfg)

    # Define once — used by both the rank re-seed and the checkpoint sync.
    _rank = int(os.environ.get("RANK", 0))
    _world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Re-seed with rank offset so training workers on each rank generate
    # different data. DDP broadcasts rank-0 model weights on trainer.fit(),
    # so model init is unaffected.
    if _rank > 0:
        pl.seed_everything(experiment.misc.seed + _rank)

    model = experiment.model
    train_dataset = experiment.data.datasets.train
    validation_dataset = getattr(experiment.data.datasets, "validation", None)
    optimizer = experiment.optimizer(model.parameters())
    epochs = experiment.misc.epochs

    # Create dataloaders
    worker_init_fn = getattr(experiment.misc, "worker_init_fn", adjust_num_batches)
    dl_config = getattr(experiment.data, "dataloaders", {})
    train_dl_config = dl_config.get("train", {})
    val_dl_config = dl_config.get("validation", {})

    train_dataloader = create_dataloader(
        dataset=train_dataset,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=getattr(experiment.misc, "pin_memory", True),
        batch_size=train_dl_config.get("batch_size", 1),
        shuffle=train_dl_config.get("shuffle", True),
        drop_last=train_dl_config.get("drop_last", True),
    )
    validation_dataloader = (
        create_dataloader(
            dataset=validation_dataset,
            num_workers=experiment.misc.num_eval_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=getattr(experiment.misc, "pin_memory", True),
            batch_size=val_dl_config.get("batch_size", 1),
            shuffle=val_dl_config.get("shuffle", False),
            drop_last=val_dl_config.get("drop_last", True),
        )
        if validation_dataset is not None
        else None
    )

    checkpointing_config = getattr(experiment.misc, "checkpointing", {})

    # Model Initialization
    ckpt_file = None
    _weights_loaded = False

    # ``resume_from`` is a single nullable string — None = fresh training,
    # a string = resume from that path (``wandb:...`` artifact or local
    # filesystem path).
    resume_path = checkpointing_config.get("resume_from", None)

    if resume_path:

        # Determine if checkpoint is from wandb or local
        if resume_path.startswith(("wandb:", "wandb/")):
            import tempfile
            import time as _time

            # Use SLURM_JOB_ID when available so concurrent jobs don't collide.
            _job_discriminator = os.environ.get("SLURM_JOB_ID", "ddp_sync")
            _sync_file = os.path.join(tempfile.gettempdir(), f"ckpt_path_{_job_discriminator}.txt")

            if _rank == 0:
                # Remove any stale sync file from a previous run before writing a new one.
                try:
                    os.remove(_sync_file)
                except OSError:
                    pass
                try:
                    api = wandb.Api()
                    artifact = api.artifact(resume_path)
                    artifact_dir = artifact.download()
                    ckpt_file = os.path.join(artifact_dir, "model.ckpt")
                except Exception as e:
                    print(f"Warning: Failed to load wandb checkpoint: {e}")
                    ckpt_file = None
                # Always write the sync file — empty string signals failure.
                if _world_size > 1:
                    with open(_sync_file, "w") as _f:
                        _f.write(ckpt_file or "")
            else:
                ckpt_file = None
                if _world_size > 1:
                    _time.sleep(2)
                    for _ in range(120):
                        if os.path.exists(_sync_file):
                            break
                        _time.sleep(1)
                    try:
                        with open(_sync_file) as _f:
                            ckpt_file = _f.read().strip() or None
                    except OSError:
                        ckpt_file = None

            if ckpt_file:
                try:
                    lit_model = LitWrapper.load_from_checkpoint(
                        ckpt_file,
                        model=model,
                        optimizer=optimizer,
                        train_config=experiment.phase_configs.train,
                        validation_config=experiment.phase_configs.validation,
                        scheduler_config=getattr(experiment.misc, "lr_scheduler", None),
                    )
                    print(f"Resuming from wandb checkpoint: {resume_path}")
                    _weights_loaded = True
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint: {e}")
                    ckpt_file = None
        else:
            # Try to load from local path
            try:
                if os.path.exists(resume_path):
                    ckpt_file = resume_path
                    lit_model = LitWrapper.load_from_checkpoint(
                        ckpt_file,
                        model=model,
                        optimizer=optimizer,
                        train_config=experiment.phase_configs.train,
                        validation_config=experiment.phase_configs.validation,
                        scheduler_config=getattr(experiment.misc, "lr_scheduler", None),
                    )
                    print(f"Resuming from local checkpoint: {resume_path}")
                    _weights_loaded = True
                else:
                    print(f"Warning: Local checkpoint not found at {resume_path}")
            except Exception as e:
                print(f"Warning: Failed to load local checkpoint: {e}")
                ckpt_file = None

    # If the checkpoint was saved with save_weights_only=True it has no
    # optimizer_states. trainer.fit(ckpt_path=...) would raise a KeyError
    # trying to restore them. In that case: weights are already in
    # lit_model — just clear ckpt_file so trainer.fit does a weights-only
    # resume, and shrink max_epochs to only run the remaining epochs.
    if ckpt_file is not None:
        _meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if "optimizer_states" not in _meta:
            _saved_epoch = _meta.get("epoch", 0)
            epochs = max(1, epochs - _saved_epoch - 1)
            print(
                f"Checkpoint saved with save_weights_only=True (no optimizer state). "
                f"Weights loaded from epoch {_saved_epoch}; running {epochs} more epochs."
            )
            ckpt_file = None
        del _meta

    # If no checkpoint loaded, initialize a new model
    if not _weights_loaded:
        lit_model = LitWrapper(
            model=model,
            optimizer=optimizer,
            train_config=experiment.phase_configs.train,
            validation_config=experiment.phase_configs.validation,
            scheduler_config=getattr(experiment.misc, "lr_scheduler", None),
        )

    # Callbacks (logger was constructed at the top of ``run_training``
    # so dataset/model construction prints land in the wandb console log).
    callbacks = initialize_callbacks(cfg, experiment)

    accelerator = (
        "cpu" if torch.backends.mps.is_available() else "auto"
    )  # matrix inverse is not compatible with mps for some reason

    # For IterableDatasets, num_batches is a global count across all DDP
    # workers. adjust_num_batches() stripes batches as
    # batches[global_worker_id::total_workers], so when
    # num_batches % total_workers != 0 the remainder goes to the first
    # (num_batches % total_workers) workers, which may all land on rank 0.
    # Using floor(num_batches / total_workers) * num_workers_per_rank
    # guarantees every rank sees the same batch count regardless of the
    # remainder. For MapDatasets, num_batches is absent and
    # len(dataloader) is already per-rank.
    _num_train_workers = max(1, getattr(experiment.misc, "num_workers", 1))
    _train_num_batches = getattr(train_dataset, "num_batches", None)
    if _train_num_batches is not None:
        _total_train_workers = _world_size * _num_train_workers
        _train_num_batches = (_train_num_batches // _total_train_workers) * _num_train_workers
    else:
        _train_num_batches = len(train_dataloader)

    _num_eval_workers = max(1, getattr(experiment.misc, "num_eval_workers", 1))
    _val_num_batches = None
    if validation_dataloader is not None:
        _val_num_batches = getattr(validation_dataset, "num_batches", None)
        if _val_num_batches is not None:
            _total_eval_workers = _world_size * _num_eval_workers
            _val_num_batches = (_val_num_batches // _total_eval_workers) * _num_eval_workers
        else:
            _val_num_batches = len(validation_dataloader)

    # Configure trainer with common parameters
    trainer_config = {
        "logger": logger,
        "max_epochs": epochs,
        "limit_train_batches": _train_num_batches,
        **({"limit_val_batches": _val_num_batches} if _val_num_batches is not None else {}),
        "log_every_n_steps": (
            getattr(experiment.misc, "log_interval", 50) if not logger else None
        ),
        "devices": "auto",
        "accelerator": accelerator,
        "strategy": DDPStrategy(find_unused_parameters=True) if _world_size > 1 else "auto",
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": (
            getattr(experiment.misc, "check_validation_every_n_epoch", 1)
        ),
        "gradient_clip_val": (getattr(experiment.misc, "gradient_clip_val", 0.5)),
        "callbacks": callbacks,
        "enable_progress_bar": (getattr(experiment.misc, "enable_progress_bar", True)),
        "enable_checkpointing": bool(
            checkpointing_config.get("local", False)
            or checkpointing_config.get("wandb", False)
        ),
        "enable_model_summary": True,
    }

    # Initialize the trainer
    trainer = pl.Trainer(**trainer_config)

    # Log hardware information
    print_hardware_info()

    # Log training configuration
    print_training_config(
        trainer,
        train_dataloader,
        validation_dataloader,
        epochs,
        train_batches_per_rank=_train_num_batches,
        val_batches_per_rank=_val_num_batches,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        ckpt_path=ckpt_file,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Thin wrapper around ``run_training``. The
    tf32 knob lives here (not in ``run_training``) so tests that call
    ``run_training`` directly don't implicitly inherit it.
    """
    torch.set_float32_matmul_precision("high")
    run_training(cfg)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Example commands — train a CNP on each benchmark.
#
# Synthetic 1D functions (GP-RBF by default; select model via model/synthetic=<name>):
#   python train.py +experiment=synthetic/default model/synthetic=cnp
#
# Image completion (CIFAR-10 by default; swap via benchmark/image=base-{dtd,svhn,...}):
#   python train.py +experiment=image/default model/image=cnp
#   python train.py +experiment=image/default model/image=cnp benchmark/image=base-dtd
#   python train.py +experiment=image/default model/image=cnp benchmark/image=base-svhn
#
# ERA5 climate reanalysis:
#   python train.py +experiment=era5/default model/era5=cnp
#
# Kolmogorov flow (2D PDE):
#   python train.py +experiment=kolmogorov/default model/kolmogorov=cnp
#
# Predator-prey (Lotka-Volterra simulation):
#   python train.py +experiment=predprey/default model/predprey=cnp
# -----------------------------------------------------------------------------
