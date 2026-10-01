import json
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.wandb_run import Run

from .callbacks import LogPerformanceCallback
from .lightning_wrapper import LitWrapper


def ensure_directory_exists(dirpath: Path | str) -> Path | None:
    """Create directory if it doesn't exist yet. Returns None for a falsy path."""
    if dirpath:
        p = Path(dirpath)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {p}")
        else:
            print(f"Directory already exists at: {p}")
        return p
    return None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        # np.int32/int64 (and float32/64) are subclasses of np.integer /
        # np.floating; the broad base classes cover them and avoid passing
        # the generic numpy scalar aliases to isinstance.
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


def _find_local_checkpoints(dirpath: Path) -> dict[str, str]:
    """
    Search a local directory for 'last' and 'best' checkpoints.
    """
    found: dict[str, str] = {}

    # Direct filename for last checkpoint
    last_path = dirpath / "last.ckpt"
    if last_path.exists():
        found["last"] = str(last_path)

    # Alternative last: any file with 'last'
    if "last" not in found:
        candidates = sorted(dirpath.glob("*last*.ckpt"))
        if candidates:
            found["last"] = str(candidates[0])

    # Direct filename for best checkpoint
    best_path = dirpath / "best.ckpt"
    if best_path.exists():
        found["best"] = str(best_path)

    # Glob fallback for best
    if "best" not in found:
        candidates = sorted(dirpath.glob("*best*.ckpt"))
        if candidates:
            found["best"] = str(candidates[0])

    return found


def _find_wandb_checkpoints(
    experiment_config: DictConfig,
    run_id: str | None = None,
) -> dict[str, str]:
    """
    Retrieve checkpoints from a Weights & Biases run.
    """
    api = wandb.Api()
    target = run_id or (
        f"{experiment_config.misc.wandb_user}/"
        f"{experiment_config.misc.project}/"
        f"{experiment_config.misc.name}"
    )
    print(f"Retrieving W&B run '{target}'")
    run = api.run(target)
    found: dict[str, str] = {}

    for artifact in run.logged_artifacts():
        if "last" in found and "best" in found:
            break
        name = artifact.name.lower()
        if "checkpoint" not in name:
            continue
        download_dir = artifact.download()
        model_file = Path(download_dir) / "model.ckpt"
        if model_file.exists():
            if "last" in name and "last" not in found:
                found["last"] = str(model_file)
            if "best" in name and "best" not in found:
                found["best"] = str(model_file)

    # Update run info in experiment_config
    if run_id and found:
        experiment_config.misc.wandb_run_id = run_id
        experiment_config.misc.name = run.name

    return found


def find_checkpoint_paths(
    experiment_config: DictConfig,
    wandb_run_id: str | None = None,
) -> dict[str, str]:
    """Find last checkpoint from local directory or W&B run."""

    # Try local directory first
    checkpoint_dir = getattr(experiment_config.misc, "checkpoint_dirpath", None)
    if checkpoint_dir:
        local_checkpoints = _find_local_checkpoints(Path(checkpoint_dir))
        if local_checkpoints:
            print("Found checkpoints locally.")
            return local_checkpoints

    # Try W&B if enabled or run ID provided
    checkpointing = getattr(experiment_config.misc, "checkpointing", {})
    wandb_enabled = (
        getattr(experiment_config.misc, "wandb_logging_enabled", False)
        and checkpointing.get("wandb", False)
    )

    if wandb_run_id or wandb_enabled:
        try:
            wandb_checkpoints = _find_wandb_checkpoints(experiment_config, wandb_run_id)
            if wandb_checkpoints:
                print("Successfully retrieved checkpoint.")
                return wandb_checkpoints
        except Exception as e:
            raise RuntimeError(f"Error retrieving from W&B: {e}")

    raise RuntimeError("No checkpoint found locally or in W&B")


def evaluate_model(
    model: LitWrapper,
    test_loader: torch.utils.data.DataLoader,
    checkpoint_type: str,
    experiment_config: DictConfig,
) -> dict[str, Any]:
    """Evaluate model on test data.

    Args:
        model: Loaded model
        test_loader: DataLoader for test data
        checkpoint_type: Type of checkpoint ('best' or 'last')
        experiment_config: Experiment configuration

    Returns:
        Dictionary with evaluation results
    """
    # Set up trainer to evaluate the model
    callbacks = []
    if getattr(experiment_config.misc, "hardware_callback", False):
        callbacks.append(LogPerformanceCallback())

    trainer = pl.Trainer(
        accelerator=(
            "cpu" if torch.backends.mps.is_available() else "auto"
        ),  # matrix inverse is not compatible with mps
        devices=1,  # Always single-GPU: eval.py manages its own lifecycle
        logger=False,  # No logging during evaluation
        enable_checkpointing=False,  # No checkpointing during evaluation
        callbacks=callbacks or None,
    )

    # Run test
    test_results = trainer.test(model, dataloaders=test_loader)[0]

    # Format results for saving/logging
    results = {
        "checkpoint_type": checkpoint_type,
        "model_name": experiment_config.misc.model_run_name,
        "experiment_name": experiment_config.misc.experiment_name,
        "metrics": test_results,
    }

    # Attach hardware stats if the callback was active
    for cb in cast(Any, trainer).callbacks:
        if isinstance(cb, LogPerformanceCallback):
            hardware = cb.get_test_summary()
            if hardware:
                results["hardware"] = hardware
            break

    return results


def init_wandb_run(cfg: DictConfig) -> Run | None:
    """Initialize or resume a wandb run for an eval invocation.

    Takes the raw Hydra ``cfg`` (not an instantiated ``experiment``) so it
    can be called before ``initialize_experiment`` — that way stdout from
    dataset construction and model ``__init__`` is captured by wandb's
    console-log redirect. Every field read here (``misc.project``,
    ``misc.wandb_run_id``, ``misc.eval_name``) exists on the raw cfg.
    """
    # cfg is a DictConfig, so to_container returns a str-keyed dict; wandb.init
    # types its config as dict[str, Any].
    payload = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    job_type = "evaluation"
    project = cfg.misc.project

    try:
        if run_id_full := cfg.misc.get("wandb_run_id", None):
            # Always prioritize using an existing run_id if available
            run_id = run_id_full.split("/")[-1]
            run = wandb.init(
                id=run_id,
                resume="allow",
                project=project,
                config=payload,
                job_type=job_type,
            )
            print(f"Resumed W&B run {run.name} (ID: {run.id})")
        else:
            # Only create a new run if no run_id was provided
            run = wandb.init(
                project=project,
                name=cfg.misc.eval_name,
                config=payload,
                job_type=job_type,
            )
            print(f"Created new W&B run {run.name} (ID: {run.id})")

        return run
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}")
        return None


def _save_locally(
    results: dict[str, Any],
    checkpoint_type: str,
    metrics_dir: Path,
) -> None:
    """Dump the full results dict to a JSON file under metrics_dir."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    ensure_directory_exists(metrics_dir)
    filepath = metrics_dir / f"test_results_{checkpoint_type}.json"
    with filepath.open("w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved results to {filepath}")


def _log_to_wandb(
    run: Run,
    metrics: dict[str, float],
    checkpoint_type: str,
) -> None:
    """Log metrics and results table to the given wandb run."""
    # Prefix metrics with checkpoint type
    prefixed = {f"{checkpoint_type}/{k}": v for k, v in metrics.items()}
    run.summary.update(prefixed)

    # # Build a Table artifact
    # table = wandb.Table(
    #     columns=["checkpoint_type", "metric", "value"],
    #     data=[[checkpoint_type, k, v] for k, v in metrics.items()],
    # )
    # run.log({f"results_{checkpoint_type}": table})

    print(
        f"Logged {len(metrics)} metrics to W&B summary for "
        f"checkpoint '{checkpoint_type}'"
    )


def log_results(
    results: dict[str, Any],
    experiment_config: DictConfig,
    checkpoint_type: str,
    wandb_run: Run | None = None,
) -> None:
    """
    Log evaluation results to Weights & Biases and/or a local file.

    Args:
        results: Dictionary with evaluation results
        experiment_config: Experiment configuration
        checkpoint_type: Type of checkpoint ('best' or 'last')
        wandb_run: Optional existing wandb run to use

    """
    if getattr(experiment_config.misc, "wandb_logging_enabled", False) and wandb_run:
        try:
            metrics_to_log = {**results["metrics"], **results.get("hardware", {})}
            _log_to_wandb(wandb_run, metrics_to_log, checkpoint_type)
        except Exception as e:
            print(f"Failed to log results to W&B: {e}")

    if metrics_dir := getattr(experiment_config.misc, "metrics_dirpath", None):
        try:
            _save_locally(results, checkpoint_type, Path(metrics_dir))
        except Exception as e:
            print(f"Failed to save results locally: {e}")
