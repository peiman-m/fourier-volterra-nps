"""Hydra-based evaluation entry point.

Split into ``run_eval(cfg)`` and ``@hydra.main main(cfg)`` mirroring
``train.py``. Invoke with Hydra overrides:

    python eval.py +experiment=synthetic/default model/synthetic=cnp misc.wandb_run_id=<id>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
from omegaconf import DictConfig
import torch
import wandb

from conf import _resolvers  # noqa: F401  (registers eval/product/range)
from utils.experiment import (
    LitWrapper,
    adjust_num_batches,
    create_dataloader,
    evaluate_model,
    find_checkpoint_paths,
    init_wandb_run,
    initialize_experiment,
    log_results,
)


def run_eval(cfg: DictConfig) -> None:
    """Full eval body. Hydra-composed ``cfg`` in, evaluation loop out."""
    try:
        # Initialize wandb BEFORE instantiate so stdout from dataset
        # construction, model ``__init__``, and the rest of the Hydra
        # instantiate walk is captured by wandb's console-log redirect.
        # ``init_wandb_run(cfg)`` reads only scalar ``cfg.misc.*`` fields
        # that exist on the raw cfg.
        wandb_run = None
        if cfg.misc.wandb_logging_enabled:
            try:
                wandb_run = init_wandb_run(cfg)
                if wandb_run:
                    print(
                        "Initialized shared wandb run for all evaluations: "
                        f"{wandb_run.name} (ID: {wandb_run.id})"
                    )
            except Exception as e:
                print(f"Error initializing shared wandb run: {e}")

        experiment = initialize_experiment(cfg)

        dl_config = getattr(experiment.data, "dataloaders", {})
        test_dl_config = dl_config.get("test", {})

        test_loader = create_dataloader(
            dataset=experiment.data.datasets.test,
            num_workers=getattr(
                experiment.misc,
                "num_test_workers",
                getattr(experiment.misc, "num_eval_workers", 1),
            ),
            # Stripe the deterministic test epoch across workers (mirrors
            # train.py). Without this, every worker replays the full epoch,
            # inflating the test loop by a factor of num_workers.
            worker_init_fn=adjust_num_batches,
            pin_memory=getattr(experiment.misc, "pin_memory", True),
            batch_size=test_dl_config.get("batch_size", 1),
            shuffle=test_dl_config.get("shuffle", False),
            drop_last=test_dl_config.get("drop_last", True),
        )

        wandb_run_id = getattr(experiment.misc, "wandb_run_id", None)
        checkpoint_paths = find_checkpoint_paths(experiment, wandb_run_id)

        for checkpoint_type, checkpoint_path in checkpoint_paths.items():
            print(f"\nEvaluating {checkpoint_type} checkpoint: {checkpoint_path}")

            model = LitWrapper.load_from_checkpoint(
                checkpoint_path,
                model=experiment.model,
                test_config=experiment.phase_configs.test,
            )

            if model is None:
                print(
                    f"Failed to load {checkpoint_type} checkpoint. "
                    "Skipping evaluation."
                )
                continue

            results = evaluate_model(model, test_loader, checkpoint_type, experiment)

            log_results(results, experiment, checkpoint_type, wandb_run)

            print(f"\nResults for {checkpoint_type} checkpoint:")
            for metric, value in results["metrics"].items():
                print(f"  {metric}: {value}")
            if hardware := results.get("hardware"):
                print("  Hardware:")
                for k, v in hardware.items():
                    print(f"    {k}: {v:.6f}")

        if wandb_run is not None:
            wandb.finish()
            print(f"Shared wandb run finished: {wandb_run.name} (ID: {wandb_run.id})")

        print("\nEvaluation complete!")

    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback

        traceback.print_exc()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Thin wrapper around ``run_eval``."""
    torch.set_float32_matmul_precision("high")
    run_eval(cfg)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Example commands — evaluate a trained CNP checkpoint on each benchmark.
# Replace <id> with your W&B run path (entity/project/run_id).
#
# Synthetic 1D functions (select model via model/synthetic=<name>):
#   python eval.py +experiment=synthetic/default model/synthetic=cnp misc.wandb_run_id=<id>
#
# Image completion (CIFAR-10 by default):
#   python eval.py +experiment=image/default model/image=cnp misc.wandb_run_id=<id>
#
# ERA5 climate reanalysis:
#   python eval.py +experiment=era5/default model/era5=cnp misc.wandb_run_id=<id>
#
# Kolmogorov flow:
#   python eval.py +experiment=kolmogorov/default model/kolmogorov=cnp misc.wandb_run_id=<id>
#
# Predator-prey — simulated test split:
#   python eval.py +experiment=predprey/default model/predprey=cnp benchmark/predprey=test_sim misc.wandb_run_id=<id>
#
# Predator-prey — real Hudson Bay hare-lynx data:
#   python eval.py +experiment=predprey/default model/predprey=cnp benchmark/predprey=test_real misc.wandb_run_id=<id>
# -----------------------------------------------------------------------------
