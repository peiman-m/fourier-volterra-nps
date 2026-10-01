"""Hydra scaffolding smoke check.

Composes ``conf/config.yaml`` and asserts:

1. ``os.getcwd()`` is unchanged after ``@hydra.main`` fires. Regression
   guard for ``hydra.job.chdir: false`` — if that flips, every relative
   path (W&B cleanup, checkpoint resume, plot dirs) silently breaks.
2. ``hydra.run.dir`` resolves to the same value as ``misc.artifacts_dir``,
   proving the ``artifacts_dir → experiment_name / model_run_name / seed``
   interpolation chain works at Hydra-init time.
3. A ``${eval:...}`` expression in a path-contributing field resolves
   successfully, proving ``conf/_resolvers.py`` registered the custom
   resolvers before Hydra resolves ``hydra.run.dir``.

This script instantiates nothing — no model, no dataset, no trainer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Importing this module registers eval/product/range with OmegaConf at
# import time, before @hydra.main resolves anything.
from conf import _resolvers  # noqa: F401,E402

_CWD_AT_IMPORT = os.getcwd()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cwd_now = os.getcwd()
    if cwd_now != _CWD_AT_IMPORT:
        raise AssertionError(
            f"cwd changed under @hydra.main: {_CWD_AT_IMPORT!r} -> {cwd_now!r}. "
            "Expected hydra.job.chdir=false in conf/config.yaml."
        )

    hydra_run_dir = HydraConfig.get().run.dir
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_container, dict)
    artifacts_dir = cfg_container["misc"]["artifacts_dir"]
    if hydra_run_dir != artifacts_dir:
        raise AssertionError(
            f"hydra.run.dir != misc.artifacts_dir: "
            f"{hydra_run_dir!r} vs {artifacts_dir!r}"
        )

    probe = OmegaConf.create({"x": "${eval:'1 + 1'}"})
    probe_container = OmegaConf.to_container(probe, resolve=True)
    assert isinstance(probe_container, dict)
    probe_value = probe_container["x"]
    if probe_value != 2:
        raise AssertionError(
            f"eval resolver did not fire correctly: got {probe_value!r}, expected 2"
        )

    probe_path = OmegaConf.create(
        {
            "misc": {
                "experiment_name": "smoke",
                "model_run_name": "run-${eval:'1 + 1'}",
                "seed": 0,
                "artifacts_dir": (
                    "artifacts/${misc.experiment_name}"
                    "/model=${misc.model_run_name}/seed=${misc.seed}"
                ),
            }
        }
    )
    probe_path_container = OmegaConf.to_container(probe_path, resolve=True)
    assert isinstance(probe_path_container, dict)
    resolved_path = probe_path_container["misc"]["artifacts_dir"]
    if "run-2" not in resolved_path:
        raise AssertionError(
            f"eval resolver did not fire inside a path-contributing field: "
            f"{resolved_path!r}"
        )

    print("hydra_smoke: cwd unchanged                           OK")
    print(f"hydra_smoke: hydra.run.dir = {hydra_run_dir}")
    print(f"hydra_smoke: misc.artifacts_dir = {artifacts_dir}")
    print(f"hydra_smoke: eval-in-path resolved to {resolved_path}")
    print("hydra_smoke: all assertions passed")


if __name__ == "__main__":
    sys.exit(main())
