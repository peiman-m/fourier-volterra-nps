"""Effective Receptive Field (ERF) analysis for 1-D synthetic NP models.

Implements the gradient-based ERF measurement of Luo, Li, Urtasun, Zemel
(NeurIPS 2016, "Understanding the Effective Receptive Field in Deep CNNs"),
adapted to neural processes:

    ERF(x_c) ∝ E_{y_c} [ ( ∂ s(x_t*; x_c, y_c) / ∂ y_c(x_c) )^2 ]

where the scalar ``s`` backpropped through the model is selectable via
``erf.target``:

* ``mean`` (default, Luo-faithful): ``s = μ(x_t*)`` — the predictive mean.
* ``var``                         : ``s = σ²(x_t*)`` — the predictive variance.
* ``logp``                        : ``s = log p(y_t*=0 | x_c, y_c, x_t*)`` —
  the full log-likelihood at a fixed evaluation point. Captures both mean
  and uncertainty channels but is dominated by ``1/σ²`` and ``1/σ⁴`` terms
  when the model is highly confident, which can amplify the ERF by many
  orders of magnitude on architectures like SF-ConvCNP. Use with care.

For each trained checkpoint, the same architecture is also evaluated with
random initialization, giving the before/after-training comparison from
§3.2 of the paper.

The script is a Hydra app, invoked exactly like ``eval.py`` plus an
``+erf=default`` group loader and optional per-field overrides. Example::

    python tools/analyze_erf.py \\
        +experiment=synthetic/default \\
        model/synthetic=convcnp-unet \\
        benchmark/synthetic/output_generator=gp-periodic \\
        misc.seed=2 \\
        +erf=default \\
        erf.n_samples=2048

(Without ``+erf=default`` you can still pass individual fields via
``+erf.n_samples=...`` etc., but the group loader is recommended on the
cluster.)

Optional ``erf.*`` overrides (note: bare keys, no leading ``+``, once the
group is loaded):

    erf.n_grid      (int, default 512)   dense context-grid size
    erf.x_max       (float, default = max(|context_range|) from the training
                     input generator) — grid spans [-x_max, x_max]
    erf.n_samples   (int, default 64)    Monte-Carlo samples for E_{y_c}
    erf.batch_size  (int, default = n_samples) chunk size for the backward
                     pass; larger K runs in ceil(n_samples/batch_size) chunks
                     with squared-gradients accumulated across them
    erf.targets     (list[float], default [-2,-1,0,1,2])  query locations
    erf.training_domain (list[float], default [-3.0, 3.0])
    erf.y_sources   (list[str], default ["normal","task"])
    erf.variants    (list[str], default ["trained","random"])
    erf.target      (str, default "mean")  scalar to backprop: one of
                     ``{"mean", "var", "logp"}``
    erf.output      (str)   path for the PDF figure (required to save)
    erf.device      (str, default "cpu")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from conf import _resolvers  # noqa: F401  (registers eval/product/range)
from utils.experiment import LitWrapper, find_checkpoint_paths, initialize_experiment


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# ERF computation
# ---------------------------------------------------------------------------

def _make_xc_grid(n_grid: int, x_max: float) -> torch.Tensor:
    """Return shape [n_grid, 1] uniform grid over [-x_max, x_max]."""
    return torch.linspace(-x_max, x_max, n_grid).unsqueeze(-1)


def _sample_yc_normal(n_samples: int, n_grid: int) -> torch.Tensor:
    """Return yc ~ N(0, 1) of shape [n_samples, n_grid, 1]."""
    return torch.randn(n_samples, n_grid, 1)


def _sample_yc_task(
    output_generator: Any,
    xc_grid: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Sample yc from the task's true generative process.

    Calls ``output_generator.sample(x)`` ``n_samples`` times so each draw
    uses a fresh set of kernel hyperparameters (GP) or fresh frequency /
    amplitude / phase (sawtooth, squarewave), matching the per-batch
    variability seen at training time.
    """
    import warnings

    # Shape expected by output_generator.sample: (batch_size, N, dim)
    x_batched = xc_grid.unsqueeze(0)  # [1, N, 1]
    samples = []
    # GP draws on a dense grid commonly require jitter for Cholesky;
    # gpytorch handles this gracefully but issues a NumericalWarning each
    # time. Suppress them so the progress output stays readable.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*jitter.*", category=Warning
        )
        warnings.filterwarnings(
            "ignore", message=".*not positive definite.*", category=Warning
        )
        for _ in range(n_samples):
            y, _ = output_generator.sample(x_batched)  # [1, N, 1]
            samples.append(y)
    return torch.cat(samples, dim=0)  # [n_samples, n_grid, 1]


_TARGET_CHOICES = ("mean", "var", "logp")


def _scalar_from_dist(
    dist: Any, yt: torch.Tensor, target: str
) -> torch.Tensor:
    """Return the per-sample scalar to backprop through the model.

    ``yt`` is the fixed evaluation point used only when ``target == "logp"``;
    the other targets ignore it. The return value has its sample dimension
    summed so a single ``.backward()`` populates ``yc.grad`` correctly (the
    cross-sample partials are zero, so summing does not entangle samples).
    """
    if target == "mean":
        return dist.mean.sum()
    if target == "var":
        # ``.variance`` is implemented by every torch.distributions.Distribution
        # subclass we use here (Normal, LowRankMultivariateNormal, …).
        return dist.variance.sum()
    if target == "logp":
        return dist.log_prob(yt).sum()
    raise ValueError(
        f"unknown erf.target={target!r}; expected one of {_TARGET_CHOICES}"
    )


def _erf_at_target(
    model: torch.nn.Module,
    xc_grid: torch.Tensor,        # [n_grid, 1]
    yc_samples: torch.Tensor,     # [K, n_grid, 1]
    x_target: float,
    y_target: float = 0.0,
    device: torch.device | str = "cpu",
    batch_size: int | None = None,
    target: str = "mean",
) -> np.ndarray:
    """Compute the per-location squared gradient ⟨(∂s / ∂y_c[i])²⟩, where
    ``s`` is selected by ``target`` (see ``_scalar_from_dist``).

    Splits the K Monte-Carlo samples into chunks of ``batch_size`` so that
    peak memory scales with the chunk size rather than K. The estimator
    itself is unchanged: we accumulate ∑_k g_{k,i}² across chunks and
    divide by K at the end.

    Returns
    -------
    erf : ndarray of shape [n_grid]
        Monte-Carlo estimate of the variance of ∂s(x_t* | xc, yc) / ∂y_c.
    """
    K = yc_samples.shape[0]
    n_grid = yc_samples.shape[1]
    if batch_size is None or batch_size <= 0:
        batch_size = K

    accum = torch.zeros(n_grid, dtype=torch.float64)  # CPU accumulator

    starts = list(range(0, K, batch_size))
    chunk_iter = tqdm(
        starts,
        desc=f"    xt={x_target:+.1f}",
        leave=False,
        file=sys.stdout,
        mininterval=10.0,
        ascii=True,
    )
    for start in chunk_iter:
        end = min(start + batch_size, K)
        k_sub = end - start

        xc = xc_grid.unsqueeze(0).expand(k_sub, -1, -1).to(device)  # [k, N, 1]
        xq = torch.full((k_sub, 1, 1), float(x_target), device=device)
        yc = (
            yc_samples[start:end]
            .detach()
            .clone()
            .to(device)
            .requires_grad_(True)
        )

        dist = model(xc=xc, yc=yc, xq=xq)
        yt = torch.full((k_sub, 1, 1), float(y_target), device=device)
        loss = _scalar_from_dist(dist, yt, target)
        loss.backward()

        grad = yc.grad  # [k_sub, N, 1]
        assert grad is not None, "yc.grad is None — autograd may be disabled"

        # Sum of squared grads over this chunk; accumulate on CPU in float64
        # so many small partial sums don't lose precision.
        accum += (
            grad.detach().pow(2).sum(dim=0).squeeze(-1).double().cpu()
        )

        # Release the chunk's autograd graph before the next iteration.
        del xc, xq, yc, dist, loss, grad

    erf = (accum / float(K)).numpy()  # [N]
    return erf


def _two_sigma_radius(
    erf: np.ndarray, xc: np.ndarray, x_center: float, mass: float = 0.9545
) -> float:
    """Return smallest r such that the fraction of total energy within
    [x_center - r, x_center + r] is at least ``mass`` (default 95.45%,
    Luo et al.'s 2-σ definition)."""
    total = erf.sum()
    if total <= 0:
        return float("nan")
    # Sort grid points by distance from x_center.
    dist_from_center = np.abs(xc - x_center)
    order = np.argsort(dist_from_center)
    cumulative = np.cumsum(erf[order]) / total
    idx = np.searchsorted(cumulative, mass)
    if idx >= len(order):
        return float(dist_from_center[order[-1]])
    return float(dist_from_center[order[idx]])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_trained_model(
    experiment: DictConfig, device: torch.device | str
) -> torch.nn.Module:
    """Load weights from the local checkpoint into experiment.model."""
    wandb_run_id = getattr(experiment.misc, "wandb_run_id", None)
    checkpoint_paths = find_checkpoint_paths(experiment, wandb_run_id)
    ckpt_path = checkpoint_paths.get("best", next(iter(checkpoint_paths.values())))
    print(f"[trained] loading checkpoint: {ckpt_path}")
    test_config = getattr(
        getattr(experiment, "phase_configs", None), "test", None
    )
    lit = LitWrapper.load_from_checkpoint(
        ckpt_path,
        model=experiment.model,
        test_config=test_config,
    )
    model = lit.model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_erf(
    results: dict,
    xc: np.ndarray,
    targets: list[float],
    training_domain: tuple[float, float],
    output_path: str | None,
    model_name: str,
    task_name: str,
    target: str = "mean",
) -> None:
    """Plot the ERF profiles in a (variants × y-sources) grid."""
    variants = list(results.keys())
    y_sources = list(next(iter(results.values())).keys())
    n_rows, n_cols = len(variants), len(y_sources)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    colors = matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, len(targets)))

    for i, variant in enumerate(variants):
        for j, source in enumerate(y_sources):
            ax = axes[i, j]
            ax.set_title(f"{variant} | $y_c$ ~ {source}")
            ax.axvspan(
                training_domain[0],
                training_domain[1],
                alpha=0.08,
                color="steelblue",
                label="Training domain" if (i == 0 and j == 0) else None,
            )
            ax.axhline(0, color="gray", lw=0.5)

            erf_per_target = results[variant][source]
            for k, xt in enumerate(targets):
                erf = erf_per_target[xt]
                if erf.max() > 0:
                    erf_norm = erf / erf.max()
                else:
                    erf_norm = erf
                radius = _two_sigma_radius(erf, xc, xt)
                ax.plot(
                    xc,
                    erf_norm,
                    color=colors[k],
                    lw=1.4,
                    label=f"$x_t^*={xt:+.1f}$  (2σ={radius:.2f})",
                )
                ax.axvline(xt, color=colors[k], lw=0.5, ls=":", alpha=0.5)

            ax.set_xlabel("$x_c$")
            if j == 0:
                ax.set_ylabel("Normalized ERF")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.5)
            ax.legend(loc="lower right", framealpha=0.85)

    target_label = {
        "mean": r"target $=\mu(x_t^*)$",
        "var": r"target $=\sigma^2(x_t^*)$",
        "logp": r"target $=\log p(y_t^*{=}0)$",
    }.get(target, f"target={target}")
    fig.suptitle(
        f"Effective Receptive Field — model: {model_name}, task: {task_name}"
        f"  ({target_label})",
        y=1.00,
    )
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"\nFigure saved to: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _write_results_json(
    output_dir: Path,
    cfg: DictConfig,
    erf_cfg: dict[str, Any],
    results: dict[str, dict[str, dict[float, np.ndarray]]],
    xc: np.ndarray,
) -> Path:
    """Write a scalar summary of the run: per-cell 2σ radius + peak,
    along with the task/model/seed identity and the ERF config that
    produced them. Designed for trivial pandas ingestion across jobs."""
    cells = []
    for variant, by_source in results.items():
        for source, by_target in by_source.items():
            for xt, erf in by_target.items():
                cells.append(
                    {
                        "variant": variant,
                        "y_source": source,
                        "x_target": float(xt),
                        "two_sigma_radius": float(
                            _two_sigma_radius(erf, xc, float(xt))
                        ),
                        "peak": float(erf.max()),
                        "energy": float(erf.sum()),
                    }
                )

    target = erf_cfg.get("target", "mean")
    backprop_descr = {
        "mean": "predictive_mean_at_xt",
        "var": "predictive_variance_at_xt",
        "logp": "log_likelihood_at_yt_eq_0",
    }.get(target, target)

    payload = {
        "task": str(cfg.misc.experiment_name),
        "model": str(cfg.model_name),
        "seed": int(cfg.misc.seed),
        "artifacts_dir": str(cfg.misc.artifacts_dir),
        "backprop_scalar": backprop_descr,
        "config": {
            k: erf_cfg[k]
            for k in (
                "n_samples",
                "batch_size",
                "n_grid",
                "x_max",
                "training_domain",
                "targets",
                "variants",
                "y_sources",
                "target",
                "device",
            )
        },
        "cells": cells,
    }

    out = output_dir / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to: {out}")
    return out


def _write_curves_npz(
    output_dir: Path,
    xc: np.ndarray,
    targets: list[float],
    results: dict[str, dict[str, dict[float, np.ndarray]]],
) -> Path:
    """Write raw ERF profiles to ``curves.npz`` so downstream analysis
    (re-plotting, alternative radius definitions, seed-averaging) does
    not need to recompute gradients.

    Layout::

        xc:               shape [N]        # shared context grid
        targets:          shape [T]        # query locations xt*
        <variant>_<src>:  shape [T, N]     # one matrix per cell;
                                           # row i is the ERF profile at xt*=targets[i]

    Cells absent from a particular run (e.g. when ``erf.variants=[random]``
    is requested) are simply not present in the file.
    """
    targets_arr = np.asarray(targets, dtype=np.float64)

    arrays: dict[str, np.ndarray] = {"xc": xc, "targets": targets_arr}
    for variant, by_source in results.items():
        for source, by_target in by_source.items():
            # Stack rows in the same order as the ``targets`` array so
            # row index ↔ target index is unambiguous downstream.
            stacked = np.stack(
                [by_target[float(xt)] for xt in targets], axis=0
            )  # [T, N]
            arrays[f"{variant}_{source}"] = stacked

    out = output_dir / "curves.npz"
    np.savez_compressed(str(out), **arrays)
    print(f"Curves saved to: {out}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _default_x_max_from_cfg(cfg: DictConfig) -> float:
    """Half-width of the training context range, as set in the input
    generator config. The ERF should be measured on the same input
    distribution the model was trained on; sampling outside it probes
    OOD behavior rather than the receptive field."""
    ctx_range = cfg.input_generators.train.context_range
    a, b = float(ctx_range[0]), float(ctx_range[1])
    return max(abs(a), abs(b))


def _default_training_domain_from_cfg(cfg: DictConfig) -> tuple[float, float]:
    ctx_range = cfg.input_generators.train.context_range
    return (float(ctx_range[0]), float(ctx_range[1]))


def _default_output_path(cfg: DictConfig) -> str:
    """Conventional location for the ERF figure: next to metrics/ and plots/
    in the per-checkpoint artifact directory. Mirrors how the rest of the
    project organizes outputs by ``(task, model, seed)``."""
    artifacts_dir = str(cfg.misc.artifacts_dir)
    return str(Path(artifacts_dir) / "erf" / "erf.pdf")


def _get_erf_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Read ``erf.*`` overrides with sensible defaults."""
    erf = cfg.get("erf", OmegaConf.create({})) or OmegaConf.create({})
    default_x_max = _default_x_max_from_cfg(cfg)
    default_train_dom = _default_training_domain_from_cfg(cfg)
    n_samples = int(erf.get("n_samples", 64))
    output = erf.get("output", None)
    if output is None:
        output = _default_output_path(cfg)
    target = str(erf.get("target", "mean"))
    if target not in _TARGET_CHOICES:
        raise ValueError(
            f"erf.target={target!r} not in {_TARGET_CHOICES}"
        )
    return {
        "n_grid": int(erf.get("n_grid", 512)),
        "x_max": float(erf.get("x_max", default_x_max)),
        "n_samples": n_samples,
        "batch_size": int(erf.get("batch_size", n_samples)),
        "targets": list(erf.get("targets", [-2.0, -1.0, 0.0, 1.0, 2.0])),
        "training_domain": tuple(erf.get("training_domain", default_train_dom)),
        "y_sources": list(erf.get("y_sources", ["normal", "task"])),
        "variants": list(erf.get("variants", ["trained", "random"])),
        "target": target,
        "output": str(output),
        "device": str(erf.get("device", "cpu")),
    }


def run_erf(cfg: DictConfig) -> None:
    """Body of the ERF analysis. Hydra-composed cfg in, figure + stdout out."""
    erf_cfg = _get_erf_cfg(cfg)
    print(f"ERF config: {erf_cfg}")

    # Build experiment once for the trained model.
    experiment = initialize_experiment(cfg)

    # Sanity check: 1-D x only.
    x_dim = int(cfg.params.x_dim)
    if x_dim != 1:
        raise ValueError(
            f"analyze_erf only supports 1-D inputs (x_dim=1); got x_dim={x_dim}."
        )

    device = torch.device(erf_cfg["device"])

    # Build a fresh, randomly-initialized model from the same cfg.model.
    # initialize_experiment(cfg) above already consumed the seeded RNG to
    # build experiment.model; re-seed (with an offset) so the random
    # variant is reproducible w.r.t. cfg.misc.seed but distinct from
    # whatever init experiment.model started from.
    from hydra.utils import instantiate

    torch.manual_seed(int(cfg.misc.seed) + 31337)
    fresh_model = instantiate(cfg.model, _convert_="all")

    # Data generator for the "task" y-source.
    test_dataset = experiment.data.datasets.test
    output_generator = test_dataset.output_generator
    print(
        f"[task gen] {type(output_generator).__name__}, "
        f"noise_std={getattr(output_generator, 'noise_std', None)}"
    )

    xc_grid = _make_xc_grid(erf_cfg["n_grid"], erf_cfg["x_max"])
    xc_np = xc_grid.squeeze(-1).numpy()

    # Pre-sample yc once per source so all variants/targets see the same draws,
    # making the cross-condition comparison apples-to-apples.
    yc_by_source: dict[str, torch.Tensor] = {}
    for source in erf_cfg["y_sources"]:
        torch.manual_seed(int(cfg.misc.seed) + 7919)
        if source == "normal":
            yc_by_source[source] = _sample_yc_normal(
                erf_cfg["n_samples"], erf_cfg["n_grid"]
            )
        elif source == "task":
            print(
                f"[task gen] drawing {erf_cfg['n_samples']} samples on "
                f"N={erf_cfg['n_grid']} grid (this may take a while)…"
            )
            yc_by_source[source] = _sample_yc_task(
                output_generator, xc_grid, erf_cfg["n_samples"]
            )
        else:
            raise ValueError(f"unknown y-source: {source!r}")

    # Build both variants and add them to the dict in the user-requested
    # order. fresh_model holds the random-init copy; trained-variant
    # loading mutates experiment.model's state dict in place, which is
    # why we instantiated fresh_model from cfg.model separately above.
    models: dict[str, torch.nn.Module] = {}
    for variant in erf_cfg["variants"]:
        if variant == "random":
            m = fresh_model.to(device)
            m.eval()
            models["random"] = m
        elif variant == "trained":
            models["trained"] = _load_trained_model(experiment, device)
        else:
            raise ValueError(f"unknown variant: {variant!r}")

    # Compute ERFs.
    n_cells = (
        len(models)
        * len(erf_cfg["y_sources"])
        * len(erf_cfg["targets"])
    )
    print(
        f"Computing ERF on {n_cells} cells "
        f"({len(models)} variants × {len(erf_cfg['y_sources'])} sources × "
        f"{len(erf_cfg['targets'])} targets)",
        flush=True,
    )
    results: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    with tqdm(
        total=n_cells,
        desc="ERF cells",
        file=sys.stdout,
        mininterval=2.0,
        ascii=True,
    ) as pbar:
        for variant_name, model in models.items():
            results[variant_name] = {}
            for source in erf_cfg["y_sources"]:
                results[variant_name][source] = {}
                yc_samples = yc_by_source[source]
                for xt in erf_cfg["targets"]:
                    pbar.set_postfix_str(
                        f"{variant_name}|{source}|xt={xt:+.1f}"
                    )
                    erf = _erf_at_target(
                        model=model,
                        xc_grid=xc_grid,
                        yc_samples=yc_samples,
                        x_target=float(xt),
                        device=device,
                        batch_size=erf_cfg["batch_size"],
                        target=erf_cfg["target"],
                    )
                    results[variant_name][source][float(xt)] = erf
                    radius = _two_sigma_radius(erf, xc_np, float(xt))
                    tqdm.write(
                        f"  [{variant_name}|{source}|xt={xt:+.1f}] "
                        f"2σ-radius={radius:.3f}, peak={erf.max():.3e}",
                        file=sys.stdout,
                    )
                    sys.stdout.flush()
                    pbar.update(1)

    task_name = str(cfg.misc.experiment_name)
    model_name = str(cfg.model_name)
    _plot_erf(
        results=results,
        xc=xc_np,
        targets=erf_cfg["targets"],
        training_domain=erf_cfg["training_domain"],
        output_path=erf_cfg["output"],
        model_name=model_name,
        task_name=task_name,
        target=erf_cfg["target"],
    )

    # Persist machine-readable outputs next to erf.pdf so downstream
    # cross-job analysis doesn't have to parse PDFs or SLURM stdout.
    output_dir = Path(erf_cfg["output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_json(output_dir, cfg, erf_cfg, results, xc_np)
    _write_curves_npz(output_dir, xc_np, erf_cfg["targets"], results)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Thin wrapper around run_erf."""
    torch.set_float32_matmul_precision("high")
    run_erf(cfg)


if __name__ == "__main__":
    main()
