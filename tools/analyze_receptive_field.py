#!/usr/bin/env python
"""
Effective spatial kernel visualisation for SFConvCNP.

Loads a trained Lightning checkpoint, extracts the learnable complex-valued
spectral weights from every Set Fourier Convolution (SFConv) layer, and
reconstructs the corresponding effective spatial kernel via the inverse
Fourier transform.  The resulting plot demonstrates that the learned kernels
have approximately *global* spatial support, in contrast to the localized
kernels of standard CNNs.

Theory
------
Each SFConv layer parameterises a convolution kernel κ through its Fourier
transform.  For a 1-D model the positive-frequency weights W_ξ ∈ ℂ are
stored for ξ ∈ {0, Δf, 2Δf, …, N_f · Δf}.  The spatial kernel is recovered
via the real-valued inverse Fourier transform:

    κ(x) = Δf · W_0 + 2Δf · Σ_{k>0} Re[ W_{ξ_k} · exp(i·2π·ξ_k·x) ]

Because W_ξ is non-zero across *all* frequencies (including the lowest ones),
κ(x) has non-negligible magnitude well beyond the local neighbourhood of any
given point, i.e. it has global support.

Usage
-----
    python tools/analyze_receptive_field.py \\
        --checkpoint  runs/sfconvcnp_synthetic/seed0/best.ckpt \\
        --output      figures/receptive_field.pdf \\
        [--x-range    -8 8]   \\
        [--n-points   2000]   \\
        [--layer-idx  0 1 2]  \\
        [--no-freq-plot]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_state_dict(ckpt_path: str) -> dict:
    """Load a PyTorch Lightning checkpoint and return its state dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Lightning wraps the model under the key "state_dict"
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    # Fallback: plain PyTorch checkpoint
    return ckpt


def find_sfconv_layers(state_dict: dict) -> dict[str, dict[str, torch.Tensor]]:
    """
    Scan the state dict and group tensors by SFConv layer.

    Returns a dict mapping a canonical layer name (e.g.
    "context_encoder_blocks.0.setfourierconv") to a sub-dict with the keys
    "weights" (or "U"/"V" for low-rank), "pos_half_freq_grid", and
    "freq_volume_element".
    """
    # Identify all keys that contain SFConv weights/buffers
    weight_keys = [k for k in state_dict if k.endswith(".weights") and
                   "setfourierconv" in k]
    low_rank_u_keys = [k for k in state_dict if k.endswith(".U") and
                       "setfourierconv" in k]

    # Use whichever parameterisation is present
    if not weight_keys and not low_rank_u_keys:
        raise RuntimeError(
            "No SFConv weight tensors found in checkpoint.  "
            "Make sure the checkpoint is from an SFConvCNP model."
        )

    anchor_keys = weight_keys if weight_keys else low_rank_u_keys

    layers = {}
    for key in sorted(anchor_keys):
        # Derive canonical prefix, e.g.
        # "model.encoder.set_encoder.context_encoder_blocks.2.setfourierconv"
        suffix = ".weights" if key in weight_keys else ".U"
        prefix = key[: -len(suffix)]

        freq_grid_key = prefix + ".pos_half_freq_grid"
        vol_key = prefix + ".freq_volume_element"

        if freq_grid_key not in state_dict:
            print(f"[warn] skipping {prefix}: freq grid buffer not found")
            continue

        entry: dict[str, torch.Tensor] = {
            "pos_half_freq_grid": state_dict[freq_grid_key],
            "freq_volume_element": state_dict.get(vol_key, torch.tensor(1.0)),
        }

        if suffix == ".weights":
            entry["weights"] = state_dict[key]
        else:
            entry["U"] = state_dict[key]
            v_key = prefix + ".V"
            if v_key in state_dict:
                entry["V"] = state_dict[v_key]

        layers[prefix] = entry

    return layers


def effective_kernel_weights(entry: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Return a representative complex weight vector W_ξ of shape [n_freq].

    For a full-rank layer (shape [quads, groups, in, out, n_freq]), we average
    the complex weights over quadrants, groups, and channel pairs so that the
    result summarises the overall spectral distribution.

    For low-rank layers (U, V), we first reconstruct W = U @ V along the
    channel axes and then average identically.
    """
    if "weights" in entry:
        W = entry["weights"]  # [quads, groups, in, out, *freq_dims]
    else:
        # Low-rank: W[..., freq] = sum_r U[..., r, freq] * V[..., r, freq]
        U = entry["U"]  # [quads, groups, in, rank, *freq_dims]
        V = entry["V"]  # [quads, groups, rank, out, *freq_dims]
        # Contract over rank dimension
        W = torch.einsum("qgir..., qgro... -> qgio...", U, V)

    # Average complex magnitude over (quads, groups, in, out), keep freq dims
    # Use mean of the complex tensor directly so phase information is preserved
    # in the frequency → spatial direction.
    # Shape after mean: [*freq_dims]
    ndim_prefix = 4  # quads, groups, in, out
    W_mean = W.mean(dim=list(range(ndim_prefix)))
    return W_mean  # complex, shape [*freq_dims]


# ---------------------------------------------------------------------------
# Spatial kernel reconstruction (1-D)
# ---------------------------------------------------------------------------

def spatial_kernel_1d(
    W_mean: torch.Tensor,
    freq_grid: torch.Tensor,
    freq_vol: float,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the 1-D effective spatial kernel κ(x) from spectral weights.

    Mirrors the _inverse_fourier logic in SetFourierConvBase:

        κ(x) = freq_vol · W_0 + 2·freq_vol · Σ_{k>0} Re[W_k · exp(i2πξ_k x)]

    Parameters
    ----------
    W_mean   : complex tensor of shape [n_freq]  (averaged spectral weights)
    freq_grid: float tensor of shape [n_freq, 1] (positive frequencies)
    freq_vol : scalar (frequency volume element = Δf for 1-D)
    x        : float tensor of shape [n_x]       (spatial query points)

    Returns
    -------
    kappa : real tensor of shape [n_x]
    """
    W = W_mean  # [n_freq], complex
    xi = freq_grid.squeeze(-1)  # [n_freq], real positive

    # phases: [n_x, n_freq]
    phases = 2.0 * torch.pi * torch.outer(x, xi)  # real
    exp_phases = torch.cos(phases) + 1j * torch.sin(phases)  # [n_x, n_freq]

    # IFT with Hermitian symmetry: DC counted once, rest doubled
    contribution = exp_phases * W.unsqueeze(0)  # [n_x, n_freq]

    # DC (ξ=0) weight is halved before doubling ↔ counted once
    kappa = 2.0 * freq_vol * contribution.real  # [n_x, n_freq]
    kappa[:, 0] *= 0.5
    kappa = kappa.sum(dim=-1)  # [n_x]

    return kappa


def spatial_kernel_nd(
    W_mean: torch.Tensor,
    freq_grid: torch.Tensor,
    freq_vol: float,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the N-D effective spatial kernel along the first spatial axis
    by marginalising over other dimensions (setting them to zero).

    For a d-D model (d > 1) this evaluates κ(x, 0, …, 0) so that the
    result is still a 1-D slice suitable for plotting.
    """
    n_freq_dims = freq_grid.shape[:-1]  # e.g. (n1, n2) for 2-D
    d = freq_grid.shape[-1]

    # Flatten W_mean to [n_total_freq]
    W_flat = W_mean.reshape(-1)

    # Build query positions along first axis only
    x_query = torch.zeros(len(x), d, dtype=x.dtype)
    x_query[:, 0] = x

    # phases: [n_x, n_total_freq]
    xi_flat = freq_grid.reshape(-1, d)  # [n_total_freq, d]
    phases = 2.0 * torch.pi * (x_query @ xi_flat.T)  # [n_x, n_total_freq]
    exp_phases = torch.cos(phases) + 1j * torch.sin(phases)

    contribution = exp_phases * W_flat.unsqueeze(0)

    kappa = 2.0 * freq_vol * contribution.real
    kappa[:, 0] *= 0.5
    kappa = kappa.sum(dim=-1)

    return kappa


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LAYER_COLORS = matplotlib.colormaps["viridis"](np.linspace(0.15, 0.85, 12))


def make_short_label(prefix: str, idx: int) -> str:
    """Return 'Layer i' with context/query tag."""
    if "context" in prefix:
        return f"Layer {idx + 1} (ctx)"
    return f"Layer {idx + 1}"


def plot_receptive_field(
    layers: dict[str, dict],
    layer_indices: list[int] | None,
    x_range: tuple[float, float],
    n_points: int,
    show_freq: bool,
    output_path: str | None,
    training_domain: tuple[float, float] = (-3.0, 3.0),
) -> None:
    """Main plotting routine."""
    x = torch.linspace(x_range[0], x_range[1], n_points)

    # Filter layer list
    all_prefixes = sorted(layers.keys())
    # Keep only context encoder (query is a copy)
    ctx_prefixes = [p for p in all_prefixes if "context" in p or
                    ("query" not in p and "context" not in p)]
    if not ctx_prefixes:
        ctx_prefixes = all_prefixes

    if layer_indices is not None:
        ctx_prefixes = [p for i, p in enumerate(ctx_prefixes) if i in layer_indices]

    n_layers = len(ctx_prefixes)
    if n_layers == 0:
        print("[error] No layers matched the requested indices.")
        sys.exit(1)

    # Detect spatial dimension
    sample_entry = layers[ctx_prefixes[0]]
    freq_grid = sample_entry["pos_half_freq_grid"]
    spatial_dim = freq_grid.shape[-1]

    n_cols = 2 if show_freq else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.0))
    if n_cols == 1:
        axes = [axes]

    ax_spatial = axes[0]
    ax_freq = axes[1] if show_freq else None

    # ── Spatial kernel ────────────────────────────────────────────────────
    ax_spatial.set_title("Effective spatial kernel $|\\kappa(x)|$")
    ax_spatial.set_xlabel("Spatial offset $x$")
    ax_spatial.set_ylabel("Kernel magnitude (a.u.)")

    # Shade training domain
    ax_spatial.axvspan(
        training_domain[0], training_domain[1],
        alpha=0.08, color="steelblue", label="Training domain"
    )
    ax_spatial.axvline(0, color="gray", lw=0.6, ls="--")

    kappa_list = []
    for i, prefix in enumerate(ctx_prefixes):
        entry = layers[prefix]
        freq_vol = float(entry["freq_volume_element"])
        W_mean = effective_kernel_weights(entry)
        freq_g = entry["pos_half_freq_grid"]

        if spatial_dim == 1:
            kappa = spatial_kernel_1d(W_mean, freq_g, freq_vol, x)
        else:
            kappa = spatial_kernel_nd(W_mean, freq_g, freq_vol, x)

        kappa_np = kappa.numpy()
        kappa_abs = np.abs(kappa_np)
        kappa_list.append(kappa_abs)

        label = make_short_label(prefix, i)
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        ax_spatial.plot(x.numpy(), kappa_abs, label=label, color=color, lw=1.6)

    # Mark the decay at the domain edge for visual clarity
    x_np = x.numpy()
    domain_edge = training_domain[1]
    ax_spatial.axvline(domain_edge, color="steelblue", lw=0.8, ls=":", alpha=0.7)
    ax_spatial.axvline(-domain_edge, color="steelblue", lw=0.8, ls=":", alpha=0.7)

    ax_spatial.legend(loc="upper right", framealpha=0.8)
    ax_spatial.set_xlim(x_range)

    # Compute and print the fraction of energy outside the training domain
    domain_mask = (x_np >= training_domain[0]) & (x_np <= training_domain[1])
    for i, (kappa_abs, prefix) in enumerate(zip(kappa_list, ctx_prefixes)):
        energy_total = np.sum(kappa_abs ** 2)
        energy_outside = np.sum(kappa_abs[~domain_mask] ** 2)
        frac = energy_outside / (energy_total + 1e-12)
        print(f"  {make_short_label(prefix, i)}: "
              f"{100*frac:.1f}% of kernel energy outside training domain")

    # ── Frequency spectrum ────────────────────────────────────────────────
    if ax_freq is not None:
        ax_freq.set_title("Learned spectral weights $|W_\\xi|$")
        ax_freq.set_xlabel("Frequency $\\xi$")
        ax_freq.set_ylabel("Mean weight magnitude (a.u.)")

        for i, prefix in enumerate(ctx_prefixes):
            entry = layers[prefix]
            W_mean = effective_kernel_weights(entry)
            freq_g = entry["pos_half_freq_grid"]

            # 1-D slice along first frequency axis
            xi = freq_g.reshape(-1, freq_grid.shape[-1])[:, 0].numpy()
            W_mag = W_mean.abs().reshape(-1).numpy()

            # Deduplicate / take first freq dimension for multi-D
            if spatial_dim > 1:
                n1 = int(round(len(xi) ** (1 / spatial_dim)))
                xi = xi[:n1]
                W_mag = W_mag[:n1]

            label = make_short_label(prefix, i)
            color = LAYER_COLORS[i % len(LAYER_COLORS)]
            ax_freq.plot(xi, W_mag, label=label, color=color, lw=1.6)

        ax_freq.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a trained Lightning checkpoint (.ckpt).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path for the figure (e.g. figures/rf.pdf). "
             "If omitted, the plot is shown interactively.",
    )
    p.add_argument(
        "--x-range", nargs=2, type=float, default=[-8.0, 8.0],
        metavar=("X_MIN", "X_MAX"),
        help="Spatial range for kernel evaluation (default: -8 to 8).",
    )
    p.add_argument(
        "--training-domain", nargs=2, type=float, default=[-3.0, 3.0],
        metavar=("DOM_MIN", "DOM_MAX"),
        help="Training input domain, shaded in the figure (default: -3 to 3).",
    )
    p.add_argument(
        "--n-points", type=int, default=2000,
        help="Number of spatial evaluation points (default: 2000).",
    )
    p.add_argument(
        "--layer-idx", nargs="+", type=int, default=None,
        metavar="IDX",
        help="Zero-based layer indices to plot (default: all context layers).",
    )
    p.add_argument(
        "--no-freq-plot", action="store_true",
        help="Suppress the frequency-spectrum subplot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = load_state_dict(str(ckpt_path))

    print("Scanning for SFConv layers …")
    layers = find_sfconv_layers(state_dict)
    print(f"Found {len(layers)} SFConv layers:")
    for prefix in sorted(layers):
        W = effective_kernel_weights(layers[prefix])
        print(f"  {prefix}  →  freq shape {W.shape}")

    plot_receptive_field(
        layers=layers,
        layer_indices=args.layer_idx,
        x_range=tuple(args.x_range),
        n_points=args.n_points,
        show_freq=not args.no_freq_plot,
        output_path=args.output,
        training_domain=tuple(args.training_domain),
    )


if __name__ == "__main__":
    main()
