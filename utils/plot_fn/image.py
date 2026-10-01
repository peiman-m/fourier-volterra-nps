import copy
from pathlib import Path

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from nps.models.base import BaseNeuralProcess

from ..data import ImageBatch
from .base import BaseNeuralProcessPlotter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class ImagePlotter(BaseNeuralProcessPlotter):
    """
    A class to handle plotting of Neural Process model predictions for
    image data.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (16.0, 8.0),
        savefig: bool = False,
        logging: bool = True,
        plot_dir: str = "fig",
        show_plots: bool = False,
    ) -> None:
        """Initialize the ImagePlotter.

        Args:
            figsize: Figure size for the plots
            savefig: Whether to save the figure to disk
            logging: Whether to log the figure to wandb
            plot_dir: Directory to save figures to
        """
        self.figsize = figsize
        self.savefig = savefig
        self.logging = logging
        self.plot_dir = Path(plot_dir)
        self.show_plots = show_plots

    def _prepare_batch(self, batch: ImageBatch) -> ImageBatch:
        """Prepare batch for plotting by taking only the first item.

        Args:
            batch: Input batch

        Returns:
            Prepared batch
        """
        batch_copy = copy.deepcopy(batch)

        # Keep only the first item in batch
        for key, value in vars(batch_copy).items():
            if isinstance(value, torch.Tensor):
                setattr(batch_copy, key, value[:1])

        return batch_copy

    def _plot_with_subplots(
        self,
        yc: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        context_prop: float,
        model_ll: float,
        fig_path: Path,
    ) -> None:
        """Create a figure with subplots for context, ground truth, mean, and std.

        Args:
            yc: Context data (masked)
            y: Ground truth data
            mean: Mean predictions
            std: Standard deviation predictions
            context_prop: Proportion of context points
            model_ll: Model log-likelihood
            fig_path: Path to save the figure
        """
        fig, axes = plt.subplots(
            figsize=self.figsize, ncols=4, nrows=1, constrained_layout=True
        )

        axes[0].imshow(y, vmax=1, vmin=0)
        axes[1].imshow(yc, vmax=1, vmin=0)
        axes[2].imshow(mean, vmax=1, vmin=0)
        im = axes[3].imshow(std, cmap="magma", vmin=std.min(), vmax=std.max())

        # Attach colorbar outside the subplots to avoid resizing the std image
        cbar = fig.colorbar(
            im,
            ax=axes[3],
            location="right",
            fraction=0.1,  # width relative to the axes
            pad=0.05,  # space between axes and colorbar
            # shrink=0.6,       # shrink length of the colorbar
            aspect=10,  # aspect ratio (length / width)
        )
        # cbar.set_label('Uncertainty (Std Dev)', fontsize=10)

        # Remove ticks and axis labels for all subplots
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axes[0].set_title("Ground truth", fontsize=12)
        axes[1].set_title("Context set", fontsize=12)
        axes[2].set_title("Prediction mean", fontsize=12)
        axes[3].set_title("Prediction std", fontsize=12)

        fig.suptitle(
            f"context/all = {context_prop:.2f}, LogP = {model_ll:.3f}", fontsize=16, y=0.96
        )

        self._handle_figure_output(
            fig,
            fig_path,
            savefig=self.savefig,
            logging=self.logging,
            show_plots=self.show_plots,
        )

        plt.close(fig)

    def __call__(
        self,
        model: BaseNeuralProcess,
        batches: list[ImageBatch],
        name: str = "plot",
        **kwargs,
    ) -> None:
        """
        Generate and display/save plots for given batches.

        Args:
            model (BaseNeuralProcess): The model used for predictions.
            batches (list[ImageBatch]): A list of data batches.
            name (str): Name for saving figures.
            **kwargs: Additional arguments to be passed to the model.
        """
        fig_dir = self.plot_dir / name
        if self.savefig:
            fig_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(batches):
            # Prepare batch for plotting
            eval_batch = self._prepare_batch(batch)

            # Handle full prediction vs context points
            plot_batch = copy.deepcopy(eval_batch)

            # Set all points as query for prediction
            plot_batch.xq = eval_batch.x
            plot_batch.mq_grid = torch.ones_like(eval_batch.mq_grid)

            # Get appropriate forward wrapper for this model and batch
            forward_wrapper = self._get_forward_wrapper(model, batch)

            # Compute model predictions
            with torch.no_grad():
                if forward_wrapper:
                    model_output = forward_wrapper(model, eval_batch)
                    model_output_plot = forward_wrapper(model, plot_batch)
                else:
                    raise RuntimeError(
                        "No forward wrapper found for model's forward call."
                    )

                model_ll = model_output.log_prob(eval_batch.yq).mean()

            # Extract distributional predictions
            pred_dist_plot = model_output_plot
            mean = pred_dist_plot.mean.cpu()
            std = pred_dist_plot.stddev.cpu()

            # Get context mask and proportion
            mc_grid = plot_batch.mc_grid.detach().cpu()
            y_grid = plot_batch.y_grid.detach().cpu()
            channels, height, width = y_grid.shape[1:]

            if channels == 1:
                ch_multiple = 3
            elif channels == 3:
                ch_multiple = 1
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")

            # Calculate proportion based on first image in plot_batch and first channel
            context_proportion = plot_batch.mc_grid[0, 0].sum() / (
                plot_batch.mc_grid[0, 0].numel()
            )

            mc_grid, y_grid = mc_grid[0], y_grid[0]
            mean, std = mean[0], std[0]

            y_grid = einops.repeat(
                y_grid, "c h w -> h w (c m)", m=ch_multiple
            )  # imshow input format
            mc_grid = einops.repeat(mc_grid, "c h w -> h w (c m)", m=ch_multiple)
            mean = einops.repeat(
                mean, "(h w) c -> h w (c m)", h=height, w=width, m=ch_multiple
            )
            std = einops.repeat(std, "(h w) c -> h w c", h=height, w=width)

            aggregated_std = std.mean(dim=-1, keepdim=True)

            if channels == 1:
                blue_pixels = torch.zeros_like(y_grid)
                blue_pixels[:, :, -1] = 1.0
                yc_grid = torch.where(mc_grid > 0, y_grid, blue_pixels)
            elif channels == 3:
                black_pixels = torch.zeros_like(y_grid)
                yc_grid = torch.where(mc_grid > 0, y_grid, black_pixels)
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")

            # Create plots
            fname = fig_dir / f"{i:03d}"
            self._plot_with_subplots(
                yc_grid.cpu().numpy(),
                y_grid.cpu().numpy(),
                mean.cpu().numpy(),
                aggregated_std.cpu().numpy(),
                float(context_proportion),
                float(model_ll),
                fname,
            )
