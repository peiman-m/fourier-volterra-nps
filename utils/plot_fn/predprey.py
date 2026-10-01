import copy
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from nps.models.base import BaseNeuralProcess

from ..data import Batch, PredPreyBatch
from .base import BaseNeuralProcessPlotter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class PredPreyPlotter(BaseNeuralProcessPlotter):
    """
    A class to handle plotting of Neural Process model predictions for
    predator-prey dynamics data. This plotter is specifically designed for
    the PredPreyBatch type, which has 2-dimensional outputs representing
    prey and predator populations.
    """

    def __init__(
        self,
        xc_range_eval: tuple[float, float],
        xq_range_eval: tuple[float, float],
        xc_range_train: tuple[float, float] | None = None,
        xq_range_train: tuple[float, float] | None = None,
        y_range: tuple[float, float] | tuple[tuple[float, float], ...] | None = None,
        figsize: tuple[float, float] = (12.0, 8.0),
        points_per_unit: int = 1,
        savefig: bool = False,
        logging: bool = True,
        plot_dir: str = "fig",
        legend_fontsize: int = 12,
        dim_labels: list[str] | None = None,
        plot_mode: str = "dense",
        show_plots: bool = False,
    ) -> None:
        """Initialize the PredPreyPlotter.

        Args:
            xc_range_eval: The range of context data for evaluation
            xq_range_eval: The range of query data for evaluation
            xc_range_train: The range of context data for training (optional)
            xq_range_train: The range of query data for training (optional)
            y_range: Y-axis limits. Can be a single tuple for all dimensions or
                     list of tuples per dimension (optional)
            figsize: Base figure size
            points_per_unit: Number of points per unit for input dimension for plotting
            savefig: Whether to save the figure to disk
            logging: Whether to log the figure to wandb
            plot_dir: Directory to save figures to
            legend_fontsize: Font size for the legend
            dim_labels: Custom labels for the two dimensions. Defaults to ["Prey", "Predator"]
            plot_mode: Mode of plotting, either "dense" to plot the dense trajectories
                      or "points" to plot only the sampled points
        """
        # Convert input ranges to NumPy arrays and validate
        self.x_range_eval = self._compute_x_range(xc_range_eval, xq_range_eval)

        self.x_range_train = (
            self._compute_x_range(xc_range_train, xq_range_train)
            if xc_range_train is not None and xq_range_train is not None
            else None
        )

        # Hydra's ``_convert_="all"`` hands YAML sequences in as plain
        # ``list``; the downstream ``__call__`` branches on
        # ``isinstance(self.y_range, tuple)`` so promote to tuple
        # (recursively, since ``y_range`` can be ``tuple[tuple, tuple]``).
        if isinstance(y_range, list):
            y_range = cast(
                Any,
                tuple(
                    tuple(item) if isinstance(item, list) else item for item in y_range
                ),
            )

        # Other configurations
        self.y_range = y_range
        self.figsize = figsize
        self.points_per_unit = points_per_unit
        self.savefig = savefig
        self.logging = logging
        self.plot_dir = Path(plot_dir)
        self.legend_fontsize = legend_fontsize

        # Use default dimension labels if not provided
        self.dim_labels = dim_labels or ["Prey", "Predator"]

        # Validate plot_mode
        if plot_mode not in ["dense", "points"]:
            raise ValueError("plot_mode must be either 'dense' or 'points'")
        self.plot_mode = plot_mode

        self.show_plots = show_plots

    @staticmethod
    def _to_numpy(range_: tuple | torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert range tuple or tensor to NumPy array if needed."""
        if isinstance(range_, np.ndarray):
            return range_
        if isinstance(range_, torch.Tensor):
            return range_.detach().cpu().numpy()
        return np.array(range_, dtype=float)

    @staticmethod
    def _compute_x_range(xc_range, xq_range) -> np.ndarray:
        """Compute the combined x range from context and query ranges."""
        xc_range = np.squeeze(PredPreyPlotter._to_numpy(xc_range)).astype(float)
        xq_range = np.squeeze(PredPreyPlotter._to_numpy(xq_range)).astype(float)

        assert (
            xc_range.ndim == 1 and xq_range.ndim == 1
        ), "Currently only supports 1D input and 2D output for predator-prey data."

        return np.array([min(xc_range[0], xq_range[0]), max(xc_range[1], xq_range[1])])

    @staticmethod
    def _sort(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorts input tensor pairs along the first dimension."""
        sorted_indices = torch.argsort(x, dim=1)
        return (
            torch.gather(x, dim=1, index=sorted_indices),
            torch.gather(
                y,
                dim=1,
                index=(
                    sorted_indices.repeat(1, 1, y.shape[2])
                    if y.shape[2] > 1
                    else sorted_indices
                ),
            ),
        )

    @staticmethod
    def _plot_line(
        ax: plt.Axes,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_std: torch.Tensor,
        alpha: float = 0.2,
        label: str | None = None,
        color: str = "tab:blue",
    ) -> plt.Line2D:
        """Plots the mean and uncertainty bounds for a model's prediction.

        Args:
            ax: The matplotlib axes to plot on
            x: Input tensor of shape [batch_size, num_points, input_dim]
            y_pred: Predicted means of shape [batch_size, num_points, 1]
            y_std: Predicted standard deviations of shape [batch_size, num_points, 1]
            alpha: Transparency for uncertainty bounds
            label: Label for the plot legend
            color: Color for the plot

        Returns:
            The Line2D object for the mean line (for legend)
        """
        x_np = x[0, :, 0].detach().cpu().numpy()
        y_pred_np = y_pred[0, :].detach().cpu().numpy()
        y_std_np = y_std[0, :].detach().cpu().numpy()

        line = ax.plot(x_np, y_pred_np, c=color, lw=2, zorder=10, label=label)[0]
        ax.fill_between(
            x_np,
            y_pred_np - 2.0 * y_std_np,
            y_pred_np + 2.0 * y_std_np,
            color=color,
            alpha=alpha,
        )

        return line

    def _plot_dense_trajectory(
        self,
        ax: plt.Axes,
        x_dense: torch.Tensor,
        y_dense: torch.Tensor,
        channel_idx: int,
        color: str = "gray",
        alpha: float = 0.3,
        linewidth: float = 1.0,
        label: str | None = None,
    ) -> plt.Line2D:
        """Plots the dense trajectory for a specific channel.

        Args:
            ax: The matplotlib axes to plot on
            x_dense: Dense input times tensor [batch_size, num_points, 1]
            y_dense: Dense output values tensor [batch_size, num_points, num_channels]
            channel_idx: Index of the channel to plot (0 for prey, 1 for predator)
            color: Color for the trajectory line
            alpha: Transparency for the line
            linewidth: Width of the line
            label: Label for the plot legend

        Returns:
            The Line2D object for the trajectory
        """
        x = x_dense[0, :, 0].detach().cpu().numpy()
        y = y_dense[0, :, channel_idx].detach().cpu().numpy()

        # Use line plot for better performance with dense data
        line = ax.plot(x, y, c=color, lw=linewidth, alpha=alpha, label=label)[0]

        return line

    def _plot_predictions(
        self,
        ax: plt.Axes,
        x_plot: torch.Tensor,
        pred_dist_plot: torch.distributions.Distribution,
        channel_idx: int,
        color: str = "tab:blue",
    ) -> list[plt.Line2D]:
        """Handles plotting of model predictions.

        Args:
            ax: The matplotlib axes to plot on
            x_plot: Input tensor for plotting
            pred_dist_plot: Prediction distribution
            channel_idx: Index of the output channel to plot (0 for prey, 1 for predator)
            color: Color for the plot

        Returns:
            List of Line2D objects for the legend
        """
        lines = []

        line = self._plot_line(
            ax,
            x_plot,
            pred_dist_plot.mean[..., channel_idx],
            pred_dist_plot.stddev[..., channel_idx],
            alpha=0.2,
            label="Model",
            color=color,
        )
        lines.append(line)

        return lines

    def __call__(
        self,
        model: BaseNeuralProcess,
        batches: list[Batch],
        name: str = "plot",
        **kwargs,
    ) -> None:
        """
        Generate and display/save plots for given batches.

        Args:
            model (BaseNeuralProcess): The model used for predictions.
            batches (list[Batch]): A list of data batches.
            name (str): Name for saving figures.
            **kwargs: Additional arguments to be passed to the model.
        """
        # Check if batches are PredPreyBatch instances
        for batch in batches:
            if not isinstance(batch, PredPreyBatch):
                raise TypeError(f"Expected PredPreyBatch, got {type(batch).__name__}")

        fig_dir = self.plot_dir / name
        if self.savefig:
            fig_dir.mkdir(parents=True, exist_ok=True)

        steps = int(
            self.points_per_unit * (self.x_range_eval[1] - self.x_range_eval[0])
        )

        x_plot = torch.linspace(self.x_range_eval[0], self.x_range_eval[1], steps).to(
            batches[0].xc
        )[None, :, None]

        for i, batch in enumerate(batches):
            # Create figure with two subplots (prey and predator)
            fig, axs = plt.subplots(
                2, 1, figsize=self.figsize, sharex=True, gridspec_kw={"hspace": 0.08}
            )

            # Ensure sorted input data
            xc, yc = self._sort(batch.xc[:1], batch.yc[:1])
            xq, yq = self._sort(batch.xq[:1], batch.yq[:1])

            batch.xc = xc
            batch.yc = yc
            batch.xq = xq
            batch.yq = yq

            # Get appropriate forward wrapper for this model and batch
            forward_wrapper = self._get_forward_wrapper(model, batch)

            # Create batch objects for forward pass
            eval_batch = copy.deepcopy(batch)  # For actual evaluation
            plot_batch = copy.deepcopy(batch)  # For plotting
            plot_batch.xq = x_plot

            # Compute model predictions
            with torch.no_grad():
                if forward_wrapper:
                    model_output = forward_wrapper(model, eval_batch)
                    model_output_plot = forward_wrapper(model, plot_batch)
                else:
                    raise RuntimeError(
                        "No forward wrapper found for model's forward call."
                    )

            model_loglik = model_output.log_prob(batch.yq).mean()

            pred_dist_plot = model_output_plot

            # Common title for the figure
            # fig.suptitle(f"$N = {xc.shape[1]}$, LogP = {model_loglik:.3f}", fontsize=20)

            # Process y-ranges if provided
            if self.y_range is not None:
                if (
                    isinstance(self.y_range, tuple)
                    and len(self.y_range) == 2
                    and all(isinstance(v, (int, float)) for v in self.y_range)
                ):
                    # Same y-limit for both prey and predator
                    y_ranges = [self.y_range, self.y_range]
                elif len(self.y_range) == 2 and all(
                    isinstance(item, tuple) for item in self.y_range
                ):
                    # Individual y-ranges for prey and predator
                    y_ranges = list(self.y_range)
                else:
                    raise ValueError(
                        "y_range must be a tuple of 2 floats or a tuple of 2 tuples"
                    )
            else:
                y_ranges = [None, None]

            # Legend elements
            legend_handles = []
            legend_labels = []

            # Plot for each channel (prey and predator)
            channel_colors = [
                "tab:blue",
                "tab:blue",
            ]  # Green for prey, red for predator

            for channel_idx, (ax, channel_color) in enumerate(zip(axs, channel_colors)):
                # Plot dense trajectory if in dense mode
                if (
                    self.plot_mode == "dense"
                    and hasattr(batch, "x_dense")
                    and hasattr(batch, "y_dense")
                ):
                    pp_batch = cast(PredPreyBatch, batch)
                    trajectory_line = self._plot_dense_trajectory(
                        ax,
                        pp_batch.x_dense,
                        pp_batch.y_dense,
                        channel_idx,
                        color="r",
                        linewidth=1.0,
                        alpha=0.5,
                        label="Trajectory",
                    )
                    if channel_idx == 0:  # Only add to legend once
                        legend_handles.append(trajectory_line)
                        legend_labels.append("Trajectory")

                # Plot context points
                context_scatter = ax.scatter(
                    xc[0, :, 0].detach().cpu(),
                    yc[0, :, channel_idx].detach().cpu(),
                    c="k",
                    s=20,
                    zorder=20,
                )
                if channel_idx == 0:  # Only add to legend once
                    legend_handles.append(context_scatter)
                    legend_labels.append("Context")

                # # Plot query points
                # query_scatter = ax.scatter(
                #     xq[0, :, 0].detach().cpu(), yq[0, :, channel_idx].detach().cpu(),
                #     c="blue", s=1, zorder=15, alpha=0.4,
                # )
                # if channel_idx == 0:  # Only add to legend once
                #     legend_handles.append(query_scatter)
                #     legend_labels.append("Query")

                # Add grey background for train range
                if self.x_range_train is not None:
                    ax.axvspan(
                        self.x_range_train[0],
                        self.x_range_train[1],
                        color="grey",
                        alpha=0.15,
                    )

                # Plot model predictions and collect legend handles
                lines = self._plot_predictions(
                    ax,
                    x_plot,
                    pred_dist_plot,
                    channel_idx,
                    color=channel_color,
                )

                if channel_idx == 0 and lines:  # Only add to legend once
                    legend_handles.append(lines[0])
                    legend_labels.append("Model")

                # Set axis limits and labels
                ax.set_xlim(self.x_range_eval)
                if y_ranges[channel_idx] is not None:
                    ax.set_ylim(y_ranges[channel_idx])

                # Set dimension label
                # ax.set_ylabel(self.dim_labels[channel_idx], fontsize=10)
                ax.grid(True)

            # Only show x-label for the bottom subplot
            # axs[-1].set_xlabel("Time", fontsize=14)

            # Add a common legend at the bottom
            # fig.legend(
            #     legend_handles, legend_labels,
            #     loc='lower center',
            #     bbox_to_anchor=(0.5, 0.02),
            #     ncol=len(legend_handles),
            #     fontsize=self.legend_fontsize,
            #     frameon=True,
            #     fancybox=True,
            #     shadow=True
            # )

            # Adjust layout to make space for the legend
            # plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave space for title and legend

            # Handle figure output using base class method
            fname = fig_dir / f"{i:03d}"
            self._handle_figure_output(
                fig,
                fname,
                savefig=self.savefig,
                logging=self.logging,
                show_plots=self.show_plots,
            )

            plt.close()
