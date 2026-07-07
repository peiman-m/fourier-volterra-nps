import copy
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from nps.models.base import BaseNeuralProcess

from ..data import SyntheticBatch
from .base import BaseNeuralProcessPlotter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class SyntheticPlotter(BaseNeuralProcessPlotter):
    """
    A class to handle plotting of Neural Process model predictions for
    synthetic data. Currently only supports 1D input and 1D output.
    """

    def __init__(
        self,
        xc_range_eval: tuple[float, float] | tuple[tuple[float, float], ...],
        xq_range_eval: tuple[float, float] | tuple[tuple[float, float], ...],
        xc_range_train: (
            tuple[float, float] | tuple[tuple[float, float], ...] | None
        ) = None,
        xq_range_train: (
            tuple[float, float] | tuple[tuple[float, float], ...] | None
        ) = None,
        figsize: tuple[float, float] = (6.5, 4.0),
        y_lim: tuple[float, float] = (-2.0, 2.0),
        points_per_unit: int = 64,
        savefig: bool = False,
        logging: bool = True,
        show_plots: bool = True,
        plot_dir: str = "fig",
    ) -> None:
        """Initialize the SyntheticPlotter."""
        # Convert input ranges to NumPy arrays and validate
        self.x_range_eval = self._compute_x_range(xc_range_eval, xq_range_eval)

        if xc_range_train is not None and xq_range_train is not None:
            self.x_range_train = self._compute_x_range(xc_range_train, xq_range_train)
        else:
            self.x_range_train = None

        # self.x_range_train = (
        #     self._compute_x_range(xc_range_train, xq_range_train)
        #     if (xc_range_train and xq_range_train)
        #     else None
        # )

        # Other configurations
        self.figsize = figsize
        self.y_lim = y_lim
        self.points_per_unit = points_per_unit
        self.savefig = savefig
        self.logging = logging
        self.plot_dir = Path(plot_dir)
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
        xc_range = np.squeeze(SyntheticPlotter._to_numpy(xc_range)).astype(float)
        xq_range = np.squeeze(SyntheticPlotter._to_numpy(xq_range)).astype(float)

        assert (
            xc_range.ndim == 1 and xq_range.ndim == 1
        ), "Currently only supports 1D input and 1D output."

        return np.array([min(xc_range[0], xq_range[0]), max(xc_range[1], xq_range[1])])

    @staticmethod
    def _sort(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorts input tensor pairs along the first dimension."""
        sorted_indices = torch.argsort(x, dim=1)
        return (
            torch.gather(x, dim=1, index=sorted_indices),
            torch.gather(y, dim=1, index=sorted_indices),
        )

    @staticmethod
    def _plot(
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_std: torch.Tensor,
        alpha: float = 0.2,
        label: str | None = None,
    ) -> None:
        """Plots the mean and uncertainty bounds for a model's prediction."""
        x = x[0, :, 0].detach().cpu()
        y_pred = y_pred[0, :, 0].detach().cpu()
        y_std = y_std[0, :, 0].detach().cpu()

        plt.plot(x, y_pred, c="tab:blue", lw=2, zorder=10)
        plt.fill_between(
            x,
            y_pred - 2.0 * y_std,
            y_pred + 2.0 * y_std,
            color="tab:blue",
            alpha=alpha,
            label=label,
        )

    def _plot_predictions(
        self,
        x_plot: torch.Tensor,
        pred_dist_plot: torch.distributions.Distribution,
    ) -> None:
        """Handles plotting of model predictions."""
        self._plot(
            x_plot,
            pred_dist_plot.mean,
            pred_dist_plot.stddev,
            alpha=0.2,
            label="Model",
        )

    def _plot_ground_truth(
        self,
        batch: Any,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xq: torch.Tensor,
        yq: torch.Tensor,
        x_plot: torch.Tensor,
    ) -> torch.Tensor | None:
        """Handles ground truth comparison in plots."""
        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            with torch.no_grad():
                gt_mean, gt_std, _ = batch.gt_pred(xc=xc, yc=yc, xq=x_plot)
                _, _, gt_loglik = batch.gt_pred(xc=xc, yc=yc, xq=xq, yq=yq)

            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu(),
                "--",
                color="tab:purple",
                lw=1.5,
                label="Ground truth",
            )
            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                "--",
                color="tab:purple",
                lw=1.5,
            )
            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                "--",
                color="tab:purple",
                lw=1.5,
            )
            return gt_loglik.mean().item()
        return None

    def __call__(
        self,
        model: BaseNeuralProcess,
        batches: list[SyntheticBatch],
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
        # Create a subfolder with the name parameter under the plot_dir
        fig_dir = self.plot_dir / name
        if self.savefig:
            fig_dir.mkdir(parents=True, exist_ok=True)

        x_plot = torch.arange(
            start=self.x_range_eval[0],
            end=self.x_range_eval[1],
            step=1 / self.points_per_unit,
        ).to(batches[0].xc)[None, :, None]

        for i, batch in enumerate(batches):
            # Ensure sorted input data
            xc, yc = self._sort(batch.xc[:1], batch.yc[:1])
            xq, yq = self._sort(batch.xq[:1], batch.yq[:1])

            batch.xc = xc
            batch.yc = yc
            batch.xq = xq
            batch.yq = yq

            plot_batch = copy.deepcopy(batch)
            plot_batch.xq = x_plot

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

            # Create figure
            fig = plt.figure(figsize=self.figsize)

            # Plot context points
            plt.scatter(
                xc[0, :, 0].detach().cpu(),
                yc[0, :, 0].detach().cpu(),
                c="k",
                label="Context",
                s=30,
                zorder=20,
            )

            # Plot query function
            plt.scatter(
                xq[0, :, 0].detach().cpu(),
                yq[0, :, 0].detach().cpu(),
                c="r",
                label="Query",
                s=10,
                zorder=0,
            )

            # Add grey background for train range
            if self.x_range_train is not None:
                plt.axvspan(
                    self.x_range_train[0],
                    self.x_range_train[1],
                    color="grey",
                    alpha=0.15,
                )

            # Plot model predictions
            self._plot_predictions(x_plot, pred_dist_plot)
            gt_loglik = self._plot_ground_truth(batch, xc, yc, xq, yq, x_plot)

            title_str = f"$N = {xc.shape[1]}$, LogP = {model_loglik:.3f}"
            if gt_loglik is not None:
                title_str += f", GT LogP = {gt_loglik:.3f}"

            plt.title(title_str, fontsize=15)
            plt.grid()
            plt.xlim(self.x_range_eval)
            plt.ylim(self.y_lim)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc="upper right", fontsize=10)
            plt.tight_layout()

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
