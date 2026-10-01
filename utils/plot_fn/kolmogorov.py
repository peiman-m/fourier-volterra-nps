import copy
from pathlib import Path
from typing import cast

import einops
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import torch

from nps.models.base import BaseNeuralProcess

from ..data import KolmogorovBatch
from .base import BaseNeuralProcessPlotter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class KolmogorovPlotter(BaseNeuralProcessPlotter):
    """
    A class to handle plotting of Neural Process model predictions for
    Kolmogorov flow data. Supports both spatial and spatiotemporal problems.
    """

    # Constants
    EXPECTED_CHANNELS = 2
    SPATIAL_COORD_DIMS = 2
    SPATIOTEMPORAL_COORD_DIMS = 3
    COLUMN_TITLES = ["Ground Truth", "Context Set", "Prediction Mean", "Prediction Std"]
    COLUMN_CMAPS = ["RdBu_r", "RdBu_r", "RdBu_r", "magma"]
    COLUMN_BASES = [0, 2, 4, 6]

    def __init__(
        self,
        figsize: tuple[float, float] = (8.0, 2.5),
        savefig: bool = False,
        logging: bool = True,
        plot_dir: str = "fig",
        show_plots: bool = False,
        time_steps: list[int] | None = None,
        max_time_steps: int = 4,
        hspace: float = 0.1,
        outer_wspace: float = 0.1,
        inner_wspace: float = 0.05,
        temp_colorbar_height: float = 0.015,
        std_colorbar_height: float = 0.015,
        colorbar_spacing: float = 0.4,
    ) -> None:
        """Initialize the KolmogorovPlotter.

        Args:
            figsize: Figure size as (width, height) in inches.
            savefig: Whether to save figures to disk.
            logging: Whether to enable logging output.
            plot_dir: Directory to save plots.
            show_plots: Whether to display plots.
            time_steps: Specific time steps to plot for spatiotemporal data.
            max_time_steps: Maximum number of time steps to plot if time_steps not specified.
            hspace: Vertical spacing between subplot rows.
            outer_wspace: Horizontal spacing between column groups.
            inner_wspace: Horizontal spacing within channel subplots.
            temp_colorbar_height: Height ratio for temperature colorbar.
            std_colorbar_height: Height ratio for standard deviation colorbar.
            colorbar_spacing: Vertical spacing between plots and first colorbar row.
        """
        self.figsize = figsize
        self.savefig = savefig
        self.logging = logging
        self.plot_dir = Path(plot_dir)
        self.show_plots = show_plots
        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        self.hspace = hspace
        self.outer_wspace = outer_wspace
        self.inner_wspace = inner_wspace
        self.temp_colorbar_height = temp_colorbar_height
        self.std_colorbar_height = std_colorbar_height
        self.colorbar_spacing = colorbar_spacing

    def _prepare_batch(self, batch: KolmogorovBatch) -> KolmogorovBatch:
        batch_copy = copy.deepcopy(batch)

        for key, value in vars(batch_copy).items():
            if isinstance(value, torch.Tensor):
                setattr(batch_copy, key, value[:1])

        return batch_copy

    def _detect_problem_mode(self, batch: KolmogorovBatch) -> str:
        coord_dims = batch.x_grid.shape[1]
        if coord_dims == self.SPATIAL_COORD_DIMS:
            return "spatial"
        elif coord_dims == self.SPATIOTEMPORAL_COORD_DIMS:
            return "spatiotemporal"
        else:
            raise ValueError(f"Unsupported coordinate dimensions: {coord_dims}")

    def _select_time_steps(self, total_time_steps: int) -> list[int]:
        if self.time_steps is not None:
            valid_steps = [t for t in self.time_steps if 0 <= t < total_time_steps]
            if not valid_steps:
                raise ValueError(
                    "No valid time steps found. "
                    f"Available range: 0-{total_time_steps-1}"
                )
            return valid_steps

        num_steps = min(self.max_time_steps, total_time_steps)
        if num_steps == total_time_steps:
            return list(range(total_time_steps))

        return cast(
            list[int], np.linspace(0, total_time_steps - 1, num_steps, dtype=int).tolist()
        )

    def _validate_channels(self, channels: int) -> None:
        if channels != self.EXPECTED_CHANNELS:
            raise ValueError(
                f"KolmogorovPlotter only supports {self.EXPECTED_CHANNELS} channels "
                f"for the 2D Kolmogorov equation, but got {channels} channels"
            )

    def _remove_ticks(self, ax) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _add_subplot_labels(
        self,
        ax,
        data_idx: int,
        ch: int,
        title: str,
        time_step: int,
        is_first_row: bool,
        is_last_row: bool,
    ) -> None:
        if is_first_row and ch == 0:
            title_fontsize = max(8, min(12, ax.figure.get_figwidth() * 0.6))
            ax.text(
                1,
                1.15,
                title,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=title_fontsize,
                fontweight="bold",
            )

        if data_idx == 0 and ch == 0:
            time_fontsize = max(8, min(10, ax.figure.get_figwidth() * 0.5))
            ax.text(
                -0.15,
                0.5,
                f"t={time_step}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=time_fontsize,
                rotation=90,
            )

        if is_last_row:
            ch_fontsize = max(6, min(8, ax.figure.get_figwidth() * 0.4))
            ax.text(
                0.5,
                -0.03,
                f"Ch {ch}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=ch_fontsize,
            )

    def _create_channel_subplots(
        self, fig, gs, num_groups: int, num_channels: int
    ) -> np.ndarray:
        axes = []
        for i in range(num_groups):
            gs_sub = gs[i].subgridspec(1, num_channels, wspace=self.inner_wspace)
            for j in range(num_channels):
                axes.append(fig.add_subplot(gs_sub[j]))
        return np.array(axes)

    def _create_spatiotemporal_subplots(
        self, fig, gs_main, num_time_steps
    ) -> np.ndarray:
        axes = []
        for row in range(num_time_steps):
            row_axes = []
            for col in range(4):
                gs_sub = gs_main[row, col].subgridspec(
                    1, self.EXPECTED_CHANNELS, wspace=self.inner_wspace
                )
                for ch in range(self.EXPECTED_CHANNELS):
                    row_axes.append(fig.add_subplot(gs_sub[ch]))
            axes.append(row_axes)
        return np.array(axes)

    def _plot_context_with_non_context_channel(
        self,
        ax: matplotlib.axes.Axes,
        context_data: np.ndarray,
        mc_mask: np.ndarray,
        ch: int,
        vmin: float,
        vmax: float,
        cmap: str = "RdBu_r",
    ) -> matplotlib.image.AxesImage:
        """Plot context data with non-context points in distinct color for a single channel."""
        # Create a composite image with context and non-context visualization
        display_data = np.full_like(context_data[ch], np.nan)

        # Set context points to their actual values
        context_mask = mc_mask[ch] == 1
        display_data[context_mask] = context_data[ch][context_mask]

        # Plot context points with original colormap
        im = ax.imshow(display_data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")

        # Overlay non-context points in gray
        non_context_mask = mc_mask[ch] == 0
        if np.any(non_context_mask):
            # Create gray overlay for non-context points
            gray_overlay = np.full_like(context_data[ch], 0.5)  # Mid-gray value
            gray_masked = np.ma.masked_where(~non_context_mask, gray_overlay)
            ax.imshow(
                gray_masked,
                cmap="gray",
                alpha=1.0,  # Full opacity for clear gray distinction
                vmin=0,
                vmax=1,
                origin="upper",
            )

        return im

    def _plot_spatial_channels(
        self, axes, data_arrays, mc_mask=None
    ) -> matplotlib.image.AxesImage:
        im = None
        for ch in range(self.EXPECTED_CHANNELS):
            for data_idx, (data, cmap) in enumerate(
                zip(data_arrays, self.COLUMN_CMAPS)
            ):
                col_idx = self.COLUMN_BASES[data_idx] + ch

                if cmap == "magma":
                    vmin, vmax = data.min(), data.max()
                    im = axes[col_idx].imshow(data[ch], vmin=vmin, vmax=vmax, cmap=cmap)
                elif (
                    data_idx == 1 and mc_mask is not None
                ):  # Context set column with mask
                    vmin, vmax = data_arrays[0].min(), data_arrays[0].max()
                    self._plot_context_with_non_context_channel(
                        axes[col_idx], data, mc_mask, ch, vmin, vmax, cmap
                    )
                else:
                    vmin, vmax = data_arrays[0].min(), data_arrays[0].max()
                    axes[col_idx].imshow(data[ch], vmin=vmin, vmax=vmax, cmap=cmap)

                axes[col_idx].text(
                    0.5,
                    -0.03,
                    f"Ch {ch}",
                    ha="center",
                    va="top",
                    transform=axes[col_idx].transAxes,
                    fontsize=8,
                )
                self._remove_ticks(axes[col_idx])
        # The "magma" column always runs (EXPECTED_CHANNELS >= 1), so im is set.
        assert im is not None
        return im

    def _add_column_titles(self, axes) -> None:
        title_fontsize = max(10, min(16, self.figsize[0] * 0.8))
        for idx, title in enumerate(self.COLUMN_TITLES):
            col_start = self.COLUMN_BASES[idx]
            axes[col_start].text(
                1,
                1.15,
                title,
                ha="center",
                va="bottom",
                transform=axes[col_start].transAxes,
                fontsize=title_fontsize,
                fontweight="bold",
            )

    def _add_colorbar(
        self,
        fig: plt.Figure,
        im: matplotlib.image.AxesImage,
        cbar_ax: matplotlib.axes.Axes,
        label: str = "Temperature",
    ) -> None:
        """Add colorbar to the dedicated colorbar axis."""
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(label, fontsize=8)

    def _add_suptitle(
        self, fig, mode: str, context_prop: float, model_ll: float
    ) -> None:
        suptitle_fontsize = max(12, min(20, self.figsize[0] * 1.0))
        fig.suptitle(
            f"Kolmogorov Flow ({mode}) - "
            f"context/all = {context_prop:.3f}, "
            f"LogP = {model_ll:.3f}",
            fontsize=suptitle_fontsize,
            y=0.98,
        )

    def _plot_time_slice(
        self,
        yc_slice: np.ndarray,
        y_slice: np.ndarray,
        mean_slice: np.ndarray,
        std_slice: np.ndarray,
        y_mc_slice: np.ndarray,
        time_step: int,
        _: float,
        axes: np.ndarray,
        row_idx: int,
        channels: int = 1,
        is_last_row: bool = False,
    ) -> matplotlib.image.AxesImage | None:
        """Plot a single time slice across columns.

        Args:
            yc_slice: Context data slice [C, H, W] or [H, W]
            y_slice: Ground truth slice [C, H, W] or [H, W]
            mean_slice: Mean prediction slice [C, H, W] or [H, W]
            std_slice: Standard deviation slice [C, H, W] or [H, W]
            y_mc_slice: Context mask slice [C, H, W] or [H, W]
            time_step: Time step index
            time_value: Actual time value
            axes: Subplot axes array
            row_idx: Row index in the subplot grid
            channels: Number of channels
            is_last_row: Whether this is the last row (for channel captions)
        """
        self._validate_channels(channels)

        data_arrays = [y_slice, yc_slice, mean_slice, std_slice]

        last_im = None
        for data_idx, (data, title, cmap) in enumerate(
            zip(data_arrays, self.COLUMN_TITLES, self.COLUMN_CMAPS)
        ):
            col_base = self.COLUMN_BASES[data_idx]

            for ch in range(self.EXPECTED_CHANNELS):
                col = col_base + ch
                vmin, vmax = (
                    (data.min(), data.max())
                    if cmap == "magma"
                    else (y_slice.min(), y_slice.max())
                )

                if (
                    data_idx == 1
                ):  # Context set column - show non-context points with distinct color
                    im = self._plot_context_with_non_context_channel(
                        axes[row_idx, col], data, y_mc_slice, ch, vmin, vmax, cmap
                    )
                else:
                    im = axes[row_idx, col].imshow(
                        data[ch], vmin=vmin, vmax=vmax, cmap=cmap
                    )

                if cmap == "magma":
                    last_im = im

                self._add_subplot_labels(
                    axes[row_idx, col],
                    data_idx,
                    ch,
                    title,
                    time_step,
                    row_idx == 0,
                    is_last_row,
                )
                self._remove_ticks(axes[row_idx, col])

        return last_im

    def _create_spatial_subplot_layout(
        self,
    ) -> tuple[
        matplotlib.figure.Figure,
        list[matplotlib.axes.Axes],
        matplotlib.axes.Axes,
        matplotlib.axes.Axes,
    ]:
        """Create subplot layout for spatial plotting with two colorbar rows."""
        fig = plt.figure(figsize=self.figsize)
        width_ratios = [1, 1, 1, 1]  # Equal width for all columns

        # Create main grid for plots (row 0) and two colorbar spaces (rows 1 and 2)
        gs_main = fig.add_gridspec(
            3,
            1,
            height_ratios=[1, self.temp_colorbar_height, self.std_colorbar_height],
            hspace=self.colorbar_spacing,
            left=0.05,
            right=0.95,
            top=0.88,
            bottom=0.1,
        )

        # Sub-grid for the 4 columns in the main plot area
        gs_plots = gs_main[0].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.outer_wspace
        )

        # Sub-grid for temperature colorbar (spans first 3 columns)
        gs_temp_colorbar = gs_main[1].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.outer_wspace
        )

        # Sub-grid for std colorbar (spans last column)
        gs_std_colorbar = gs_main[2].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.outer_wspace
        )

        axes = []
        for col in range(4):
            # Create sub-gridspec for channels within each column
            gs_sub = gs_plots[col].subgridspec(
                1, self.EXPECTED_CHANNELS, wspace=self.inner_wspace
            )
            for ch in range(self.EXPECTED_CHANNELS):
                axes.append(fig.add_subplot(gs_sub[ch]))

        # Create colorbar axes
        temp_cbar_ax = fig.add_subplot(gs_temp_colorbar[1:3])  # Spans first 3 columns
        std_cbar_ax = fig.add_subplot(gs_std_colorbar[1:3])  # Spans last column

        return fig, axes, temp_cbar_ax, std_cbar_ax

    def _plot_spatial_single(
        self,
        yc: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_mc_grid: torch.Tensor,
        context_prop: float,
        model_ll: float,
        fig_path: Path,
    ) -> None:
        """Create a figure for spatial problem (single image).

        Args:
            yc: Context data [C, H, W]
            y: Ground truth data [C, H, W]
            mean: Mean predictions [H*W, C]
            std: Standard deviation predictions [H*W, C]
            y_mc_grid: Context mask grid [C, H, W]
            context_prop: Proportion of context points
            model_ll: Model log-likelihood
            fig_path: Path to save the figure
        """
        channels, height, width = y.shape

        # Reshape predictions to spatial format
        mean_reshaped = einops.rearrange(mean, "(h w) c -> c h w", h=height, w=width)
        std_reshaped = einops.rearrange(std, "(h w) c -> c h w", h=height, w=width)

        self._validate_channels(channels)

        # Create figure with new layout
        fig, axes, temp_cbar_ax, std_cbar_ax = self._create_spatial_subplot_layout()

        # Convert mask to numpy for consistent handling
        y_mc_np = y_mc_grid.cpu().numpy()

        data_arrays = [y, yc, mean_reshaped, std_reshaped]

        # Plot channels and collect images for colorbars
        y_values_im = None
        std_im = None

        for ch in range(self.EXPECTED_CHANNELS):
            for data_idx, (data, cmap) in enumerate(
                zip(data_arrays, self.COLUMN_CMAPS)
            ):
                col_idx = self.COLUMN_BASES[data_idx] + ch

                if cmap == "magma":
                    vmin, vmax = data.min(), data.max()
                    im = axes[col_idx].imshow(data[ch], vmin=vmin, vmax=vmax, cmap=cmap)
                    if ch == 0:  # Store for std colorbar
                        std_im = im
                elif (
                    data_idx == 1 and y_mc_np is not None
                ):  # Context set column with mask
                    vmin, vmax = data_arrays[0].min(), data_arrays[0].max()
                    im = self._plot_context_with_non_context_channel(
                        axes[col_idx], data, y_mc_np, ch, vmin, vmax, cmap
                    )
                    if ch == 0:  # Store for y_values colorbar
                        y_values_im = im
                else:
                    vmin, vmax = data_arrays[0].min(), data_arrays[0].max()
                    im = axes[col_idx].imshow(data[ch], vmin=vmin, vmax=vmax, cmap=cmap)
                    if ch == 0 and data_idx < 3:  # Store for y_values colorbar
                        y_values_im = im

                axes[col_idx].text(
                    0.5,
                    -0.03,
                    f"Ch {ch}",
                    ha="center",
                    va="top",
                    transform=axes[col_idx].transAxes,
                    fontsize=8,
                )
                self._remove_ticks(axes[col_idx])

        self._add_column_titles(axes)

        # Add colorbar for y values (first 3 columns)
        if y_values_im is not None:
            self._add_colorbar(fig, y_values_im, temp_cbar_ax, label="Y Values")

        # Add colorbar for standard deviation (last column)
        if std_im is not None:
            self._add_colorbar(fig, std_im, std_cbar_ax, label="Standard Deviation")

        self._add_suptitle(fig, "Spatial", context_prop, model_ll)

        self._handle_figure_output(
            fig,
            fig_path,
            savefig=self.savefig,
            logging=self.logging,
            show_plots=self.show_plots,
        )

        plt.close(fig)

    def _create_spatiotemporal_subplot_layout(self, num_time_steps: int) -> tuple[
        matplotlib.figure.Figure,
        list[list[matplotlib.axes.Axes]],
        matplotlib.axes.Axes,
        matplotlib.axes.Axes,
    ]:
        """Create subplot layout for spatiotemporal plotting with two colorbar rows."""
        fig = plt.figure(
            figsize=(self.figsize[0], self.figsize[1] * num_time_steps / 2)
        )
        width_ratios = [1, 1, 1, 1]  # Equal width for all columns

        # Create main grid for plots and two colorbar spaces
        gs_main = fig.add_gridspec(
            3,
            1,
            height_ratios=[1, self.temp_colorbar_height, self.std_colorbar_height],
            hspace=self.colorbar_spacing,
            left=0.05,
            right=0.90,
            top=0.88,
            bottom=0.1,
        )

        # Sub-grid for the time steps and columns in the main plot area
        gs_plots = gs_main[0].subgridspec(
            num_time_steps,
            4,
            width_ratios=width_ratios,
            hspace=self.hspace,
            wspace=self.outer_wspace,
        )

        # Sub-grid for temperature colorbar (spans first 3 columns)
        gs_temp_colorbar = gs_main[1].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.outer_wspace
        )

        # Sub-grid for std colorbar (spans last column)
        gs_std_colorbar = gs_main[2].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.outer_wspace
        )

        axes = []
        for row in range(num_time_steps):
            row_axes = []
            for col in range(4):
                gs_sub = gs_plots[row, col].subgridspec(
                    1, self.EXPECTED_CHANNELS, wspace=self.inner_wspace
                )
                for ch in range(self.EXPECTED_CHANNELS):
                    row_axes.append(fig.add_subplot(gs_sub[ch]))
            axes.append(row_axes)

        # Create colorbar axes
        temp_cbar_ax = fig.add_subplot(gs_temp_colorbar[1:3])  # Spans first 3 columns
        std_cbar_ax = fig.add_subplot(gs_std_colorbar[1:3])  # Spans last column

        return fig, axes, temp_cbar_ax, std_cbar_ax

    def _plot_spatiotemporal_subplots(
        self,
        yc: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_mc_grid: torch.Tensor,
        selected_time_steps: list[int],
        context_prop: float,
        model_ll: float,
        fig_path: Path,
    ) -> None:
        """Create a figure with subplots for multiple time steps
        (spatiotemporal mode).

        Args:
            yc: Context data [C, T, H, W]
            y: Ground truth data [C, T, H, W]
            mean: Mean predictions [T*H*W, C]
            std: Standard deviation predictions [T*H*W, C]
            y_mc_grid: Context mask grid [C, T, H, W]
            selected_time_steps: Time steps to plot
            context_prop: Proportion of context points
            model_ll: Model log-likelihood
            fig_path: Path to save the figure
        """
        num_time_steps = len(selected_time_steps)
        total_t, height, width = y.shape[1:]

        # Create figure with new layout
        fig, axes, temp_cbar_ax, std_cbar_ax = (
            self._create_spatiotemporal_subplot_layout(num_time_steps)
        )

        mean_reshaped = einops.rearrange(
            mean, "(t h w) c -> c t h w", t=total_t, h=height, w=width
        )
        std_reshaped = einops.rearrange(
            std, "(t h w) c -> c t h w", t=total_t, h=height, w=width
        )

        # Convert mask to numpy for consistent handling
        y_mc_np = y_mc_grid.cpu().numpy()

        y_values_im = None
        std_im = None

        for i, time_step in enumerate(selected_time_steps):
            y_slice = y[:, time_step]
            yc_slice = yc[:, time_step]
            mean_slice = mean_reshaped[:, time_step]
            std_slice = std_reshaped[:, time_step]
            y_mc_slice = y_mc_np[:, time_step]

            is_last_row = i == len(selected_time_steps) - 1
            data_arrays = [y_slice, yc_slice, mean_slice, std_slice]

            for data_idx, (data, title, cmap) in enumerate(
                zip(data_arrays, self.COLUMN_TITLES, self.COLUMN_CMAPS)
            ):
                col_base = self.COLUMN_BASES[data_idx]

                for ch in range(self.EXPECTED_CHANNELS):
                    col = col_base + ch
                    vmin, vmax = (
                        (data.min(), data.max())
                        if cmap == "magma"
                        else (y_slice.min(), y_slice.max())
                    )

                    if (
                        data_idx == 1
                    ):  # Context set column - show non-context points with distinct color
                        im = self._plot_context_with_non_context_channel(
                            axes[i][col], data, y_mc_slice, ch, vmin, vmax, cmap
                        )
                        if i == 0 and ch == 0:  # Store for y_values colorbar
                            y_values_im = im
                    else:
                        im = axes[i][col].imshow(
                            data[ch], vmin=vmin, vmax=vmax, cmap=cmap
                        )
                        if i == 0 and ch == 0:  # Store for colorbars
                            if cmap == "magma":
                                std_im = im
                            elif data_idx < 3:
                                y_values_im = im

                    self._add_subplot_labels(
                        axes[i][col],
                        data_idx,
                        ch,
                        title,
                        time_step,
                        i == 0,
                        is_last_row,
                    )
                    self._remove_ticks(axes[i][col])

        # Add colorbar for y values (first 3 columns)
        if y_values_im is not None:
            self._add_colorbar(fig, y_values_im, temp_cbar_ax, label="Y Values")

        # Add colorbar for standard deviation (last column)
        if std_im is not None:
            self._add_colorbar(fig, std_im, std_cbar_ax, label="Standard Deviation")

        self._add_suptitle(fig, "Spatiotemporal", context_prop, model_ll)

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
        batches: list[KolmogorovBatch],
        name: str = "plot",
        **kwargs,
    ) -> None:
        """
        Generate and display/save plots for given batches.

        Args:
            model (BaseNeuralProcess): The model used for predictions.
            batches (list[KolmogorovBatch]): A list of data batches.
            name (str): Name for saving figures.
            **kwargs: Additional arguments to be passed to the model.
        """
        fig_dir = self.plot_dir / name
        if self.savefig:
            fig_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(batches):
            # Prepare batch for plotting
            eval_batch = self._prepare_batch(batch)

            self._validate_channels(eval_batch.y_grid.shape[1])

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
            y_mc_grid = plot_batch.y_mc_grid.detach().cpu()
            y_grid = plot_batch.y_grid.detach().cpu()
            x_grid = plot_batch.x_grid.detach().cpu()

            # Detect problem mode
            problem_mode = self._detect_problem_mode(plot_batch)

            # Calculate context proportion (across all points)
            context_proportion = y_mc_grid[0, 0].sum() / y_mc_grid[0, 0].numel()

            # Take first item in batch for plotting
            y_grid, x_grid = y_grid[0], x_grid[0]
            y_mc_grid = y_mc_grid[0]
            mean, std = mean[0], std[0]
            yc_grid = torch.where(y_mc_grid > 0, y_grid, torch.zeros_like(y_grid))

            # Create plots based on problem mode
            fname = fig_dir / f"{i:03d}"

            if problem_mode == "spatial":
                # y_grid is [C, H, W], mean/std are [H*W, C]
                self._plot_spatial_single(
                    yc_grid.cpu().numpy(),
                    y_grid.cpu().numpy(),
                    mean.cpu().numpy(),
                    std.cpu().numpy(),
                    y_mc_grid,
                    float(context_proportion),
                    float(model_ll),
                    fname,
                )
            elif problem_mode == "spatiotemporal":
                # y_grid is [C, T, H, W], mean/std are [T*H*W, C]
                _, time_steps, _, _ = y_grid.shape

                # Select which time steps to visualize
                selected_time_steps = self._select_time_steps(time_steps)

                self._plot_spatiotemporal_subplots(
                    yc_grid.cpu().numpy(),
                    y_grid.cpu().numpy(),
                    mean.cpu().numpy(),
                    std.cpu().numpy(),
                    y_mc_grid,
                    selected_time_steps,
                    float(context_proportion),
                    float(model_ll),
                    fname,
                )
            else:
                raise ValueError(f"Unknown problem mode: {problem_mode}")
