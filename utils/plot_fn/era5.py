import copy
import pickle
from pathlib import Path
from typing import Any, cast

import einops
import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

from nps.models.base import BaseNeuralProcess

from ..data import ERA5Batch
from .base import BaseNeuralProcessPlotter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class ERA5Plotter(BaseNeuralProcessPlotter):
    """
    A class to handle plotting of Neural Process model predictions for ERA5 data.
    Supports both spatial and spatiotemporal problems with optional geospatial mapping.

    CAUTION: Ensure that all longitude values follow the specified coordinate format.
    In particular, the x-values in all batches must adhere to this standard. If the
    coordinates have not been normalized, they should be provided directly in this
    format. If the coordinates are normalized, the normalization statistics must be
    chosen so that rescaling them restores values within this format.
    """

    # Display configuration constants
    EXPECTED_CHANNELS = 1
    COLUMN_TITLES = ["Ground Truth", "Context Set", "Prediction Mean", "Prediction Std"]
    COLUMN_CMAPS = ["RdBu_r", "RdBu_r", "RdBu_r", "magma"]
    DEFAULT_COORDINATE_RANGES = {"lat": (-90, 90), "lon": (-180, 180)}

    def __init__(
        self,
        figsize: tuple[float, float] = (8.0, 4.5),
        savefig: bool = False,
        logging: bool = True,
        plot_dir: str = "fig",
        show_plots: bool = False,
        time_steps: list[int] | None = None,
        max_time_steps: int = 3,
        use_cartopy: bool = True,
        stats_cache_dir: str | None = None,
        coords_normalized: bool = True,
        x_has_surface_elevation: bool = False,
        subplot_rows_spacing: float = 0.2,
        column_spacing: float = 0.3,
        show_latitude: bool = True,
        show_longitude: bool = True,
        lat_lon_fontsize: float = 5,
        num_lat_ticks: int | None = None,
        num_lon_ticks: int | None = None,
        temp_colorbar_height: float = 0.015,
        std_colorbar_height: float = 0.015,
        colorbar_spacing: float = 0.3,
    ) -> None:
        """
        Initialize ERA5Plotter.

        Args:
            figsize: Figure size for plots
            savefig: Whether to save figures to disk
            logging: Whether to log figures to wandb
            plot_dir: Directory for saving plots
            show_plots: Whether to display plots
            time_steps: Specific time steps to plot (spatiotemporal mode)
            max_time_steps: Maximum number of time steps to plot
            use_cartopy: Whether to use cartopy for geospatial mapping
            stats_cache_dir: Directory containing cached ERA5Dataset normalization statistics
            coords_normalized: Whether latitudes and longitudes in the batch are normalized.
                If True and using cartopy, stats_cache_dir must be provided for denormalization
            x_has_surface_elevation: Whether batch.x includes surface elevation as a feature
            subplot_rows_spacing: Vertical spacing between subplot rows
            column_spacing: Horizontal spacing between column groups
            show_latitude: Whether to display latitude values on plots
            show_longitude: Whether to display longitude values on plots
            lat_lon_fontsize: Font size for latitude and longitude labels
            num_lat_ticks: Number of latitude ticks to display (None for automatic)
            num_lon_ticks: Number of longitude ticks to display (None for automatic)
            temp_colorbar_height: Height ratio for temperature colorbar (default 0.05)
            std_colorbar_height: Height ratio for standard deviation colorbar (default 0.05)
            colorbar_spacing: Vertical spacing between plot and colorbars (default 0.05)
        """
        # Core configuration
        self.figsize = figsize
        self.savefig = savefig
        self.logging = logging
        self.plot_dir = Path(plot_dir)
        self.show_plots = show_plots

        # Time configuration
        self.time_steps = time_steps
        self.max_time_steps = max_time_steps

        # Coordinate system configuration
        self.use_cartopy = use_cartopy
        self.stats_cache_dir = Path(stats_cache_dir) if stats_cache_dir else None
        self.coords_normalized = coords_normalized
        self.x_has_surface_elevation = x_has_surface_elevation

        # Layout configuration
        self.subplot_rows_spacing = subplot_rows_spacing
        self.column_spacing = column_spacing
        self.temp_colorbar_height = temp_colorbar_height
        self.std_colorbar_height = std_colorbar_height
        self.colorbar_spacing = colorbar_spacing

        # Display configuration
        self.show_latitude = show_latitude
        self.show_longitude = show_longitude
        self.lat_lon_fontsize = lat_lon_fontsize
        self.num_lat_ticks = num_lat_ticks
        self.num_lon_ticks = num_lon_ticks

        # Internal state
        self._norm_stats = None

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate plotter configuration and dependencies."""
        if self.use_cartopy and not CARTOPY_AVAILABLE:
            raise ImportError(
                "Cartopy is required for geospatial plotting but is not installed. "
                "Install it with: pip install cartopy"
            )

        if self.use_cartopy and self.coords_normalized and not self.stats_cache_dir:
            raise ValueError(
                "When using cartopy with normalized coordinates, stats_cache_dir must be provided "
                "to denormalize coordinates to standard ranges (lat: [-90,90], lon: [-180,180])"
            )

    def _load_normalization_stats(self) -> dict | None:
        """Load cached normalization statistics from ERA5Dataset."""
        if not self.stats_cache_dir or self._norm_stats is not None:
            return self._norm_stats

        try:
            stats_files = list(self.stats_cache_dir.glob("era5_stats_*.pkl"))
            if not stats_files:
                return None

            latest_stats_file = max(stats_files, key=lambda p: p.stat().st_mtime)
            with open(latest_stats_file, "rb") as f:
                self._norm_stats = pickle.load(f)

            return self._norm_stats
        except Exception as e:
            print(f"Warning: Could not load normalization statistics: {e}")
            return None

    def _denormalize_coordinates(
        self, coords: torch.Tensor, *, has_time: bool, has_elevation: bool
    ) -> torch.Tensor:
        """Denormalize coordinates using cached statistics if available."""
        if not self.coords_normalized:
            return coords

        stats = self._load_normalization_stats()
        if stats is None:
            return coords

        coords_denorm = coords.clone()
        coord_mapping = {"latitude": -2, "longitude": -1}

        if has_time:
            coord_mapping["numerical_time"] = 0
        if has_elevation:
            coord_mapping["surface_elevation"] = -3

        for name, ch_idx in coord_mapping.items():
            if name in stats["coords_mean"]:
                mean, std = stats["coords_mean"][name], stats["coords_std"][name]
                coords_denorm[ch_idx, ...] = coords_denorm[ch_idx, ...] * std + mean

        return self._validate_coordinate_ranges(coords_denorm)

    def _validate_coordinate_ranges(self, coords: torch.Tensor) -> torch.Tensor:
        """Validate and clip coordinates to valid ranges for cartopy compatibility."""
        lat_idx, lon_idx = -2, -1

        for coord_name, idx, (min_val, max_val) in [
            ("Latitude", lat_idx, self.DEFAULT_COORDINATE_RANGES["lat"]),
            ("Longitude", lon_idx, self.DEFAULT_COORDINATE_RANGES["lon"]),
        ]:
            values = coords[idx, ...]
            val_min, val_max = values.min().item(), values.max().item()

            if val_min < min_val or val_max > max_val:
                warning_msg = (
                    f"Warning: {coord_name} values [{val_min:.2f}, {val_max:.2f}] "
                    f"fall outside standard range [{min_val}, {max_val}]. Clipping to valid range."
                )
                if self.use_cartopy:
                    warning_msg += " This is required for cartopy compatibility."
                print(warning_msg)
                coords[idx, ...] = torch.clamp(values, min=min_val, max=max_val)

        return coords

    def _prepare_batch(self, batch: ERA5Batch) -> ERA5Batch:
        """Prepare batch by taking the first item."""
        batch_copy = copy.deepcopy(batch)
        for key, value in vars(batch_copy).items():
            if isinstance(value, torch.Tensor):
                setattr(batch_copy, key, value[:1])
        return batch_copy

    def _detect_problem_mode(self, batch: ERA5Batch) -> str:
        """Detect whether this is a spatial or spatiotemporal problem."""
        ndim = len(batch.x_grid.shape)

        if ndim == 4:  # [B, C, H, W]
            return "spatial"
        elif ndim == 5:  # [B, C, T, H, W] or [B, C, D, H, W]
            return "spatial" if self.x_has_surface_elevation else "spatiotemporal"
        elif ndim == 6:  # [B, C, T, D, H, W]
            return "spatiotemporal"
        else:
            raise ValueError(f"Unsupported x_grid shape: {batch.x_grid.shape}")

    def _extract_spatial_coordinates(
        self, x_grid: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract spatial coordinates from grid tensor."""
        lat_coords = x_grid[-2].cpu().numpy()  # [H, W]
        lon_coords = x_grid[-1].cpu().numpy()  # [H, W]
        return lat_coords, lon_coords

    def _calculate_color_limits(
        self, ground_truth: np.ndarray, error_margin: float = 0.1
    ) -> tuple[float, float]:
        """Calculate color limits based on ground truth with error margin."""
        valid_gt = ground_truth[~np.isnan(ground_truth)]
        if len(valid_gt) <= 0:
            raise ValueError(
                "No valid ground truth values found. "
                "All ground truth data contains NaN values. "
                "Please check your input data for validity."
            )

        gt_min, gt_max = np.min(valid_gt), np.max(valid_gt)
        gt_range = gt_max - gt_min
        margin = error_margin * gt_range
        return gt_min - margin, gt_max + margin

    def _validate_model_output(
        self, std_data: np.ndarray, time_step: int | None = None
    ) -> tuple[float, float]:
        """Validate model output and return std limits."""
        if np.any(np.isnan(std_data)):
            time_info = f" for time step {time_step}" if time_step is not None else ""
            raise ValueError(
                f"Standard deviation values contain NaN{time_info}. "
                "Model outputs should not produce NaN values. "
                "Please check your model implementation and training."
            )
        return std_data.min(), std_data.max()

    def _select_time_steps(self, total_time_steps: int) -> list[int]:
        """Select which time steps to plot."""
        if self.time_steps is not None:
            valid_steps = [t for t in self.time_steps if 0 <= t < total_time_steps]
            if not valid_steps:
                raise ValueError(
                    f"No valid time steps found. Available range: 0-{total_time_steps-1}"
                )
            return valid_steps

        num_steps = min(self.max_time_steps, total_time_steps)
        if num_steps == total_time_steps:
            return list(range(total_time_steps))
        return cast(
            list[int], np.linspace(0, total_time_steps - 1, num_steps, dtype=int).tolist()
        )

    def _remove_ticks(self, ax: matplotlib.axes.Axes) -> None:
        """Remove axis ticks and labels unless coordinates should be shown."""
        if self.use_cartopy and CARTOPY_AVAILABLE:
            return

        if not self.show_longitude:
            ax.set_xticks([])
        if not self.show_latitude:
            ax.set_yticks([])

    def _create_spatial_subplot_layout(self, use_projection: bool = False) -> tuple[
        matplotlib.figure.Figure,
        list[matplotlib.axes.Axes],
        matplotlib.axes.Axes,
        matplotlib.axes.Axes,
    ]:
        """Create subplot layout for spatial plotting (4 columns: GT, Context, Mean, Std) with two colorbar rows."""
        fig = plt.figure(figsize=self.figsize)
        width_ratios = [1, 1, 1, 1]  # Equal width for all columns
        left_margin = 0.1 if self.show_latitude else 0.05

        # Create main grid for plots (row 0) and two colorbar spaces (rows 1 and 2)
        gs_main = fig.add_gridspec(
            3,
            1,
            height_ratios=[1, self.temp_colorbar_height, self.std_colorbar_height],
            hspace=self.colorbar_spacing,
            left=left_margin,
            right=0.95,
            top=0.9,
            bottom=0.1,
        )

        # Sub-grid for the 4 columns in the main plot area
        gs_plots = gs_main[0].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.column_spacing
        )

        # Sub-grid for temperature colorbar (spans 2nd and 3rd columns)
        gs_temp_colorbar = gs_main[1].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.column_spacing
        )

        # Sub-grid for std colorbar (spans last column)
        gs_std_colorbar = gs_main[2].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.column_spacing
        )

        axes = []
        for col in range(4):
            subplot_kw = (
                {"projection": ccrs.PlateCarree()}
                if use_projection and self.use_cartopy
                else {}
            )
            ax = fig.add_subplot(gs_plots[col], **subplot_kw)
            axes.append(ax)

        # Create colorbar axes
        temp_cbar_ax = fig.add_subplot(
            gs_temp_colorbar[1:3]
        )  # Spans 2nd and 3rd columns
        std_cbar_ax = fig.add_subplot(gs_std_colorbar[1:3])  # Spans last column

        return fig, axes, temp_cbar_ax, std_cbar_ax

    def _create_spatiotemporal_subplot_layout(
        self, num_time_steps: int, use_projection: bool = False
    ) -> tuple[
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
        # Adjust left margin when showing latitude to accommodate time step labels
        left_margin = 0.12 if self.show_latitude else 0.05

        # Create main grid for plots and two colorbar spaces
        gs_main = fig.add_gridspec(
            3,
            1,
            height_ratios=[1, self.temp_colorbar_height, self.std_colorbar_height],
            hspace=self.colorbar_spacing,
            left=left_margin,
            right=0.90,
            top=0.9,
            bottom=0.1,
        )

        # Sub-grid for the time steps and columns in the main plot area
        gs_plots = gs_main[0].subgridspec(
            num_time_steps,
            4,
            width_ratios=width_ratios,
            hspace=self.subplot_rows_spacing,
            wspace=self.column_spacing,
        )

        # Sub-grid for temperature colorbar (spans 2nd and 3rd columns)
        gs_temp_colorbar = gs_main[1].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.column_spacing
        )

        # Sub-grid for std colorbar (spans last column)
        gs_std_colorbar = gs_main[2].subgridspec(
            1, 4, width_ratios=width_ratios, wspace=self.column_spacing
        )

        axes = []
        for row in range(num_time_steps):
            row_axes = []
            for col in range(4):
                subplot_kw = (
                    {"projection": ccrs.PlateCarree()}
                    if use_projection and self.use_cartopy
                    else {}
                )
                ax = fig.add_subplot(gs_plots[row, col], **subplot_kw)
                row_axes.append(ax)
            axes.append(row_axes)

        # Create colorbar axes
        temp_cbar_ax = fig.add_subplot(
            gs_temp_colorbar[1:3]
        )  # Spans 2nd and 3rd columns
        std_cbar_ax = fig.add_subplot(gs_std_colorbar[1:3])  # Spans last column

        return fig, axes, temp_cbar_ax, std_cbar_ax

    def _add_column_titles(self, axes: list[matplotlib.axes.Axes]) -> None:
        """Add titles to subplot columns."""
        title_fontsize = max(10, min(10, self.figsize[0] * 0.5))
        for idx, title in enumerate(self.COLUMN_TITLES):
            axes[idx].text(
                1,
                1.15,
                title,
                ha="center",
                va="bottom",
                transform=axes[idx].transAxes,
                fontsize=title_fontsize,
            )

    def _add_colorbar(
        self,
        fig: plt.Figure,
        im: matplotlib.cm.ScalarMappable,
        cbar_ax: matplotlib.axes.Axes,
        label: str = "Temperature",
    ) -> None:
        """Add colorbar to the dedicated colorbar axis."""
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(label, fontsize=8)

    def _add_suptitle(
        self, fig: plt.Figure, mode: str, context_prop: float, model_ll: float
    ) -> None:
        """Add figure title."""
        suptitle_fontsize = max(12, min(12, self.figsize[0] * 0.5))
        title = f"ERA5 ({mode}) - context/all = {context_prop:.3f}, LogP = {model_ll:.3f}"
        fig.suptitle(title, fontsize=suptitle_fontsize, y=0.98)

    def _calculate_tick_positions(
        self, coords: np.ndarray | None, num_ticks: int | None
    ) -> np.ndarray | None:
        """Calculate tick positions for coordinate arrays."""
        if coords is None or num_ticks is None:
            return None
        coord_min, coord_max = coords.min(), coords.max()
        return np.linspace(coord_min, coord_max, num_ticks)

    def _setup_cartopy_features(
        self,
        ax: matplotlib.axes.Axes,
        lat_coords: np.ndarray | None = None,
        lon_coords: np.ndarray | None = None,
    ) -> None:
        """Add cartopy features to axis."""
        if not (self.use_cartopy and CARTOPY_AVAILABLE):
            return

        # With a cartopy projection the axis is a GeoAxes (add_feature/gridlines);
        # the annotated matplotlib Axes type doesn't carry those methods.
        geo_ax: Any = ax

        # Add geographic features
        geo_ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        geo_ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        geo_ax.add_feature(cfeature.OCEAN, alpha=0.3)
        geo_ax.add_feature(cfeature.LAND, alpha=0.3)

        if not (self.show_latitude or self.show_longitude):
            return

        # Add gridlines with calculated tick positions
        xticks = self._calculate_tick_positions(lon_coords, self.num_lon_ticks)
        yticks = self._calculate_tick_positions(lat_coords, self.num_lat_ticks)

        gl = geo_ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
            xlocs=xticks,
            ylocs=yticks,
        )
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {"size": self.lat_lon_fontsize if self.show_longitude else 0}
        gl.ylabel_style = {"size": self.lat_lon_fontsize if self.show_latitude else 0}
        gl.ylines = self.show_latitude
        gl.xlines = self.show_longitude

    def _add_coordinate_labels(
        self, ax: matplotlib.axes.Axes, lat_coords: np.ndarray, lon_coords: np.ndarray
    ) -> None:
        """Add coordinate labels for non-cartopy plots."""
        if not (self.show_latitude or self.show_longitude):
            return

        H, W = lat_coords.shape

        if self.show_longitude:
            x_positions = (
                list(range(W))
                if self.num_lon_ticks is None
                else np.linspace(0, W - 1, self.num_lon_ticks, dtype=int)
            )
            x_labels = [f"{lon_coords[H//2, pos]:.1f}°" for pos in x_positions]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=self.lat_lon_fontsize)
        else:
            ax.set_xticks([])

        if self.show_latitude:
            y_positions = (
                list(range(H))
                if self.num_lat_ticks is None
                else np.linspace(0, H - 1, self.num_lat_ticks, dtype=int)
            )
            y_labels = [f"{lat_coords[pos, W//2]:.1f}°" for pos in y_positions]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=self.lat_lon_fontsize)
        else:
            ax.set_yticks([])

    def _plot_spatial_data(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        title: str,
        vmin: float,
        vmax: float,
        cmap: str = "RdBu_r",
    ) -> matplotlib.cm.ScalarMappable:
        """Plot spatial data on given axis."""
        if self.use_cartopy and CARTOPY_AVAILABLE:
            # Use pcolormesh for cartopy
            im = ax.pcolormesh(
                lon_coords,
                lat_coords,
                data,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                shading="auto",
            )
        else:
            # Use imshow for regular plots
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
            # Add coordinate labels for non-cartopy plots
            self._add_coordinate_labels(ax, lat_coords, lon_coords)

        ax.set_title(title, fontsize=12)
        return im

    def _plot_context_with_non_context(
        self,
        ax: matplotlib.axes.Axes,
        context_data: np.ndarray,
        mc_mask: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        title: str,
        vmin: float,
        vmax: float,
        cmap: str = "RdBu_r",
    ) -> matplotlib.cm.ScalarMappable:
        """Plot context data with non-context points in distinct color."""
        # Create a masked array where non-context points are shown in gray
        # and context points use the original colormap

        # Create a composite image with context and non-context visualization
        display_data = np.full_like(context_data, np.nan)

        # Set context points to their actual values
        context_mask = mc_mask == 1
        display_data[context_mask] = context_data[context_mask]

        if self.use_cartopy and CARTOPY_AVAILABLE:
            # Plot context points with original colormap
            im = ax.pcolormesh(
                lon_coords,
                lat_coords,
                display_data,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                shading="auto",
            )

            # Plot non-context points in gray
            non_context_mask = mc_mask == 0
            if np.any(non_context_mask):
                non_context_data = np.full_like(context_data, np.nan)
                non_context_data[non_context_mask] = 0  # Use a neutral value for gray
                ax.pcolormesh(
                    lon_coords,
                    lat_coords,
                    non_context_data,
                    transform=ccrs.PlateCarree(),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                    alpha=0.5,
                    shading="auto",
                )
        else:
            # Use imshow for regular plots
            im = ax.imshow(
                display_data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper"
            )

            # Overlay non-context points in gray
            non_context_mask = mc_mask == 0
            if np.any(non_context_mask):
                non_context_overlay = np.ma.masked_where(
                    ~non_context_mask, np.ones_like(context_data)
                )
                ax.imshow(
                    non_context_overlay,
                    cmap="gray",
                    alpha=0.5,
                    vmin=0,
                    vmax=1,
                    origin="upper",
                )

            # Add coordinate labels for non-cartopy plots
            self._add_coordinate_labels(ax, lat_coords, lon_coords)

        ax.set_title(title, fontsize=12)
        return im

    def _plot_ground_truth_with_nan_mask(
        self,
        ax: matplotlib.axes.Axes,
        ground_truth_data: np.ndarray,
        m_mask: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        title: str,
        vmin: float,
        vmax: float,
        cmap: str = "RdBu_r",
    ) -> matplotlib.cm.ScalarMappable:
        """Plot ground truth data with NaN points (m_grid == 0) in distinct color."""
        # Create a masked array where NaN points are shown in red
        # and valid points use the original colormap

        # Create a composite image with valid and NaN visualization
        display_data = np.full_like(ground_truth_data, np.nan)

        # Set valid points to their actual values
        valid_mask = m_mask == 1
        display_data[valid_mask] = ground_truth_data[valid_mask]

        if self.use_cartopy and CARTOPY_AVAILABLE:
            # Plot valid points with original colormap
            im = ax.pcolormesh(
                lon_coords,
                lat_coords,
                display_data,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                shading="auto",
            )

            # Plot NaN points in red
            nan_mask = m_mask == 0
            if np.any(nan_mask):
                nan_data = np.full_like(ground_truth_data, np.nan)
                nan_data[nan_mask] = 0  # Use a neutral value for red color
                ax.pcolormesh(
                    lon_coords,
                    lat_coords,
                    nan_data,
                    transform=ccrs.PlateCarree(),
                    vmin=0,
                    vmax=1,
                    cmap="Reds",
                    alpha=0.8,
                    shading="auto",
                )
        else:
            # Use imshow for regular plots
            im = ax.imshow(
                display_data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper"
            )

            # Overlay NaN points in red
            nan_mask = m_mask == 0
            if np.any(nan_mask):
                nan_overlay = np.ma.masked_where(
                    ~nan_mask, np.ones_like(ground_truth_data)
                )
                ax.imshow(
                    nan_overlay, cmap="Reds", alpha=0.8, vmin=0, vmax=1, origin="upper"
                )

            # Add coordinate labels for non-cartopy plots
            self._add_coordinate_labels(ax, lat_coords, lon_coords)

        ax.set_title(title, fontsize=12)
        return im

    def _plot_spatial_single(
        self,
        yc: torch.Tensor,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        x_grid: torch.Tensor,
        mc_grid: torch.Tensor,
        m_grid: torch.Tensor,
        context_prop: float,
        model_ll: float,
        fig_path: Path,
    ) -> None:
        """Create a figure for spatial problem (single time step)."""

        # Denormalize coordinates and values
        x_grid_denorm = self._denormalize_coordinates(
            x_grid, has_time=False, has_elevation=self.x_has_surface_elevation
        )
        y_grid_denorm = y
        yc_grid_denorm = yc

        # Reshape mean and std to match y shape
        _, H, W = y.shape  # Remove channel dimension
        mean_grid = mean.view(1, H, W)
        std_grid = std.view(1, H, W)

        mean_grid_denorm = mean_grid
        std_grid_denorm = std_grid

        # Extract spatial coordinates
        lat_coords, lon_coords = self._extract_spatial_coordinates(x_grid_denorm)

        # Convert to numpy and take first channel
        y_np = y_grid_denorm[0].cpu().numpy()  # [H, W]
        yc_np = yc_grid_denorm[0].cpu().numpy()  # [H, W]
        mean_np = mean_grid_denorm[0].cpu().numpy()  # [H, W]
        std_np = std_grid_denorm[0].cpu().numpy()  # [H, W]
        mc_np = mc_grid[0, 0].cpu().numpy()  # [H, W]
        m_np = m_grid[0, 0].cpu().numpy()  # [H, W]

        # Create figure with new layout
        fig, axes, temp_cbar_ax, std_cbar_ax = self._create_spatial_subplot_layout(
            use_projection=self.use_cartopy
        )

        # Setup cartopy features if enabled
        for ax in axes:
            self._setup_cartopy_features(ax, lat_coords, lon_coords)

        # Calculate color limits
        vmin, vmax = self._calculate_color_limits(y_np)
        std_vmin, std_vmax = self._validate_model_output(std_np)

        # Plot data - special handling for ground truth (index 0) and context set (index 1)
        data_arrays = [y_np, yc_np, mean_np, std_np]
        vmins = [vmin, vmin, vmin, std_vmin]
        vmaxs = [vmax, vmax, vmax, std_vmax]

        y_values_im = None
        std_im = None
        for i, (data, cmap, vm_min, vm_max) in enumerate(
            zip(data_arrays, self.COLUMN_CMAPS, vmins, vmaxs)
        ):
            if (
                i == 0
            ):  # Ground truth column - show NaN points (m_grid == 0) with distinct color
                im = self._plot_ground_truth_with_nan_mask(
                    axes[i],
                    data,
                    m_np,  # Use m_np as m_grid mask
                    lat_coords,
                    lon_coords,
                    "",
                    vm_min,
                    vm_max,
                    cmap,
                )
                y_values_im = im  # Store for y_values colorbar
            elif (
                i == 1
            ):  # Context set column - show non-context points with distinct color
                im = self._plot_context_with_non_context(
                    axes[i],
                    data,
                    mc_np,
                    lat_coords,
                    lon_coords,
                    "",
                    vm_min,
                    vm_max,
                    cmap,
                )
            elif i == 2:  # Prediction mean column
                im = self._plot_spatial_data(
                    axes[i], data, lat_coords, lon_coords, "", vm_min, vm_max, cmap
                )
            else:  # i == 3, Standard deviation column
                im = self._plot_spatial_data(
                    axes[i], data, lat_coords, lon_coords, "", vm_min, vm_max, cmap
                )
                std_im = im  # Store for std colorbar
            self._remove_ticks(axes[i])

        # Add column titles
        self._add_column_titles(axes)

        # Add colorbar for temperature values (2nd and 3rd columns)
        if y_values_im is not None:
            self._add_colorbar(fig, y_values_im, temp_cbar_ax, label="Temperature")

        # Add colorbar for standard deviation (last column)
        if std_im is not None:
            self._add_colorbar(fig, std_im, std_cbar_ax, label="Standard Deviation")

        # Add suptitle
        self._add_suptitle(fig, "Spatial", context_prop, model_ll)

        self._handle_figure_output(
            fig,
            fig_path,
            savefig=self.savefig,
            logging=self.logging,
            show_plots=self.show_plots,
        )

        plt.close(fig)

    def _plot_spatiotemporal_subplots(
        self,
        yc_grid: torch.Tensor,
        y_grid: torch.Tensor,
        mean_flat: torch.Tensor,
        std_flat: torch.Tensor,
        x_grid: torch.Tensor,
        mc_grid: torch.Tensor,
        m_grid: torch.Tensor,
        selected_time_steps: list[int],
        context_prop: float,
        model_ll: float,
        fig_path: Path,
    ) -> None:
        """Create figure with subplots for multiple time steps."""

        # Denormalize coordinates and values
        x_grid_denorm = self._denormalize_coordinates(
            x_grid, has_time=True, has_elevation=self.x_has_surface_elevation
        )
        y_grid_denorm = y_grid
        yc_grid_denorm = yc_grid

        # Reshape mean_flat and std_flat to match y shape
        _, T, H, W = y_grid.shape  # Remove channel dimension
        mean_grid = mean_flat.view(1, T, H, W)
        std_grid = std_flat.view(1, T, H, W)

        mean_grid_denorm = mean_grid
        std_grid_denorm = std_grid

        num_time_steps = len(selected_time_steps)

        # Create figure with new layout
        fig, axes, temp_cbar_ax, std_cbar_ax = (
            self._create_spatiotemporal_subplot_layout(
                num_time_steps, use_projection=self.use_cartopy
            )
        )

        # Note: We'll setup cartopy features for each time step separately since coordinates may vary
        y_values_im = None
        std_im = None
        for i, time_step in enumerate(selected_time_steps):
            # Extract data for this time step
            y_grid_slice = y_grid_denorm[0, time_step].cpu().numpy()  # [H, W]
            yc_grid_slice = yc_grid_denorm[0, time_step].cpu().numpy()  # [H, W]
            mean_grid_slice = mean_grid_denorm[0, time_step].cpu().numpy()  # [H, W]
            std_grid_slice = std_grid_denorm[0, time_step].cpu().numpy()  # [H, W]
            mc_grid_slice = mc_grid[0, time_step].cpu().numpy()
            m_grid_slice = m_grid[0, time_step].cpu().numpy()

            # Extract spatial coordinates for this time step
            lat_coords, lon_coords = self._extract_spatial_coordinates(
                x_grid_denorm[:, time_step]
            )

            # Setup cartopy features for this time step's subplots
            for col in range(4):
                self._setup_cartopy_features(axes[i][col], lat_coords, lon_coords)

                # Calculate color limits for this time step
            vmin, vmax = self._calculate_color_limits(y_grid_slice)
            std_vmin, std_vmax = self._validate_model_output(std_grid_slice, time_step)

            # Plot data for this time step - special handling for context set (index 1)
            data_arrays = [y_grid_slice, yc_grid_slice, mean_grid_slice, std_grid_slice]
            vmins = [vmin, vmin, vmin, std_vmin]
            vmaxs = [vmax, vmax, vmax, std_vmax]

            for j, (data, cmap, vm_min, vm_max) in enumerate(
                zip(data_arrays, self.COLUMN_CMAPS, vmins, vmaxs)
            ):
                # Add column titles only for first row
                title = self.COLUMN_TITLES[j] if i == 0 else ""
                if (
                    j == 0
                ):  # Ground truth column - show NaN points (m_grid == 0) with distinct color
                    im = self._plot_ground_truth_with_nan_mask(
                        axes[i][j],
                        data,
                        m_grid_slice,
                        lat_coords,
                        lon_coords,
                        title,
                        vm_min,
                        vm_max,
                        cmap,
                    )
                    if i == 0:  # Store first y_values image for colorbar
                        y_values_im = im
                elif (
                    j == 1
                ):  # Context set column - show non-context points with distinct color
                    im = self._plot_context_with_non_context(
                        axes[i][j],
                        data,
                        mc_grid_slice,
                        lat_coords,
                        lon_coords,
                        title,
                        vm_min,
                        vm_max,
                        cmap,
                    )
                else:
                    im = self._plot_spatial_data(
                        axes[i][j],
                        data,
                        lat_coords,
                        lon_coords,
                        title,
                        vm_min,
                        vm_max,
                        cmap,
                    )
                    if j == 3 and i == 0:  # Store first std image for colorbar
                        std_im = im
                self._remove_ticks(axes[i][j])

            # Add time step label
            # Position further left when latitude is shown to avoid overlap with latitude ticks
            x_pos = -0.25 if self.show_latitude else -0.15
            axes[i][0].text(
                x_pos,
                0.5,
                f"t={time_step}",
                ha="center",
                va="center",
                transform=axes[i][0].transAxes,
                fontsize=10,
                rotation=90,
            )

        # Add colorbar for temperature values (2nd and 3rd columns)
        if y_values_im is not None:
            self._add_colorbar(fig, y_values_im, temp_cbar_ax, label="Temperature")

        # Add colorbar for standard deviation (last column)
        if std_im is not None:
            self._add_colorbar(fig, std_im, std_cbar_ax, label="Standard Deviation")

        # Add suptitle
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
        batches: list[ERA5Batch],
        name: str = "plot",
        **kwargs,
    ) -> None:
        """
        Generate and display/save plots for given batches.

        Args:
            model: The model used for predictions
            batches: list of data batches
            name: Name for saving figures
            **kwargs: Additional arguments to be passed to the model
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
            plot_batch.xq = einops.rearrange(
                eval_batch.x_grid, "b c ... -> b (...) c"
            )
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
            mean_flat = pred_dist_plot.mean.cpu()
            std_flat = pred_dist_plot.stddev.cpu()

            # Get tensors for plotting
            m_grid = plot_batch.m_grid.detach().cpu()
            y_grid = plot_batch.y_grid.detach().cpu()
            x_grid = plot_batch.x_grid.detach().cpu()

            mc_grid = plot_batch.mc_grid.detach().cpu()
            yc_grid = (
                torch.where(
                    plot_batch.y_mc_grid.detach().cpu() > 0,
                    plot_batch.y_grid.detach().cpu(),
                    torch.zeros_like(y_grid),
                )
            )

            # Detect problem mode
            problem_mode = self._detect_problem_mode(plot_batch)

            # Remove elevation dimension if present
            if self.x_has_surface_elevation:
                m_grid.squeeze_(dim=-3)
                y_grid.squeeze_(dim=-3)
                x_grid.squeeze_(dim=-3)
                mc_grid.squeeze_(dim=-3)
                yc_grid.squeeze_(dim=-3)

            # Calculate context proportion
            context_proportion = mc_grid[0].sum() / m_grid[0].numel()
            context_proportion = context_proportion.item()

            # Take first item in batch for plotting
            m_grid, y_grid, x_grid = m_grid[0], y_grid[0], x_grid[0]
            mc_grid, yc_grid = mc_grid[0], yc_grid[0]
            mean_flat, std_flat = mean_flat[0], std_flat[0]

            # Create plots based on problem mode
            fname = fig_dir / f"{i:03d}"

            if problem_mode == "spatial":
                self._plot_spatial_single(
                    yc_grid,
                    y_grid,
                    mean_flat,
                    std_flat,
                    x_grid,
                    mc_grid,
                    m_grid,
                    context_proportion,
                    float(model_ll),
                    fname,
                )
            elif problem_mode == "spatiotemporal":
                # Select which time steps to visualize
                _, time_steps = y_grid.shape[1], y_grid.shape[1]
                selected_time_steps = self._select_time_steps(time_steps)

                self._plot_spatiotemporal_subplots(
                    yc_grid,
                    y_grid,
                    mean_flat,
                    std_flat,
                    x_grid,
                    mc_grid,
                    m_grid,
                    selected_time_steps,
                    context_proportion,
                    float(model_ll),
                    fname,
                )
            else:
                raise ValueError(f"Unknown problem mode: {problem_mode}")
