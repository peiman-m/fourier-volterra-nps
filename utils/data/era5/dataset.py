import random
import warnings
from dataclasses import dataclass, replace

import einops
import numpy as np
import torch

from ..base import BaseBatch, BaseIterableDataset, EvalOn, as_batch
from .processor import ERA5DataProcessor


@dataclass
class ERA5Batch(BaseBatch):
    # Grid coordinates:
    # - use_time=True, use_surface_elevation=False: [B, 3, T, H, W]
    # - use_time=True, use_surface_elevation=True: [B, 4, T, D, H, W] (D=1)
    # - use_time=False, use_surface_elevation=False: [B, 2, H, W]
    # - use_time=False, use_surface_elevation=True: [B, 3, D, H, W] (D=1)

    x: torch.Tensor  # Flattened grid coordinates
    y: torch.Tensor  # Flattened output values

    xc: torch.Tensor  # Flattened context grid coordinates
    yc: torch.Tensor  # Flattened context output values

    xq: torch.Tensor  # Flattened query grid coordinates
    yq: torch.Tensor  # Flattened query output values

    x_grid: torch.Tensor
    y_grid: torch.Tensor  # same spatio-temporal shape as x_grid

    # Base masks with channel dim = 1, shared for both x and y grids
    m_grid: torch.Tensor  # [B, 1, ...] Non-missing mask (base)
    mc_grid: torch.Tensor  # [B, 1, ...] Context mask (base)
    mq_grid: torch.Tensor  # [B, 1, ...] Query mask (base)

    # Properties for accessing broadcasted masks
    @property
    def x_m_grid(self) -> torch.Tensor:
        """Non-missing mask broadcasted to x_grid shape"""
        return self.m_grid.expand_as(self.x_grid)

    @property
    def y_m_grid(self) -> torch.Tensor:
        """Non-missing mask broadcasted to y_grid shape"""
        return self.m_grid.expand_as(self.y_grid)

    @property
    def x_mc_grid(self) -> torch.Tensor:
        """Context mask broadcasted to x_grid shape"""
        return self.mc_grid.expand_as(self.x_grid)

    @property
    def y_mc_grid(self) -> torch.Tensor:
        """Context mask broadcasted to y_grid shape"""
        return self.mc_grid.expand_as(self.y_grid)

    @property
    def x_mq_grid(self) -> torch.Tensor:
        """Query mask broadcasted to x_grid shape"""
        return self.mq_grid.expand_as(self.x_grid)

    @property
    def y_mq_grid(self) -> torch.Tensor:
        """Query mask broadcasted to y_grid shape"""
        return self.mq_grid.expand_as(self.y_grid)


@as_batch.register(ERA5Batch)
def _as_batch_era5(batch: ERA5Batch, *, eval_on: EvalOn = "query") -> ERA5Batch:
    if eval_on == "query":
        return batch
    if eval_on == "context":
        return replace(
            batch,
            xq=batch.xc,
            yq=batch.yc,
            mq_grid=batch.mc_grid,
        )
    raise ValueError(f"Unsupported eval_on: {eval_on!r}")


class ERA5Dataset(BaseIterableDataset):
    """Data generator for ERA5 dataset with reproducible random sampling.

    This generator creates batches by randomly sampling spatial and temporal windows
    from ERA5 data, with configurable grid sizes and context/query point ratios.
    """

    # Maximum attempts for sampling
    MAX_GRID_ATTEMPTS = 10  # Maximum attempts to find a valid spatial grid
    MAX_TIME_ATTEMPTS = 10  # Maximum attempts to find valid time windows per grid

    def __init__(
        self,
        *,
        processor: ERA5DataProcessor,
        grid_size_range: tuple[int | tuple[int, int], ...],
        max_p_nan: float,  # Maximum proportion of missing values in sampled grid
        min_pc: float | None = None,
        max_pc: float | None = None,
        min_pq: float | None = None,
        max_pq: float | None = None,
        time_resolution: int = 1,
        use_time: bool = True,
        use_surface_elevation: bool = False,
        use_all_queries: bool = True,  # whether to use all remaining point (if any) as query points
        **kwargs,
    ):
        """Initialize ERA5Dataset.

        Args:
            processor: ERA5DataProcessor instance to sample from
            grid_size_range: Tuple specifying grid dimensions. Each element can be:
                - int: Fixed size for that dimension
                - tuple[int, int]: Range to randomly sample from
            max_p_nan: Maximum proportion of missing values allowed in sampled grids
            min_pc, max_pc: Min/max proportion of points to use as context
            min_pq, max_pq: Min/max proportion of points to use as query points
            time_resolution: Temporal stride for time dimension sampling
            use_time: Whether to include time dimension in sampling
            use_surface_elevation: Whether to include surface elevation in coordinates
            use_all_queries: If True, use all non-context points as query points
        """
        super().__init__(**kwargs)

        # Convert grid_size_range from OmegaConf ListConfig to tuple if needed
        if hasattr(grid_size_range, "__iter__") and not isinstance(
            grid_size_range, tuple
        ):
            grid_size_range = tuple(
                (
                    tuple(dim)
                    if hasattr(dim, "__iter__") and not isinstance(dim, (int, str))
                    else dim
                )
                for dim in grid_size_range
            )

        # Set defaults
        self.max_p_nan = min(1.0, max(max_p_nan, 0.0))
        self.min_pc = min(1.0, min_pc or 0.0)
        self.max_pc = min(max_pc or 1.0, 1.0)
        self.min_pq = min(1.0, min_pq or 0.0)
        self.max_pq = min(max_pq or 1.0, 1.0)
        self.use_all_queries = use_all_queries

        # Validate that we can allocate minimum points
        if self.min_pc + self.min_pq > 1.0:
            raise ValueError(
                "Sum of minimum pixel proportions ({self.min_pc + self.min_pq}) exceeds 1.0"
            )

        # Warn if use_all_queries is True but query bounds are specified
        if self.use_all_queries and (min_pq is not None or max_pq is not None):
            warnings.warn(
                "Query point proportions (min_pq, max_pq) "
                "are specified but will be ignored because "
                "use_all_queries=True. All non-context points "
                "will be used as query points."
            )

        # Inputs/outputs setup
        if not use_time:
            if len(grid_size_range) != 2:
                raise ValueError("grid_size_range must be (H, W) if use_time is False")
            if use_surface_elevation:
                self.input_vars = ["surface_elevation", "latitude", "longitude"]
            else:
                self.input_vars = ["latitude", "longitude"]
        else:
            if len(grid_size_range) != 3:
                raise ValueError(
                    "grid_size_range must be (T, H, W) if use_time is True"
                )
            if use_surface_elevation:
                self.input_vars = [
                    "numerical_time",
                    "surface_elevation",
                    "latitude",
                    "longitude",
                ]
            else:
                self.input_vars = ["numerical_time", "latitude", "longitude"]

        self.output_vars: list[str] = ["t2m"]

        # How large each sampled grid should be (in indicies).
        self.grid_size_range = grid_size_range

        # Validate grid_size_range
        self._validate_grid_size_range()

        # Store max grid size for compatibility checks
        self._max_grid_size = tuple(
            max_val if isinstance(dim, tuple) else dim
            for dim, max_val in zip(
                grid_size_range,
                [
                    max(dim) if isinstance(dim, tuple) else dim
                    for dim in grid_size_range
                ],
            )
        )
        self.time_resolution = time_resolution
        if self.time_resolution < 1:
            raise ValueError("time_resolution must be >= 1")
        self.use_time = use_time
        self.use_surface_elevation = use_surface_elevation

        # --- Validate using processor BEFORE dropping it ---
        if use_surface_elevation and "surface_elevation" not in processor.data.variables:
            raise ValueError(
                "Surface elevation 'surface_elevation' not found in dataset. "
                "Ensure 'geopotential' is included in variables."
            )

        # --- Export numpy arrays and load coords/paths ---
        numpy_paths = processor.export_numpy()

        # Load coordinates as plain numpy arrays (small, stays in memory)
        self._coords: dict[str, np.ndarray] = {}
        coord_names = ["numerical_time", "latitude", "longitude"]
        if "surface_elevation" in numpy_paths:
            coord_names.append("surface_elevation")
        for name in coord_names:
            self._coords[name] = np.load(numpy_paths[name])

        # Store data variable paths (pickle-safe strings, not mmap objects)
        self._data_var_paths: dict[str, str] = {
            var: numpy_paths[var] for var in self.output_vars
        }

        # Lazy mmap cache — populated on first access per worker
        self._mmap_cache: dict[str, np.ndarray] = {}

        # Cache dimension sizes from coords
        self._n_lat = len(self._coords["latitude"])
        self._n_lon = len(self._coords["longitude"])
        self._n_time = len(self._coords["numerical_time"])

        # Print subset/size info before dropping the processor
        print(
            f'[ERA5Dataset] [{processor.subset}] '
            f'n_time={self._n_time}, n_lat={self._n_lat}, n_lon={self._n_lon}, '
            f'grid_size_range={self.grid_size_range}'
        )

        # Drop processor reference — no longer needed
        del processor

        # Quick sanity on grid fit using max sizes
        H = self._max_grid_size[-2]
        W = self._max_grid_size[-1]
        if H > self._n_lat or W > self._n_lon:
            raise ValueError(
                f"max grid_size_range (H={H}, W={W}) does not fit in dataset "
                f"(lat={self._n_lat}, lon={self._n_lon})"
            )
        if self.use_time:
            T = self._max_grid_size[0]
            needed = self.time_resolution * T
            if needed > self._n_time:
                raise ValueError(
                    f"Requested time window of length {needed} "
                    f"(T={T}, res={self.time_resolution}) "
                    f"exceeds available time steps ({self._n_time})"
                )

    def _validate_grid_size_range(self) -> None:
        """Validate grid_size_range parameter"""
        for i, dim in enumerate(self.grid_size_range):
            if isinstance(dim, tuple):
                if len(dim) != 2:
                    raise ValueError(
                        f"grid_size_range dimension {i}: "
                        f"range must be a 2-tuple, got {dim}"
                    )
                min_val, max_val = dim
                if not isinstance(min_val, int) or not isinstance(max_val, int):
                    raise ValueError(
                        f"grid_size_range dimension {i}: "
                        f"range values must be integers, got {dim}"
                    )
                if min_val < 1:
                    raise ValueError(
                        f"grid_size_range dimension {i}: "
                        f"minimum grid size must be >= 1, got {min_val}"
                    )
                if min_val > max_val:
                    raise ValueError(
                        f"grid_size_range dimension {i}: "
                        f"min ({min_val}) must be <= max ({max_val})"
                    )
            elif isinstance(dim, int):
                if dim < 1:
                    raise ValueError(
                        f"grid_size_range dimension {i}: "
                        f"grid size must be >= 1, got {dim}"
                    )
            else:
                raise ValueError(
                    f"grid_size_range dimension {i}: "
                    f"must be int or tuple[int, int], got {type(dim)}"
                )

    def _sample_grid_size(self) -> tuple[int, ...]:
        """Sample actual grid size from grid_size_range"""
        sampled_size = []
        for dim in self.grid_size_range:
            if isinstance(dim, tuple):
                min_val, max_val = dim
                sampled_size.append(self._randint(min_val, max_val))
            else:
                sampled_size.append(dim)
        return tuple(sampled_size)

    def _get_data_var(self, var: str) -> np.ndarray:
        """Lazily open a memory-mapped numpy array for a data variable.

        Each DataLoader worker process gets its own _mmap_cache dict,
        so no locking is needed. Fancy indexing on the returned memmap
        copies selected data into a regular np.ndarray.
        """
        if var not in self._mmap_cache:
            self._mmap_cache[var] = np.load(
                self._data_var_paths[var], mmap_mode="r"
            )
        return self._mmap_cache[var]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}  # Don't pickle mmap objects — just paths
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        # _mmap_cache is empty; _get_data_var() will lazily re-open per-worker

    def generate_batch(self) -> ERA5Batch:
        idxs = self._sample_grid(batch_size=self.batch_size)
        return self.sample_batch(idxs=idxs)

    def _randint(self, low: int, high: int) -> int:
        # inclusive low, inclusive high via random.randint semantics
        if high < low:
            raise ValueError(f"_randint got invalid range [{low}, {high}]")
        # random.randint is inclusive-high; add 1
        return random.randint(low, high)

    def _sample_grid(
        self, batch_size: int
    ) -> list[tuple[list[int] | int, list[int], list[int]]]:
        """Return list of (time_window, lat_indices, lon_indices)
        for each batch item."""
        # Sample a single grid size for the entire batch to maintain consistency
        grid_size = self._sample_grid_size()
        H = grid_size[-2]
        W = grid_size[-1]

        for _ in range(self.MAX_GRID_ATTEMPTS):
            # Spatial window
            h0 = self._randint(0, self._n_lat - H)
            w0 = self._randint(0, self._n_lon - W)
            lat_idx = list(range(h0, h0 + H))
            lon_idx = list(range(w0, w0 + W))

            # Reference NaN pattern for this spatial patch (across a time slice)
            ref_nan_mask = self._get_grid_nan_mask(
                lat_idx=lat_idx, lon_idx=lon_idx, time_idx=None
            )
            ref_nan_count = int(ref_nan_mask.sum().item())
            p_nan = ref_nan_count / float(ref_nan_mask.numel())
            if p_nan >= self.max_p_nan:
                continue

            # Collect per-item time windows with the same NaN count
            chosen: dict[int, list[int] | int] = {}
            for _ in range(batch_size):
                found = False
                for _ in range(self.MAX_TIME_ATTEMPTS):
                    if self.use_time:
                        T = grid_size[0]
                        high = self._n_time - self.time_resolution * T - 1
                        t0 = self._randint(0, high)
                        t_window = list(
                            range(
                                t0, t0 + self.time_resolution * T, self.time_resolution
                            )
                        )
                        dedup_key = t0  # ensure unique start times
                    else:
                        t0 = self._randint(0, self._n_time - 1)
                        t_window = t0
                        dedup_key = t0

                    if dedup_key in chosen:
                        continue

                    cur_nan_mask = self._get_grid_nan_mask(
                        lat_idx=lat_idx, lon_idx=lon_idx, time_idx=t_window
                    )
                    if int(cur_nan_mask.sum().item()) == ref_nan_count:
                        chosen[dedup_key] = t_window
                        found = True
                        break

                if not found:
                    break

            if len(chosen) == batch_size:
                return [(tw, lat_idx, lon_idx) for _, tw in chosen.items()]

        raise RuntimeError(
            "Failed to find valid spatial grid with consistent NaN "
            f"patterns after {self.MAX_GRID_ATTEMPTS} attempts."
        )

    def sample_batch(
        self, idxs: list[tuple[list[int] | int, list[int], list[int]]]
    ) -> ERA5Batch:
        # Will build tensors from these later.
        x_batch: list[torch.Tensor] = []
        y_batch: list[torch.Tensor] = []

        xc_batch: list[torch.Tensor] = []
        yc_batch: list[torch.Tensor] = []

        xq_batch: list[torch.Tensor] = []
        yq_batch: list[torch.Tensor] = []

        x_grid_batch: list[torch.Tensor] = []
        y_grid_batch: list[torch.Tensor] = []

        mc_grid_batch: list[torch.Tensor] = []
        mq_grid_batch: list[torch.Tensor] = []
        m_grid_batch: list[torch.Tensor] = []

        nc, nq = None, None

        # TODO: can we batch this?
        for time_idx, lat_idx, lon_idx in idxs:
            # ----------------- Build coordinate grid -----------------
            x_grid = self._build_x_grid(
                time_idx, lat_idx, lon_idx
            )  # [Cx, (T,), (D=1,) H W]
            # ----------------- Build output grid ---------------------
            y_grid = self._fetch_y_grid(
                time_idx, lat_idx, lon_idx
            )  # [Cy, (T,), (D=1,) H W]

            # Base non-missing mask (True = valid)
            y_mask_grid = ~torch.isnan(y_grid).any(dim=0, keepdim=True)  # [1, ...]
            valid_mask_flat = y_mask_grid.flatten()
            n_valid = int(valid_mask_flat.sum().item())
            if n_valid == 0:
                raise RuntimeError(
                    "Sampled patch has no valid (non-NaN) outputs after masking."
                )

            # ----------------- Sample counts once per batch -----------------
            if nc is None and nq is None:
                nc, nq = self._sample_point_counts(n_valid)
                if (
                    nc is None
                    or nq is None
                    or nc <= 0
                    or nq <= 0
                    or (nc + nq) > n_valid
                ):
                    raise ValueError(
                        "Invalid context/query counts from _sample_point_counts: "
                        f"nc={nc}, nq={nq}, n_valid={n_valid}"
                    )

            # ----------------- Draw indices -----------------
            # NOTE: sampled once per item with the same nc/nq to keep batch shape uniform
            # nc/nq are set on the first item and reused thereafter.
            assert nc is not None and nq is not None
            valid_idx_flat = torch.where(valid_mask_flat)[0]
            perm = torch.randperm(len(valid_idx_flat))
            valid_idx_flat = valid_idx_flat[perm]

            # m_idx_flat  = valid_idx_flat[:nc + nq]
            m_idx_flat = valid_idx_flat
            mc_idx_flat = valid_idx_flat[:nc]
            mq_idx_flat = valid_idx_flat[nc : nc + nq]

            # Unravel into grid indices (drop channel)
            unravel_shape = y_grid.shape[1:]  # (..., possibly T or D)
            m_idx_grid = torch.unravel_index(m_idx_flat, unravel_shape)
            mc_idx_grid = torch.unravel_index(mc_idx_flat, unravel_shape)
            mq_idx_grid = torch.unravel_index(mq_idx_flat, unravel_shape)

            # Gather flattened samples along the last N dims
            x = x_grid[:, *m_idx_grid]
            y = y_grid[:, *m_idx_grid]

            xc = x_grid[:, *mc_idx_grid]
            yc = y_grid[:, *mc_idx_grid]

            xq = x_grid[:, *mq_idx_grid]
            yq = y_grid[:, *mq_idx_grid]

            # Base masks (channel dim = 1)
            m_grid = torch.zeros_like(y_grid[[0]])
            mc_grid = torch.zeros_like(y_grid[[0]])
            mq_grid = torch.zeros_like(y_grid[[0]])
            m_grid[:, *m_idx_grid] = True
            mc_grid[:, *mc_idx_grid] = True
            mq_grid[:, *mq_idx_grid] = True

            # Fill NaNs in y (they're masked anyway)
            # y_grid = torch.nan_to_num(y_grid, nan=-9999.99)

            # Collect
            x_batch.append(x)
            y_batch.append(y)

            xc_batch.append(xc)
            yc_batch.append(yc)

            xq_batch.append(xq)
            yq_batch.append(yq)

            x_grid_batch.append(x_grid)
            y_grid_batch.append(y_grid)

            mc_grid_batch.append(mc_grid)
            mq_grid_batch.append(mq_grid)
            m_grid_batch.append(m_grid)

        x = torch.stack(x_batch).movedim(1, -1)
        y = torch.stack(y_batch).movedim(1, -1)

        xc = torch.stack(xc_batch).movedim(1, -1)
        yc = torch.stack(yc_batch).movedim(1, -1)

        xq = torch.stack(xq_batch).movedim(1, -1)
        yq = torch.stack(yq_batch).movedim(1, -1)

        x_grid = torch.stack(x_grid_batch)
        y_grid = torch.stack(y_grid_batch)

        mc_grid = torch.stack(mc_grid_batch)
        mq_grid = torch.stack(mq_grid_batch)
        m_grid = torch.stack(m_grid_batch)

        return ERA5Batch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xq=xq,
            yq=yq,
            x_grid=x_grid,
            y_grid=y_grid,
            m_grid=m_grid,
            mc_grid=mc_grid,
            mq_grid=mq_grid,
        )

    def _build_x_grid(
        self,
        time_idx: list[int] | int,
        lat_idx: list[int],
        lon_idx: list[int],
    ) -> torch.Tensor:
        """Assemble coordinate channels into a grid tensor"""
        coords: list[torch.Tensor] = []
        use_time = self.use_time
        T = (len(time_idx) if isinstance(time_idx, list) else 1) if use_time else None
        H, W = len(lat_idx), len(lon_idx)

        # numerical_time
        if "numerical_time" in self.input_vars:
            t_vals = torch.as_tensor(
                self._coords["numerical_time"][time_idx], dtype=torch.float32
            )
            if isinstance(time_idx, list):
                t_ch = einops.repeat(t_vals, "T -> T H W", H=H, W=W)
            else:
                # Single index but still broadcast to [T=1,H,W]
                t_ch = einops.repeat(t_vals, "-> 1 H W", H=H, W=W)
            coords.append(t_ch)

        # surface_elevation (lat,lon) — 2D orthogonal indexing
        if "surface_elevation" in self.input_vars:
            se_vals = torch.as_tensor(
                self._coords["surface_elevation"][np.ix_(lat_idx, lon_idx)],
                dtype=torch.float32,
            )
            if self.use_time:
                se_vals = einops.repeat(se_vals, "H W -> T H W", T=(T or 1))
            coords.append(se_vals)

        # latitude — 1D
        if "latitude" in self.input_vars:
            lat_vals = torch.as_tensor(
                self._coords["latitude"][lat_idx],
                dtype=torch.float32,
            )
            if use_time:
                lat_ch = einops.repeat(lat_vals, "H -> T H W", T=(T or 1), W=W)
            else:
                lat_ch = einops.repeat(lat_vals, "H -> H W", W=W)
            coords.append(lat_ch)

        # longitude — 1D
        if "longitude" in self.input_vars:
            lon_vals = torch.as_tensor(
                self._coords["longitude"][lon_idx],
                dtype=torch.float32,
            )
            if use_time:
                lon_ch = einops.repeat(lon_vals, "W -> T H W", T=(T or 1), H=H)
            else:
                lon_ch = einops.repeat(lon_vals, "W -> H W", H=H)
            coords.append(lon_ch)

        x_grid = torch.stack(coords, dim=0)  # [Cx, (T,), H, W]
        if self.use_surface_elevation:
            # Insert D=1 axis after time
            if use_time:
                x_grid = einops.rearrange(x_grid, "C T H W -> C T 1 H W")
            else:
                x_grid = einops.rearrange(
                    x_grid, "C H W -> C 1 H W"
                )  # keep a D axis even w/o time
        return x_grid

    def _fetch_y_grid(
        self,
        time_idx: list[int] | int,
        lat_idx: list[int],
        lon_idx: list[int],
    ) -> torch.Tensor:
        vals = []
        for var in self.output_vars:
            data_var = self._get_data_var(var)
            if isinstance(time_idx, list):
                raw = data_var[np.ix_(time_idx, lat_idx, lon_idx)]  # (T, H, W)
            else:
                raw = data_var[time_idx][np.ix_(lat_idx, lon_idx)]  # (H, W)
            arr = torch.as_tensor(raw, dtype=torch.float32)
            vals.append(arr)
        y_grid = torch.stack(vals, dim=0)  # [Cy, (T,), H, W]
        if self.use_surface_elevation:
            # Match the extra D=1 axis convention used in x_grid
            if self.use_time:
                y_grid = einops.rearrange(y_grid, "C T H W -> C T 1 H W")
            else:
                y_grid = einops.rearrange(y_grid, "C H W -> C 1 H W")
        return y_grid

    def _get_grid_nan_mask(
        self,
        lat_idx: list[int],
        lon_idx: list[int],
        time_idx: int | list[int] | None = None,
    ) -> torch.Tensor:
        t_idx: list[int] | int = [0] if time_idx is None else time_idx
        vals = []
        for var in self.output_vars:
            data_var = self._get_data_var(var)
            if isinstance(t_idx, list):
                raw = data_var[np.ix_(t_idx, lat_idx, lon_idx)]  # (T, H, W)
            else:
                raw = data_var[t_idx][np.ix_(lat_idx, lon_idx)]  # (H, W)
            arr = torch.as_tensor(raw, dtype=torch.float32)
            vals.append(arr)
        y = torch.stack(vals, dim=0)  # [Cy, ..., H, W]
        return torch.isnan(y).any(dim=0)  # True where missing

    def _sample_point_counts(
        self,
        n_total: int,
    ) -> tuple[int, int]:
        """Sample (nc, nq) given total valid points."""
        # If using all remaining points as query points,
        # we only need to sample context points
        if self.use_all_queries:
            min_nc = max(int(self.min_pc * n_total), 1)
            max_nc = (
                min(int(self.max_pc * n_total), int(n_total * (1 - self.min_pq))) + 1
            )
            if max_nc <= min_nc:
                # Fallback to at least 1 context if possible
                max_nc = max(min_nc + 1, min(n_total, min_nc + 1))
            nc = int(torch.randint(low=min_nc, high=max_nc, size=()).item())
            # Use all remaining points as query points
            nq = n_total - nc
            return nc, nq

        # Sample total used points n first
        min_p = self.min_pc + self.min_pq
        max_p = min(self.max_pc + self.max_pq, 1.0)
        min_n = max(int(n_total * min_p), 1)
        max_n = min(int(n_total * max_p), n_total) + 1
        if max_n <= min_n:
            max_n = min_n + 1
        n = int(torch.randint(low=min_n, high=max_n, size=()).item())

        # Then sample context portion
        min_nc = max(int(self.min_pc * n_total), 1)
        max_nc = min(int(self.max_pc * n_total), n - int(self.min_pq * n_total)) + 1
        if max_nc <= min_nc:
            max_nc = min_nc + 1
        nc = int(torch.randint(low=min_nc, high=max_nc, size=()).item())
        nq = n - nc
        return nc, nq
