import warnings
from dataclasses import dataclass, replace

import einops
import torch

from ..base import BaseBatch, BaseMapDataset, EvalOn, as_batch
from .processor import KolmogorovDataProcessor
from .transforms import KolmogorovCropStrategy


@dataclass
class KolmogorovBatch(BaseBatch):
    x: torch.Tensor  # Flattened grid coordinates [B, N, D]
                     # where D=2 (spatial) or 3 (spatio-temporal)
    w: torch.Tensor  # Flattened vorticity values [B, N, 1]
    y: torch.Tensor  # Flattened trajectory values [B, N, C]

    xc: torch.Tensor  # Context grid coordinates [B, Nc, D]
    wc: torch.Tensor  # Context vorticity values [B, Nc, 1]
    yc: torch.Tensor  # Context trajectory values [B, Nc, C]

    xq: torch.Tensor  # Query grid coordinates [B, Nq, D]
    wq: torch.Tensor  # Query vorticity values [B, Nq, 1]
    yq: torch.Tensor  # Query trajectory values [B, Nq, C]

    x_grid: torch.Tensor  # Grid coordinates [B, D, ...]
                          # where ... = (H, W) or (T, H, W)
    w_grid: torch.Tensor  # Vorticity values [B, 1, ...]
    y_grid: torch.Tensor  # Trajectory values [B, C, ...]

    # Base masks with channel dim = 1, shared for both x, w, and y grids
    mc_grid: torch.Tensor  # [B, 1, ...] Context mask (base)
    mq_grid: torch.Tensor  # [B, 1, ...] Query mask (base)

    # Properties for accessing broadcasted masks
    @property
    def x_mc_grid(self) -> torch.Tensor:
        """Context mask broadcasted to x_grid shape"""
        return self.mc_grid.expand_as(self.x_grid)

    @property
    def w_mc_grid(self) -> torch.Tensor:
        """Context mask broadcasted to w_grid shape"""
        return self.mc_grid.expand_as(self.w_grid)

    @property
    def y_mc_grid(self) -> torch.Tensor:
        """Context mask broadcasted to y_grid shape"""
        return self.mc_grid.expand_as(self.y_grid)

    @property
    def x_mq_grid(self) -> torch.Tensor:
        """Query mask broadcasted to x_grid shape"""
        return self.mq_grid.expand_as(self.x_grid)

    @property
    def w_mq_grid(self) -> torch.Tensor:
        """Query mask broadcasted to w_grid shape"""
        return self.mq_grid.expand_as(self.w_grid)

    @property
    def y_mq_grid(self) -> torch.Tensor:
        """Query mask broadcasted to y_grid shape"""
        return self.mq_grid.expand_as(self.y_grid)


@as_batch.register(KolmogorovBatch)
def _as_batch_kolmogorov(
    batch: KolmogorovBatch, *, eval_on: EvalOn = "query"
) -> KolmogorovBatch:
    if eval_on == "query":
        return batch
    if eval_on == "context":
        return replace(
            batch,
            xq=batch.xc,
            yq=batch.yc,
            wq=batch.wc,
            mq_grid=batch.mc_grid,
        )
    raise ValueError(f"Unsupported eval_on: {eval_on!r}")


class KolmogorovDataset(BaseMapDataset):
    """
    DataLoader for Kolmogorov data with random context/query point sampling.
    We assume that all samples in a dataset have similar size.
    """

    def __init__(
        self,
        *,
        processor: KolmogorovDataProcessor,
        crop_strategy: KolmogorovCropStrategy,
        min_pc: float | None = None,
        max_pc: float | None = None,
        min_pq: float | None = None,
        max_pq: float | None = None,
        samples_per_epoch: int | None = None,
        use_all_queries: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the Kolmogorov data loader.

        Args:
            processor: DataProcessor containing Kolmogorov data
            crop_strategy: Strategy for cropping samples (random or strided tiling)
            min_pc: Minimum proportion of context points
            max_pc: Maximum proportion of context points
            min_pq: Minimum proportion of query points
            max_pq: Maximum proportion of query points
            use_all_queries: If True, use all remaining points as query points
            samples_per_epoch: Number of samples per epoch
            random_state: If set, pins crop positions and context/query splits
                per index for reproducible validation across runs and models.
        """
        # Extract only the data we need (don't keep processor reference)
        # This avoids memory duplication when DataLoader uses multiprocessing
        self.trajectories = processor.subset_trajectories  # [N_subset, C, T, H, W]
        self.vorticities = processor.subset_vorticities    # [N_subset, 1, T, H, W]
        self.grid = processor.grid                          # [3, T, H, W]
        self.mode = processor.mode
        self.crop_strategy = crop_strategy

        N, _, T, H, W = self.trajectories.shape

        # Compute dataset length based on mode, accounting for tiling
        if self.mode == "spatial":
            self._tiles_per_sample = self.crop_strategy.crops_per_sample(H, W)
            total_samples = N * T * self._tiles_per_sample
        else:
            self._tiles_per_sample = self.crop_strategy.crops_per_sample(T, H, W)
            total_samples = N * self._tiles_per_sample
        total_samples = int(total_samples)  # product of integer counts

        if total_samples == 0:
            if self.mode == "spatial":
                raise ValueError(
                    f"total_samples == 0: N={N}, T={T}, "
                    f"tiles_per_sample={self._tiles_per_sample}. "
                    "Check crop/stride parameters or ensure data is not empty."
                )
            else:
                raise ValueError(
                    f"total_samples == 0: N={N}, "
                    f"tiles_per_sample={self._tiles_per_sample}. "
                    "Check crop/stride parameters or ensure data is not empty."
                )

        # Validate and set samples per epoch
        samples_per_epoch = samples_per_epoch or total_samples
        if samples_per_epoch > total_samples:
            warnings.warn(
                f"Requested samples_per_epoch ({samples_per_epoch}) "
                f"exceeds total samples ({total_samples}). Adjusting "
                f"to {total_samples}."
            )
        samples_per_epoch = min(max(1, samples_per_epoch), total_samples)

        print(f'[{type(self).__name__}] [{self.mode}/{processor.subset}] Total samples: {total_samples}')
        print(f'[{type(self).__name__}] [{self.mode}/{processor.subset}] Samples per epoch: {samples_per_epoch}')

        # Proportion-based parameters
        self.min_pc = min(1.0, min_pc or 0.0)
        self.max_pc = min(max_pc or 1.0, 1.0)
        self.min_pq = min(1.0, min_pq or 0.0)
        self.max_pq = min(max_pq or 1.0, 1.0)

        # Validate that we can allocate minimum points
        if self.min_pc + self.min_pq > 1.0:
            raise ValueError(
                f"Sum of minimum pixel proportions ({self.min_pc + self.min_pq}) exceeds 1.0"
            )

        # Warn if use_all_queries is True but query bounds are specified
        if use_all_queries and (min_pq is not None or max_pq is not None):
            warnings.warn(
                "Query point proportions (min_pq, max_pq) "
                "are specified but will be ignored because "
                "use_all_queries=True. All non-context pixels "
                "will be used as query points."
            )

        self.use_all_queries = use_all_queries
        self._random_state = random_state

        # Initialize parent class
        super().__init__(
            samples_per_epoch=samples_per_epoch,
        )

    def _sample_point_counts(
        self, n_total: int, generator: torch.Generator | None = None
    ) -> tuple[int, int]:
        """Sample number of context and query points using proportion-based logic.

        Args:
            n_total: Total number of points.
            generator: Optional local RNG generator (keeps global state clean).

        Returns:
            Tuple of (context_points, query_points).
        """
        if self.use_all_queries:
            min_nc = max(int(self.min_pc * n_total), 1)
            max_nc = (
                min(int(self.max_pc * n_total), int(n_total * (1 - self.min_pq))) + 1
            )
            if max_nc <= min_nc:
                max_nc = max(min_nc + 1, min(n_total, min_nc + 1))
            nc = int(torch.randint(low=min_nc, high=max_nc, size=(), generator=generator).item())
            nq = n_total - nc
            return nc, nq

        min_p = self.min_pc + self.min_pq
        max_p = min(self.max_pc + self.max_pq, 1.0)

        min_n = max(int(n_total * min_p), 1)
        max_n = min(int(n_total * max_p), n_total) + 1

        if max_n <= min_n:
            max_n = min_n + 1
        n = int(torch.randint(low=min_n, high=max_n, size=(), generator=generator).item())

        min_nc = max(int(self.min_pc * n_total), 1)
        max_nc = min(int(self.max_pc * n_total), n - int(self.min_pq * n_total)) + 1
        if max_nc <= min_nc:
            max_nc = min_nc + 1
        nc = int(torch.randint(low=min_nc, high=max_nc, size=(), generator=generator).item())
        nq = n - nc
        return nc, nq

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return raw Kolmogorov data for one sample.

        Args:
            idx: Sample index into the dataset.

        Returns:
            Dict with 'trajectory', 'vorticity', and 'grid' tensors.
        """
        _, _, T, H, W = self.trajectories.shape

        rng_ctx = (
            torch.random.fork_rng(devices=[])
            if self._random_state is not None
            else torch.random.fork_rng(enabled=False)
        )
        with rng_ctx:
            if self._random_state is not None:
                torch.manual_seed(self._random_state + idx)

            if self.mode == "spatial":
                traj_time_idx  = idx // self._tiles_per_sample
                tile_local_idx = idx %  self._tiles_per_sample
                traj_idx = traj_time_idx // T
                time_idx = traj_time_idx %  T
                h0, ch, w0, cw = self.crop_strategy.get_crop_coords(tile_local_idx, H, W)
                return {
                    "trajectory": self.trajectories[traj_idx, :, time_idx, h0:h0 + ch, w0:w0 + cw],
                    "vorticity":  self.vorticities[traj_idx,  :, time_idx, h0:h0 + ch, w0:w0 + cw],
                    "grid":       self.grid[1:,                  time_idx,  h0:h0 + ch, w0:w0 + cw],
                    **( {"_idx": idx} if self._random_state is not None else {} ),
                }

            traj_idx  = idx // self._tiles_per_sample
            local_idx = idx %  self._tiles_per_sample
            t0, ct, h0, ch, w0, cw = self.crop_strategy.get_crop_coords(local_idx, T, H, W)
            return {
                "trajectory": self.trajectories[traj_idx, :, t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
                "vorticity":  self.vorticities[traj_idx,  :, t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
                "grid":       self.grid[:,                   t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
                **( {"_idx": idx} if self._random_state is not None else {} ),
            }

    def collate_fn(self, samples: list[dict[str, torch.Tensor]]) -> KolmogorovBatch:
        """Collate Kolmogorov samples into a batch with context/query splitting.

        Args:
            samples: List of dicts with 'trajectory', 'vorticity', 'grid' tensors.

        Returns:
            KolmogorovBatch with context/query splits.
        """
        # Batch-level generator: seeds nc/nq from the first sample's index so all
        # samples in the batch share the same context/query counts (uniform shapes).
        # Offset by samples_per_epoch so this seed never aliases per-sample permutation
        # seeds (which are in [random_state, random_state + samples_per_epoch)).
        batch_gen: torch.Generator | None = None
        if self._random_state is not None:
            batch_gen = torch.Generator()
            batch_gen.manual_seed(int(self._random_state + self.samples_per_epoch + samples[0]["_idx"]))

        x_grid = torch.stack([s["grid"] for s in samples], dim=0)
        w_grid = torch.stack([s["vorticity"] for s in samples], dim=0)
        y_grid = torch.stack([s["trajectory"] for s in samples], dim=0)

        if self.mode == "spatio-temporal":
            batch_size, num_channels, steps, height, width = y_grid.shape
            total_points = steps * height * width
        elif self.mode == "spatial":
            batch_size, num_channels, height, width = y_grid.shape
            total_points = height * width
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        # Sample point counts (batch-level: same nc/nq for all samples)
        nc, nq = self._sample_point_counts(total_points, generator=batch_gen)

        # Flatten for easier indexing
        x = einops.rearrange(x_grid, "b d ... -> b (...) d")
        w = einops.rearrange(w_grid, "b c ... -> b (...) c")
        y = einops.rearrange(y_grid, "b c ... -> b (...) c")

        # Per-sample permutations: each seeded from its own _idx so that a sample's
        # context/query assignment is the same regardless of batch composition or DDP rank.
        # Offset by 2*samples_per_epoch to avoid aliasing with crop seeds (offset 0)
        # and nc/nq seeds (offset samples_per_epoch).
        if self._random_state is not None:
            rows = []
            for s in samples:
                sg = torch.Generator()
                sg.manual_seed(self._random_state + 2 * self.samples_per_epoch + int(s["_idx"]))
                rows.append(torch.rand(total_points, dtype=torch.float64, generator=sg))
            randperm = torch.stack(rows).argsort(dim=-1)
        else:
            randperm = torch.rand(batch_size, total_points, dtype=torch.float64).argsort(dim=-1)

        # Select disjoint indices
        idx_c = randperm[:, :nc]
        idx_q = randperm[:, nc : nc + nq]

        # Sort so mask-based flattening and gather both agree
        idx_c, _ = idx_c.sort(dim=1)
        idx_q, _ = idx_q.sort(dim=1)

        # Get coordinate dimensions
        coord_dims = x_grid.shape[1]

        # Gather context and query coordinates/values
        _rep_last = lambda idx, d: einops.repeat(idx, "b n -> b n d", d=d)

        xc = torch.gather(x, 1, _rep_last(idx_c, coord_dims))
        wc = torch.gather(w, 1, _rep_last(idx_c, 1))
        yc = torch.gather(y, 1, _rep_last(idx_c, num_channels))

        xq = torch.gather(x, 1, _rep_last(idx_q, coord_dims))
        wq = torch.gather(w, 1, _rep_last(idx_q, 1))
        yq = torch.gather(y, 1, _rep_last(idx_q, num_channels))

        # Create base masks with channel dim = 1
        mc = torch.zeros(batch_size, total_points)
        mq = torch.zeros(batch_size, total_points)
        mc.scatter_(1, idx_c, 1.0)
        mq.scatter_(1, idx_q, 1.0)

        # Reshape masks to base dimensions
        if self.mode == "spatio-temporal":
            pattern = "b (t h w) -> b 1 t h w"
            axes_lengths = dict(t=steps, h=height, w=width)
        elif self.mode == "spatial":
            pattern = "b (h w) -> b 1 h w"
            axes_lengths = dict(h=height, w=width)
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        mc = einops.rearrange(mc, pattern, **axes_lengths)
        mq = einops.rearrange(mq, pattern, **axes_lengths)

        return KolmogorovBatch(
            x=x,
            w=w,
            y=y,
            xc=xc,
            wc=wc,
            yc=yc,
            xq=xq,
            wq=wq,
            yq=yq,
            x_grid=x_grid,
            w_grid=w_grid,
            y_grid=y_grid,
            mc_grid=mc,
            mq_grid=mq,
        )
