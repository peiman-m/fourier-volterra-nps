from dataclasses import dataclass, replace
from typing import Any, cast

import einops
import torch

from ..base import BaseBatch, BaseMapDataset, EvalOn, as_batch


@dataclass
class ImageBatch(BaseBatch):
    x: torch.Tensor  # Flattened pixel coordinates [B, H*W, 2]
    y: torch.Tensor  # Flattened pixel values [B, H*W, C]

    xc: torch.Tensor  # Context pixel coordinates [B, Nc, 2]
    yc: torch.Tensor  # Context pixel values [B, Nc, C]

    xq: torch.Tensor  # Query pixel coordinates [B, Nq, 2]
    yq: torch.Tensor  # Query pixel values [B, Nq, C]

    x_grid: torch.Tensor  # Pixel coordinates [B, 2, H, W]
    y_grid: torch.Tensor  # Pixel values [B, C, H, W]

    # Base masks with channel dim = 1, shared for both x and y grids
    mc_grid: torch.Tensor  # [B, 1, H, W] Context mask (base)
    mq_grid: torch.Tensor  # [B, 1, H, W] Query mask (base)

    # Properties for accessing broadcasted masks
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


@as_batch.register(ImageBatch)
def _as_batch_image(batch: ImageBatch, *, eval_on: EvalOn = "query") -> ImageBatch:
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


class ImageDataset(BaseMapDataset):
    """
    DataLoader for image data with random context/query point sampling.
    We assume that all images in a dataset have similar size.
    """

    def __init__(
        self,
        *,
        processor,
        image_shape: tuple[int, int],
        normalize_coords: bool = True,
        min_nc: int | None = None,
        max_nc: int | None = None,
        min_nq: int | None = None,
        max_nq: int | None = None,
        samples_per_epoch: int | None = None,
        use_all_queries: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the image data loader.

        Args:
            processor: DataProcessor containing image data
            image_shape: (height, width) of images
            normalize_coords: Whether to normalize coordinates to [-1, 1]
            min_nc: Minimum number of context points
            max_nc: Maximum number of context points
            min_nq: Minimum number of query points
            max_nq: Maximum number of query points
            use_all_queries: If True, use all remaining pixels as the held-out set after sampling context
            samples_per_epoch: Number of samples per epoch
            random_state: If set, pins context/query splits per index for reproducible
                validation across runs and models.
        """
        self.processor = processor
        self.image_shape = image_shape
        self.normalize_coords = normalize_coords

        # Validate and set samples per epoch
        total_samples = len(processor)
        samples_per_epoch = samples_per_epoch or total_samples
        samples_per_epoch = min(max(1, samples_per_epoch), total_samples)

        subset = getattr(processor, 'subset', '?')
        print(f'[{type(self).__name__}] [{subset}] Total samples: {total_samples}')
        print(f'[{type(self).__name__}] [{subset}] Samples per epoch: {samples_per_epoch}')

        # Initialize point count bounds
        height, width = self.image_shape
        total_pixels = height * width

        _min_nc = max(1, min_nc or 1)
        _max_nc = min(max_nc or total_pixels - 1, total_pixels - 1)
        _min_nq = max(1, min_nq or 1)
        _max_nq = min(max_nq or total_pixels - 1, total_pixels - 1)

        # Validate that we can allocate minimum points
        if _min_nc + _min_nq > total_pixels:
            raise ValueError(
                f"Sum of minimum points ({_min_nc + _min_nq}) "
                f"exceeds total pixels ({total_pixels})"
            )

        # Warn if use_all_queries is True but query bounds are specified
        if use_all_queries and (min_nq is not None or max_nq is not None):
            import warnings

            warnings.warn(
                "Query point bounds (min_nq, max_nq) are specified but will be ignored "
                "because use_all_queries=True. All non-context pixels will be used as the held-out set."
            )

        # Store point count parameters
        self.min_nc = _min_nc
        self.max_nc = _max_nc
        self.min_nq = _min_nq
        self.max_nq = _max_nq
        self.use_all_queries = use_all_queries
        self._random_state = random_state

        # Initialize parent class
        super().__init__(
            samples_per_epoch=samples_per_epoch,
        )

        # Cache for coordinate grids
        self._coord_grid_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _sample_point_counts(
        self, n_max: int, generator: torch.Generator | None = None
    ) -> tuple[int, int]:
        """Sample number of context and query points.

        Args:
            n_max: Maximum total number of points.
            generator: Optional local RNG generator (keeps global state clean).

        Returns:
            Tuple of (context_points, query_points).
        """
        if self.use_all_queries:
            nc = int(torch.randint(
                low=self.min_nc,
                high=min(self.max_nc, n_max - self.min_nq) + 1,
                size=(),
                generator=generator,
            ).item())
            nq = n_max - nc
            return nc, nq

        n_min = self.min_nc + self.min_nq
        n_max_capped = min(n_max, self.max_nc + (self.max_nq or n_max))

        if n_min > n_max_capped:
            raise ValueError(
                f"Insufficient points for minimum requirements: "
                f"got {n_max_capped}, need {n_min}"
            )

        n = int(torch.randint(low=n_min, high=n_max_capped + 1, size=(), generator=generator).item())
        nc = int(torch.randint(
            low=self.min_nc,
            high=min(self.max_nc, n - self.min_nq) + 1,
            size=(),
            generator=generator,
        ).item())
        nq = n - nc
        return nc, nq

    def _get_coordinate_grid(self, height: int, width: int) -> torch.Tensor:
        """Get or create cached coordinate grid."""
        key = (height, width)
        if key not in self._coord_grid_cache:
            self._coord_grid_cache[key] = self._create_coordinate_grid(height, width)
        return self._coord_grid_cache[key]

    def _create_coordinate_grid(self, height: int, width: int) -> torch.Tensor:
        """Create a coordinate grid for the given dimensions."""
        if self.normalize_coords:
            h_coords = torch.linspace(-1, 1, height)
            w_coords = torch.linspace(-1, 1, width)
        else:
            h_coords = torch.arange(height)
            w_coords = torch.arange(width)

        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        return torch.stack([h_grid, w_grid], dim=-1)  # Shape: (H, W, 2)

    def __getitem__(self, idx: int) -> torch.Tensor | dict:
        """Return image tensor [C, H, W] for the given index."""
        img_data = self.processor[idx]
        img = img_data[0] if isinstance(img_data, tuple) else img_data
        # Ensure 3D tensor (C, H, W)
        if img.ndim == 2:  # (H, W)
            img = img[None, ...]  # Add channel dimension
        if self._random_state is not None:
            return {"image": img, "_idx": idx}
        return img

    def collate_fn(self, samples: list[torch.Tensor | dict]) -> ImageBatch:
        """Collate image samples into a batch with context/query splitting.

        Args:
            samples: List of image tensors [C, H, W], or dicts with 'image' and '_idx'
                keys when random_state is set.

        Returns:
            ImageBatch with context/query splits
        """
        if self._random_state is not None:
            # _random_state set => __getitem__ returned dicts with "image"/"_idx".
            imgs = [cast(dict, s)["image"] for s in samples]
        else:
            imgs = cast(list[torch.Tensor], samples)
        y_grid = torch.stack(imgs, dim=0)  # [B, C, H, W]

        batch_size, num_channels, height, width = y_grid.shape
        total_pixels = height * width

        # Batch-level generator: seeds nc/nq from the first sample's index.
        # Offset by samples_per_epoch so this seed never aliases per-sample
        # permutation seeds (offset 2*samples_per_epoch) or future crop seeds (offset 0).
        batch_gen: torch.Generator | None = None
        if self._random_state is not None:
            batch_gen = torch.Generator()
            batch_gen.manual_seed(int(self._random_state + self.samples_per_epoch + cast(dict, samples[0])["_idx"]))

        # Sample point counts (batch-level: same nc/nq for all samples)
        nc, nq = self._sample_point_counts(total_pixels, generator=batch_gen)

        # Create coordinate grid
        coord_grid = self._get_coordinate_grid(height, width)
        x_grid = einops.repeat(coord_grid, "h w d -> b d h w", b=batch_size)

        # Flatten for easier indexing
        y = einops.rearrange(y_grid, "b c h w -> b (h w) c")
        x = einops.rearrange(x_grid, "b d h w -> b (h w) d")

        # Per-sample permutations: each seeded from its own _idx so the context/query
        # assignment is the same regardless of batch composition or DDP rank.
        # Offset by 2*samples_per_epoch to avoid aliasing with nc/nq seeds.
        if self._random_state is not None:
            rows = []
            for s in samples:
                sg = torch.Generator()
                sg.manual_seed(self._random_state + 2 * self.samples_per_epoch + int(cast(dict, s)["_idx"]))
                rows.append(torch.rand(total_pixels, dtype=torch.float64, generator=sg))
            randperm = torch.stack(rows).argsort(dim=-1)
        else:
            randperm = torch.rand(batch_size, total_pixels, dtype=torch.float64).argsort(dim=-1)

        # Select disjoint indices …
        idx_c = randperm[:, :nc]
        idx_q = randperm[:, nc : nc + nq]

        # … then sort them so mask-based flattening and gather both agree
        idx_c, _ = idx_c.sort(dim=1)
        idx_q, _ = idx_q.sort(dim=1)

        # Gather context and query coordinates/values
        xc = torch.gather(x, 1, einops.repeat(idx_c, "b n -> b n d", d=2))
        yc = torch.gather(
            y, 1, einops.repeat(idx_c, "b n -> b n d", d=num_channels)
        )
        xq = torch.gather(x, 1, einops.repeat(idx_q, "b n -> b n d", d=2))
        yq = torch.gather(
            y, 1, einops.repeat(idx_q, "b n -> b n d", d=num_channels)
        )

        # Create base masks with channel dim = 1
        mc = torch.zeros(batch_size, total_pixels)
        mq = torch.zeros(batch_size, total_pixels)
        mc.scatter_(1, idx_c, 1.0)
        mq.scatter_(1, idx_q, 1.0)

        # Reshape masks to base dimensions [B, 1, H, W]
        mc = einops.rearrange(mc, "b (h w) -> b 1 h w", h=height, w=width)
        mq = einops.rearrange(mq, "b (h w) -> b 1 h w", h=height, w=width)

        return ImageBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xq=xq,
            yq=yq,
            x_grid=x_grid,
            y_grid=y_grid,
            mc_grid=mc,
            mq_grid=mq,
        )
