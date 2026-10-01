from abc import ABC, abstractmethod

import numpy as np
import torch


class KolmogorovCropStrategy(ABC):
    @abstractmethod
    def crops_per_sample(self, *dims: int) -> int:
        """
        Number of crops yielded per sample.
        spatial:         dims = (H, W)
        spatio-temporal: dims = (T, H, W)
        """
        ...

    @abstractmethod
    def get_crop_coords(self, local_idx: int, *dims: int) -> tuple[int, ...]:
        """
        Return crop coordinates for the local_idx-th crop.
        spatial:         dims = (H, W)    → (h0, ch, w0, cw)
        spatio-temporal: dims = (T, H, W) → (t0, ct, h0, ch, w0, cw)
        """
        ...


class KolmogorovRandomCropStrategy(KolmogorovCropStrategy):
    """
    Random crop strategy for the train split.

    Always yields exactly one crop per sample. Crop positions are sampled
    uniformly at random — stateless across calls, safe for num_workers > 0.

    time_crop_size defaults to 1 and is optional. It is only used in
    spatio-temporal mode (dims = (T, H, W)); in spatial mode (dims = (H, W))
    it is silently ignored. height_crop_size and width_crop_size are required.

    Also implements __call__(sample) so it can be used as a drop-in transform
    on KolmogorovDataProcessor for exploration/debugging.
    """

    def __init__(
        self,
        height_crop_size: int,
        width_crop_size: int,
        time_crop_size: int = 1,
    ) -> None:
        for name, val in [
            ("time_crop_size", time_crop_size),
            ("height_crop_size", height_crop_size),
            ("width_crop_size", width_crop_size),
        ]:
            if val is None:
                raise ValueError(f"{name} must not be None")
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be an int >= 1, got {val!r}")

        self.time_crop_size = time_crop_size
        self.height_crop_size = height_crop_size
        self.width_crop_size = width_crop_size

    # ------------------------------------------------------------------
    # CropStrategy interface
    # ------------------------------------------------------------------

    def crops_per_sample(self, *dims: int) -> int:
        return 1

    def get_crop_coords(self, local_idx: int, *dims: int) -> tuple[int, ...]:
        if len(dims) == 2:
            H, W = dims
            h0 = self._random_start(H, self.height_crop_size, "height_crop_size")
            w0 = self._random_start(W, self.width_crop_size, "width_crop_size")
            return (h0, self.height_crop_size, w0, self.width_crop_size)
        elif len(dims) == 3:
            T, H, W = dims
            t0 = self._random_start(T, self.time_crop_size, "time_crop_size")
            h0 = self._random_start(H, self.height_crop_size, "height_crop_size")
            w0 = self._random_start(W, self.width_crop_size, "width_crop_size")
            return (t0, self.time_crop_size, h0, self.height_crop_size, w0, self.width_crop_size)
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {len(dims)}: {dims}")

    # ------------------------------------------------------------------
    # Callable interface — for use as processor.transform
    # ------------------------------------------------------------------

    def __call__(self, sample: dict) -> dict:
        """
        Apply a random crop to a sample dict returned by KolmogorovDataProcessor.

        Dispatches on trajectory.ndim:
          4D [C, T, H, W] — spatio-temporal: crops all three dims; grid is [3, T, H, W]
          3D [C, H, W]    — spatial: crops H/W only; grid is already [2, H, W] (processor
                            already stripped the time channel via grid[1:, time_idx])
        """
        trajectory = sample["trajectory"]
        if trajectory.ndim == 4:
            _, T, H, W = trajectory.shape
            t0, ct, h0, ch, w0, cw = self.get_crop_coords(0, T, H, W)
            return {
                "trajectory": trajectory[:, t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
                "vorticity":  sample["vorticity"][:, t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
                "grid":       sample["grid"][:, t0:t0 + ct, h0:h0 + ch, w0:w0 + cw],
            }
        elif trajectory.ndim == 3:
            _, H, W = trajectory.shape
            h0, ch, w0, cw = self.get_crop_coords(0, H, W)
            return {
                "trajectory": trajectory[:, h0:h0 + ch, w0:w0 + cw],
                "vorticity":  sample["vorticity"][:, h0:h0 + ch, w0:w0 + cw],
                "grid":       sample["grid"][:, h0:h0 + ch, w0:w0 + cw],
            }
        else:
            raise ValueError(
                f"Expected trajectory.ndim in {{3, 4}}, got {trajectory.ndim}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_start(self, dim_size: int, crop_size: int, name: str) -> int:
        if crop_size > dim_size:
            raise ValueError(
                f"{name} ({crop_size}) exceeds dimension size ({dim_size})"
            )
        if crop_size == dim_size:
            return 0
        return int(torch.randint(0, dim_size - crop_size + 1, (1,)).item())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"time_crop_size={self.time_crop_size}, "
            f"height_crop_size={self.height_crop_size}, "
            f"width_crop_size={self.width_crop_size})"
        )


class KolmogorovStridedCropStrategy(KolmogorovCropStrategy):
    """
    Strided tiling strategy for validation and test splits.

    Enumerates all (possibly overlapping) tiles of a fixed crop size across
    the full sample in a deterministic, reproducible order.

    crop_size / strides accept:
      scalar int  — broadcast to all dims for the active mode
      tuple       — must match the number of dims (2 for spatial, 3 for spatio-temporal);
                    length is validated at call time since mode is unknown at construction

    Setting strides == crop_size per dimension → non-overlapping tiling.
    Setting strides <  crop_size per dimension → overlapping tiling.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, ...],
        strides: int | tuple[int, ...],
    ) -> None:
        crop_size = self._list_to_tuple(crop_size)
        strides   = self._list_to_tuple(strides)

        if crop_size is None:
            raise ValueError("crop_size must not be None")
        if strides is None:
            raise ValueError("strides must not be None")

        self._validate_positive(crop_size, "crop_size")
        self._validate_positive(strides, "strides")

        self.crop_size = crop_size
        self.strides   = strides

    # ------------------------------------------------------------------
    # CropStrategy interface
    # ------------------------------------------------------------------

    def crops_per_sample(self, *dims: int) -> int:
        n = len(dims)
        crop_sizes = self._resolve(self.crop_size, n, "crop_size")
        strides    = self._resolve(self.strides,    n, "strides")

        total = 1
        for d, c, s in zip(dims, crop_sizes, strides):
            if c > d:
                raise ValueError(
                    f"crop_size {c} exceeds dimension size {d} "
                    f"(dims={dims}, crop_size={self.crop_size})"
                )
            total *= (d - c) // s + 1
        return total

    def get_crop_coords(self, local_idx: int, *dims: int) -> tuple[int, ...]:
        n = len(dims)
        crop_sizes = self._resolve(self.crop_size, n, "crop_size")
        strides    = self._resolve(self.strides,    n, "strides")

        tile_counts  = tuple((d - c) // s + 1 for d, c, s in zip(dims, crop_sizes, strides))
        tile_indices = np.unravel_index(local_idx, tile_counts)

        coords: list[int] = []
        for ti, c, s in zip(tile_indices, crop_sizes, strides):
            coords.extend([int(ti) * s, c])
        return tuple(coords)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_to_tuple(val):
        # Hydra's ``_convert_="all"`` hands YAML sequences in as plain
        # ``list``; promote to ``tuple`` so downstream ``isinstance(..., tuple)``
        # checks in ``_validate_positive`` / ``_resolve`` match.
        if isinstance(val, list):
            return tuple(val)
        return val

    @staticmethod
    def _validate_positive(val, name: str) -> None:
        if isinstance(val, tuple):
            for i, v in enumerate(val):
                if v < 1:
                    raise ValueError(f"{name}[{i}] must be >= 1, got {v}")
        else:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

    @staticmethod
    def _resolve(val, n_dims: int, name: str) -> tuple[int, ...]:
        if isinstance(val, int):
            return (val,) * n_dims
        if isinstance(val, tuple):
            if len(val) != n_dims:
                raise ValueError(
                    f"{name} has length {len(val)} but active mode has {n_dims} dims"
                )
            return val
        raise ValueError(f"{name} must be int or tuple, got {type(val)}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_size={self.crop_size}, "
            f"strides={self.strides})"
        )
