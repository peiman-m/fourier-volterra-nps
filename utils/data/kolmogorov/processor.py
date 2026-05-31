import hashlib
import pickle
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import einops
import h5py
import numpy as np
import torch
from tqdm import tqdm

from .kolmogorov import KolmogorovFlow


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class KolmogorovDataProcessor:
    """
    Kolmogorov Flow data processor and trajectory generator.

    This class handles trajectory generation, caching, normalization, and data
    management for Kolmogorov flow simulations. It can be used as a standalone
    PyTorch Dataset for exploration, but for training, KolmogorovDataset should
    be used instead for efficient batching and memory management.

    Responsibilities:
    - Lazy trajectory generation using KolmogorovFlow simulation
    - Intelligent caching based on simulation parameters
    - Computing statistics and normalizing data
    - Creating train/validation/test splits
    - Providing access to spatial or spatio-temporal data

    Usage:
        # For exploration and debugging:
        processor = KolmogorovDataProcessor(
            dirpath='./data', subset='train', mode='spatio-temporal'
        )
        sample = processor[0]  # Get a single trajectory
        print(f"Dataset has {len(processor)} samples")

        # For training (recommended):
        processor = KolmogorovDataProcessor(dirpath='./data', subset='train')
        dataset = KolmogorovDataset(dataset=processor, min_pc=0.1, max_pc=0.5, ...)
        loader = DataLoader(dataset, batch_size=16, num_workers=4)

    Note:
        KolmogorovDataset extracts only subset data from the processor during
        initialization and drops the processor reference to avoid memory
        duplication in DataLoader workers.
    """

    # Default split ratios
    DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, validation, test

    def __init__(
        self,
        dirpath: str | Path,
        *,
        # Simulation parameters
        size: int = 256,
        dt: float = 0.01,
        reynolds: int = 1000,
        trajectory_length: int = 64,
        burn_in_length: int = 64,
        num_trajectories: int = 1024,
        # Dataset parameters
        mode: Literal["spatial", "spatio-temporal"] = "spatio-temporal",
        subset: Literal["train", "validation", "test"] = "train",
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        # Processing options
        spatial_coarsen_factor: int = 4,  # Spatial downsampling
        temporal_coarsen_factor: int = 1,  # Temporal downsampling
        normalize_trajectory: bool = True,  # Normalize velocity field trajectories
        normalize_vorticity: bool = True,  # Normalize vorticity fields
        normalize_grid: bool = True,  # Normalize spatial-temporal grid
        # Tranformations and augmentations
        transform: Callable | None = None,
        # Caching and generation
        use_cached: bool = True,
        force_regenerate: bool = False,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Initialize KolmogorovDataProcessor.

        Args:
            dirpath: Directory to store/load trajectory data
            size: Grid size for simulation (before coarsening)
            dt: Time step for simulation
            reynolds: Reynolds number for fluid dynamics
            trajectory_length: Number of time steps per trajectory
            burn_in_length: Number of initial time steps to warm up the simulation.
                These snapshots will be discarded.
            num_trajectories: Total number of trajectories to generate
            mode: Whether to use "spatial" or "spatio-temporal" mode.
            subset: Which subset to use ("train", "validation", "test")
            split_ratios: Ratios for train/validation/test splits
            spatial_coarsen_factor: Factor to downsample spatial resolution
            temporal_coarsen_factor: Factor to downsample temporal resolution
            normalize_trajectory: Whether to normalize trajectory data
            normalize_vorticity: Whether to normalize vorticity data
            normalize_grid: Whether to normalize grid coordinates
            transform: Optional transformation to apply to trajectories and/or grid
            use_cached: Whether to use cached data if available
            force_regenerate: Force regeneration even if cache exists
            seed: Random seed for reproducibility
            verbose: Whether to print progress messages
        """
        # Convert OmegaConf ListConfig to tuple if needed
        if hasattr(split_ratios, "_content"):  # Check if it's a ListConfig
            split_ratios = cast(tuple[float, float, float], tuple(split_ratios))

        # Validate inputs
        if mode not in ["spatial", "spatio-temporal"]:
            raise ValueError(
                "mode must be one of ['spatial', 'spatio-temporal'], "
                f"got {mode}"
            )

        if subset not in ["train", "validation", "test"]:
            raise ValueError(
                "subset must be one of ['train', 'validation', 'test'], "
                f"got {subset}"
            )

        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")

        if spatial_coarsen_factor < 1 or temporal_coarsen_factor < 1:
            raise ValueError("coarsen_factor must be >= 1")

        # Store parameters
        self.dirpath = Path(dirpath)
        self.size = size
        self.dt = dt
        self.reynolds = reynolds
        self.trajectory_length = trajectory_length
        self.burn_in_length = burn_in_length
        self.num_trajectories = num_trajectories
        self.mode = mode
        self.subset = subset
        self.split_ratios = split_ratios
        self.spatial_coarsen_factor = spatial_coarsen_factor
        self.temporal_coarsen_factor = temporal_coarsen_factor
        self.normalize_trajectory = normalize_trajectory
        self.normalize_vorticity = normalize_vorticity
        self.normalize_grid = normalize_grid
        self.transform = transform
        self.seed = seed
        self.verbose = verbose

        # Create cache strategy
        self._create_cache_strategy()

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)  # Also seed Python's random module

        # Try loading from cache first
        if use_cached and not force_regenerate and self._load_cache():
            if verbose:
                print(
                    f"[INFO] [{self.subset}] Loaded Kolmogorov dataset from cache: {self.cache_trajectories}"
                )
            self._select_subset()
            return

        if verbose:
            print(f"[INFO] [{self.subset}] Generating new Kolmogorov flow trajectories")

        # Generate new trajectories if needed
        self._generate_trajectories()
        self._process_and_split()
        self._select_subset()

        # Save to cache
        if use_cached:
            self._save_cache()

        if verbose:
            print(
                f"[INFO] Kolmogorov dataset ready: {len(self)} trajectories in '{subset}' subset"
            )

    def _create_cache_strategy(self) -> None:
        """Create cache paths based on simulation parameters."""
        # Hash simulation parameters for cache key
        params = {
            "size": self.size,
            "dt": self.dt,
            "reynolds": self.reynolds,
            "trajectory_length": self.trajectory_length,
            "burn_in_length": self.burn_in_length,
            "num_trajectories": self.num_trajectories,
            "spatial_coarsen_factor": self.spatial_coarsen_factor,
            "temporal_coarsen_factor": self.temporal_coarsen_factor,
            "normalize_trajectory": self.normalize_trajectory,
            "normalize_vorticity": self.normalize_vorticity,
            "normalize_grid": self.normalize_grid,
            "seed": self.seed,
            "split_ratios": self.split_ratios,
        }

        params_str = str(sorted(params.items()))
        cache_key = hashlib.md5(params_str.encode()).hexdigest()

        self.cache_dir = self.dirpath / "cache"
        self.cache_trajectories = self.cache_dir / f"{cache_key}_trajectories.h5"
        self.cache_metadata = self.cache_dir / f"{cache_key}_metadata.pkl"

        ensure_dir(self.cache_dir)

    def _generate_trajectories(self) -> None:
        """Generate Kolmogorov flow trajectories using existing simulation."""
        if self.verbose:
            print(
                f"[INFO] Generating {self.num_trajectories} Kolmogorov flow trajectories\n"
                f"       Grid size: {self.size}x{self.size}, dt: {self.dt}, Reynolds: {self.reynolds}\n"
                f"       Trajectory length: {self.trajectory_length}, extra burn-in: {self.burn_in_length} "
                f"(total steps: {self.trajectory_length + self.burn_in_length})"
            )

        # Use the existing KolmogorovFlow class
        kolmogorov = KolmogorovFlow(size=self.size, dt=self.dt, reynolds=self.reynolds)

        # Determine batch size for generation
        batch_size = min(32, self.num_trajectories)
        num_batches = (self.num_trajectories + batch_size - 1) // batch_size

        all_trajectories = []
        all_vorticities = []

        for batch_idx in tqdm(
            range(num_batches),
            desc="Generating trajectory batches",
            disable=not self.verbose,
        ):
            # Calculate actual batch size for this iteration
            current_batch_size = min(
                batch_size, self.num_trajectories - batch_idx * batch_size
            )

            # Generate initial conditions with deterministic seed
            batch_seed = self.seed + batch_idx
            u0 = kolmogorov.prior(
                (current_batch_size,), seed=batch_seed
            )  # [B, C, H, W]

            # Generate trajectory
            trajectory = kolmogorov.trajectory(
                u0,
                self.trajectory_length + self.burn_in_length,
            )  # [T, B, C, H, W]

            # Remove burn-in steps
            trajectory = trajectory[self.burn_in_length :]  # [T, B, C, H, W]

            # Compute vorticity
            vorticity = kolmogorov.vorticity(trajectory)  # [T, B, H, W]

            # Apply spatial coarsening if requested
            if self.spatial_coarsen_factor > 1 or self.temporal_coarsen_factor > 1:
                trajectory = KolmogorovFlow.coarsen(
                    trajectory,
                    self.spatial_coarsen_factor,
                    self.temporal_coarsen_factor,
                )

                vorticity = KolmogorovFlow.coarsen(
                    vorticity,
                    self.spatial_coarsen_factor,
                    self.temporal_coarsen_factor,
                )

            all_trajectories.append(trajectory)
            all_vorticities.append(vorticity)

        # Concatenate all batches:
        all_trajectories = torch.cat(all_trajectories, dim=1)  # [T, N_total, C, H, W]
        all_vorticities = torch.cat(all_vorticities, dim=1)  # [T, N_total, H, W]

        # Create time array and spatial coordinates
        grid = kolmogorov.get_grid(
            num_time_steps=self.trajectory_length + self.burn_in_length,
            start_time=0.0,
        )  # [T, 3, H, W]

        # Discard the burn-in time steps
        grid = grid[self.burn_in_length :]  # [T, 3, H, W]

        # Apply spatial coarsening if requested
        if self.spatial_coarsen_factor > 1 or self.temporal_coarsen_factor > 1:
            grid = KolmogorovFlow.coarsen(
                grid,
                self.spatial_coarsen_factor,
                self.temporal_coarsen_factor,
            )

        # Rearrange to [N_total, C, T, H, W]
        self.trajectories = einops.rearrange(all_trajectories, "t b c h w -> b c t h w")
        self.vorticities = einops.rearrange(all_vorticities, "t b h w -> b 1 t h w")
        self.grid = einops.rearrange(grid, "t c h w -> c t h w")

        if self.verbose:
            print(
                f"[INFO] Generated trajectory tensor shape: {tuple(self.trajectories.shape)}"
            )
            print(
                f"[INFO] Generated vorticity tensor shape: {tuple(self.vorticities.shape)}"
            )
            print(f"[INFO] Generated spatial grid shape: {tuple(self.grid.shape)}")

    def _process_and_split(self) -> None:
        """Process trajectories and create train/validation/test splits."""
        # Create train/validation/test splits first
        N = self.trajectories.shape[0]  # Number of trajectories
        indices = torch.randperm(N, generator=torch.Generator().manual_seed(self.seed))

        train_end = int(self.split_ratios[0] * N)
        val_end = train_end + int(self.split_ratios[1] * N)

        self.split_indices = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        if self.verbose:
            for split, idx in self.split_indices.items():
                print(f"[INFO] {split}: {len(idx)} trajectories")

        # Compute statistics for normalization using only training data
        if any(
            [self.normalize_trajectory, self.normalize_vorticity, self.normalize_grid]
        ):
            self._compute_statistics()
            self._normalize_data()

    def _compute_statistics(self) -> None:
        """Compute statistics for normalization using only training data."""
        # Get training indices
        train_indices = self.split_indices["train"]
        self.stats = {}

        # Compute trajectory statistics
        if self.normalize_trajectory:
            train_trajectories = self.trajectories[train_indices]
            flattened_trajectories = einops.rearrange(
                train_trajectories, "n c t h w -> (n t h w) c"
            )

            self.stats["trajectories"] = {
                "mean": flattened_trajectories.mean(dim=0),  # [C]
                "std": flattened_trajectories.std(dim=0),  # [C]
                "min": flattened_trajectories.min(dim=0)[0],  # [C]
                "max": flattened_trajectories.max(dim=0)[0],  # [C]
            }

            # Ensure std is not zero
            self.stats["trajectories"]["std"] = torch.clamp(
                self.stats["trajectories"]["std"], min=1e-8
            )

            if self.verbose:
                print(
                    f"[INFO] Trajectory statistics computed from training subset "
                    f"({train_trajectories.shape[0]} trajectories): "
                    f"Mean={self.stats['trajectories']['mean'].detach().cpu().numpy()}, "
                    f"Std={self.stats['trajectories']['std'].detach().cpu().numpy()}"
                )

        # Compute vorticity statistics
        if self.normalize_vorticity:
            train_vorticities = self.vorticities[train_indices]
            flattened_vorticities = train_vorticities.view(-1)

            self.stats["vorticities"] = {
                "mean": flattened_vorticities.mean(),  # scalar
                "std": flattened_vorticities.std(),  # scalar
                "min": flattened_vorticities.min(),  # scalar
                "max": flattened_vorticities.max(),  # scalar
            }

            # Ensure std is not zero
            self.stats["vorticities"]["std"] = torch.clamp(
                self.stats["vorticities"]["std"], min=1e-8
            )

            if self.verbose:
                print(
                    f"[INFO] Vorticity statistics computed from training subset: "
                    f"Mean={self.stats['vorticities']['mean']:.6f}, "
                    f"Std={self.stats['vorticities']['std']:.6f}"
                )

        # Compute grid statistics
        if self.normalize_grid:
            # Grid shape: [3, T, H, W] - time, x, y coordinates
            flattened_grid = einops.rearrange(self.grid, "c t h w -> (t h w) c")

            self.stats["grid"] = {
                "mean": flattened_grid.mean(dim=0),  # [3] for t, x, y
                "std": flattened_grid.std(dim=0),  # [3] for t, x, y
                "min": flattened_grid.min(dim=0)[0],  # [3] for t, x, y
                "max": flattened_grid.max(dim=0)[0],  # [3] for t, x, y
            }

            # Ensure std is not zero
            self.stats["grid"]["std"] = torch.clamp(self.stats["grid"]["std"], min=1e-8)

            if self.verbose:
                print(
                    f"[INFO] Grid statistics computed: "
                    f"Mean={self.stats['grid']['mean'].detach().cpu().numpy()}, "
                    f"Std={self.stats['grid']['std'].detach().cpu().numpy()}"
                )

    def _normalize_data(self) -> None:
        """Apply normalization to trajectories, vorticities, and grid as specified."""

        # Normalize trajectories
        if self.normalize_trajectory and "trajectories" in self.stats:
            # Broadcasting: [N, C, T, H, W] with [C]
            mean = self.stats["trajectories"]["mean"].view(1, -1, 1, 1, 1)
            std = self.stats["trajectories"]["std"].view(1, -1, 1, 1, 1)
            self.trajectories = (self.trajectories - mean) / std

            if self.verbose:
                print("[INFO] Applied z-score normalization to trajectories")

        # Normalize vorticities
        if self.normalize_vorticity and "vorticities" in self.stats:
            # Broadcasting: [N, T, H, W] with scalar
            mean = self.stats["vorticities"]["mean"]
            std = self.stats["vorticities"]["std"]
            self.vorticities = (self.vorticities - mean) / std

            if self.verbose:
                print("[INFO] Applied z-score normalization to vorticity fields")

        # Normalize grid
        if self.normalize_grid and "grid" in self.stats:
            # Broadcasting: [3, T, H, W] with [3]
            mean = self.stats["grid"]["mean"].view(-1, 1, 1, 1)
            std = self.stats["grid"]["std"].view(-1, 1, 1, 1)
            self.grid = (self.grid - mean) / std

            if self.verbose:
                print("[INFO] Applied z-score normalization to spatial grid")

    def _select_subset(self) -> None:
        """Select the appropriate subset of trajectories."""
        subset_indices = self.split_indices[self.subset]

        # Select trajectories for this subset
        self.subset_trajectories = self.trajectories[
            subset_indices
        ]  # [N_subset, C, T, H, W]
        self.subset_vorticities = self.vorticities[
            subset_indices
        ]  # [N_subset, 1, T, H, W]

    def _save_cache(self) -> bool:
        """Save trajectories to HDF5 and metadata to pickle."""
        try:
            if self.verbose:
                print(f"[INFO] Caching trajectories to: {self.cache_trajectories}")

            # Save trajectories and grid efficiently with HDF5
            with h5py.File(self.cache_trajectories, "w") as f:
                f.create_dataset(
                    "trajectories", data=self.trajectories.numpy(), compression="gzip"
                )
                f.create_dataset(
                    "vorticities", data=self.vorticities.numpy(), compression="gzip"
                )
                f.create_dataset("grid", data=self.grid.numpy(), compression="gzip")

            # Save metadata
            metadata = {
                "split_indices": self.split_indices,
                "stats": self.stats if hasattr(self, "stats") else None,
                "parameters": self._get_param_dict(),
            }

            with open(self.cache_metadata, "wb") as f:
                pickle.dump(metadata, f)

            if self.verbose:
                print("[INFO] Successfully saved trajectory cache")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to save trajectory cache: {e}")
            return False

    def _load_cache(self) -> bool:
        """Load trajectories from cache if available and valid."""
        if not (self.cache_trajectories.exists() and self.cache_metadata.exists()):
            return False
        try:
            # Load metadata first to validate parameters
            with open(self.cache_metadata, "rb") as f:
                metadata = pickle.load(f)

            # Check if parameters match
            cached_params = metadata["parameters"]
            current_params = self._get_param_dict()

            if cached_params != current_params:
                if self.verbose:
                    print(
                        "[INFO] Cache parameters mismatch - regenerating trajectories"
                    )
                return False

            # Load trajectories and grid
            with h5py.File(self.cache_trajectories, "r") as f:
                self.trajectories = torch.from_numpy(
                    cast(h5py.Dataset, f["trajectories"])[:]
                )
                self.vorticities = torch.from_numpy(
                    cast(h5py.Dataset, f["vorticities"])[:]
                )
                self.grid = torch.from_numpy(cast(h5py.Dataset, f["grid"])[:])

            # Load other cached data
            self.split_indices = metadata["split_indices"]
            self.stats = metadata["stats"]

            return True

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to load trajectory cache: {e}")
            return False

    def _get_param_dict(self) -> dict[str, Any]:
        """Get dictionary of parameters for cache validation."""
        return {
            "size": self.size,
            "dt": self.dt,
            "reynolds": self.reynolds,
            "trajectory_length": self.trajectory_length,
            "burn_in_length": self.burn_in_length,
            "num_trajectories": self.num_trajectories,
            "spatial_coarsen_factor": self.spatial_coarsen_factor,
            "temporal_coarsen_factor": self.temporal_coarsen_factor,
            "normalize_trajectory": self.normalize_trajectory,
            "normalize_vorticity": self.normalize_vorticity,
            "normalize_grid": self.normalize_grid,
            "seed": self.seed,
            "split_ratios": self.split_ratios,
        }

    def __len__(self) -> int:
        """Return dataset length based on mode."""
        if self.mode == "spatial":
            # In spatial mode, length is sum of all trajectory lengths
            return (
                self.subset_trajectories.shape[0] * self.subset_trajectories.shape[2]
            )  # N * T
        else:
            # In spatio-temporal mode, length is number of trajectories
            return self.subset_trajectories.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:
        """
        Get data by index based on mode.

        Args:
            idx: Index into the dataset

        Returns:
            Dictionary containing:
            - In spatio-temporal mode:
                - 'trajectory': Velocity field trajectory [C, T, H, W]
                - 'vorticity': Vorticity trajectory [1, T, H, W]
                - 'grid': Simulation grid [3, T, H, W] (time, x, y coordinates)
            - In spatial mode:
                - 'trajectory': Single spatial snapshot [C, H, W]
                - 'vorticity': Single vorticity snapshot [1, H, W]
                - 'grid': Simulation grid for this time step [3, H, W]
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if self.mode == "spatial":
            # In spatial mode, index maps to specific time step in trajectory
            traj_length = self.subset_trajectories.shape[2]  # T
            traj_idx = idx // traj_length  # Which trajectory
            time_idx = idx % traj_length  # Which time step

            result = {
                "trajectory": self.subset_trajectories[
                    traj_idx, :, time_idx
                ],  # [C, H, W]
                "vorticity": self.subset_vorticities[
                    traj_idx, :, time_idx
                ],  # [1, H, W]
                "grid": self.grid[1:, time_idx],  # [2, H, W]
            }
        else:
            # In spatio-temporal mode, return full trajectory
            result = {
                "trajectory": self.subset_trajectories[idx],  # [C, T, H, W]
                "vorticity": self.subset_vorticities[idx],  # [1, T, H, W]
                "grid": self.grid,  # [3, T, H, W]
            }

        if self.transform is not None:
            return self.transform(result)

        return result

    @classmethod
    def clear_cache(
        cls,
        dirpath: str | Path,
        verbose: bool = True,
    ) -> int:
        """
        Clear all cached data in the specified directory.

        Args:
            dirpath: Path to data directory
            verbose: Whether to print progress

        Returns:
            Number of cache files removed
        """
        cache_dir = Path(dirpath) / "cache"
        if not cache_dir.exists():
            return 0

        removed = 0
        for file_path in cache_dir.iterdir():
            if file_path.is_file() and (file_path.suffix in [".h5", ".pkl"]):
                file_path.unlink()
                removed += 1
                if verbose:
                    print(f"[INFO] Cleaned up cache file: {file_path}")

        return removed
