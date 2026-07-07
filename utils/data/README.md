# Data Pipeline Guide

This document describes the data pipeline architecture used in this project: how raw data is processed, loaded, and served to models during training and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Processor → Dataset → DataLoader](#processor--dataset--dataloader)
   - [Map-Style vs Iterable Datasets](#map-style-vs-iterable-datasets)
3. [Base Classes](#base-classes)
4. [Reproducibility](#reproducibility)
   - [Map-Style Datasets](#map-style-datasets-image-kolmogorov)
   - [Iterable Datasets](#iterable-datasets-synthetic-era5)
5. [Dataset Implementations](#dataset-implementations)
   - [Image (CIFAR-10, DTD, SVHN)](#image-cifar-10-dtd-svhn)
   - [Kolmogorov Flow](#kolmogorov-flow)
   - [ERA5 Climate](#era5-climate)
   - [Synthetic](#synthetic)
   - [Predator-Prey](#predator-prey)
6. [Multiprocessing Safety](#multiprocessing-safety)
7. [Adding a New Dataset](#adding-a-new-dataset)

## Overview

The pipeline follows a three-layer pattern:

1. **Processor** — loads raw data, applies normalization, caches results
2. **Dataset** — extracts data from the processor, drops the processor reference, serves samples
3. **DataLoader** — wraps the dataset with batching, shuffling, and multi-worker loading

This separation keeps heavy I/O and preprocessing in the processor, while the dataset remains lightweight and safe for multiprocessing.

## Architecture

### Processor → Dataset → DataLoader

```
Processor (heavy I/O, normalization, caching)
    │
    ▼
Dataset (lightweight, holds only tensors/mmaps)
    │
    ▼
DataLoader (batching, shuffling, num_workers)
```

Every dataset extracts what it needs from the processor during `__init__` and then **drops the processor reference**. This ensures the dataset object is safe to pickle (required by PyTorch's `spawn`-based multiprocessing on macOS) and avoids issues with non-fork-safe file handles (HDF5, NetCDF4).

### Map-Style vs Iterable Datasets

| | Map-Style (`BaseMapDataset`) | Iterable (`BaseIterableDataset`) |
|---|---|---|
| **Base class** | `torch.utils.data.Dataset` | `torch.utils.data.IterableDataset` |
| **Data access** | `__getitem__(idx)` + `collate_fn` | `generate_batch()` |
| **Shuffling** | Handled by DataLoader's sampler | Handled internally |
| **Use when** | Data is finite and pre-extracted into tensors | Data is generated on-the-fly or too large to hold in memory |
| **Examples** | Image, Kolmogorov | Synthetic, ERA5, PredPrey |

## Base Classes

### `BaseBatch`
Abstract dataclass base for all batch types. Provides `.to(device)` to move all tensor fields.

### `Batch`
Standard batch with fields: `x`, `y`, `xc`, `yc`, `xq`, `yq`. Flat
batches add grid-form siblings alongside the per-sample slots:
- `ImageBatch` — `x_grid`, `y_grid`, `mc_grid`, `mq_grid`.
- `KolmogorovBatch` — adds vorticity channels (`w`, `wc`, `wq`,
  `w_grid`) plus `x_grid`, `y_grid`, `mc_grid`, `mq_grid`.
- `ERA5Batch` — adds a non-missing-data mask (`m_grid`) plus `x_grid`,
  `y_grid`, `mc_grid`, `mq_grid`.

### `BaseMapDataset`
For fixed/cached datasets. Subclasses implement:
- `__getitem__(idx)` — return raw data for one sample
- `collate_fn(samples)` — stack samples, sample context/query points, return a batch
- `_sample_point_counts(n_max)` — sample number of context and query points

### `BaseIterableDataset`
For dynamically generated data. Subclasses implement:
- `generate_batch()` — produce one complete batch

Supports deterministic mode (caches batches on first iteration for reproducible validation/test sets).

## Reproducibility

The two dataset families have different reproducibility models. This section documents what is guaranteed to be fixed, what is not, and what changes break reproducibility.

### At a glance

| | `ImageDataset` | `KolmogorovDataset` | `SyntheticDataset` | `ERA5Dataset` | `PredPreySimDataset` | `PredPreyRealDataset` |
|---|---|---|---|---|---|---|
| **Type** | Map-style | Map-style | Iterable | Iterable | Iterable | Iterable |
| **Reproducibility parameter** | `random_state` | `random_state` | `deterministic_seed` | `deterministic_seed` | `deterministic_seed` | `deterministic_seed` |
| **Per-sample split fixed** | ✅ index-seeded | ✅ index-seeded | ✅ cached after first iter | ✅ cached after first iter | ✅ cached after first iter | ✅ cached after first iter |
| **nc/nq fixed** | ⚠️ batch-size-dependent | ⚠️ batch-size-dependent | ✅ cached after first iter | ✅ cached after first iter | ✅ cached after first iter | ✅ cached after first iter |
| **Crop position fixed** | — | ✅ index-seeded | — | — | — | — |
| **DDP-safe** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (rank-0 download + barrier) |
| **Invariant to `num_workers`** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Breaks on `batch_size` change** | nc/nq only | nc/nq only | everything | everything | everything | everything |
| **Breaks on `samples_per_epoch` change** | nc/nq only | nc/nq only | everything | everything | everything | everything |

### Map-Style Datasets (Image, Kolmogorov)

Reproducibility is controlled by the `random_state` parameter. When `random_state=None` (default), all randomness comes from the global RNG — suitable for training.

**How it works when `random_state` is set:**

- **Per-sample context/query split** — seeded by `random_state + 2 * samples_per_epoch + idx`. Each sample index always produces the same pixel/point permutation, regardless of run, epoch, DDP rank, `num_workers`, or which other samples appear in the same batch.
- **Crop coordinates** (Kolmogorov only) — seeded by `random_state + idx` inside `torch.random.fork_rng`, so crop positions are also index-pinned and do not affect the global RNG.
- **Number of context/query points (nc/nq)** — seeded by `random_state + samples_per_epoch + samples[0]["_idx"]` (the first sample's index in the collated batch). This is consistent within a run but is **not invariant to batch size** (see caution below).

**What is fixed:**

| Property | Fixed? |
|---|---|
| Per-sample context/query pixel permutation | ✅ Yes — index-seeded |
| Crop position (Kolmogorov) | ✅ Yes — index-seeded |
| Results across DDP ranks | ✅ Yes — rank-independent |
| Results across `num_workers` settings | ✅ Yes — worker-independent |
| Results across runs (same config) | ✅ Yes — deterministic from `random_state` |

**What breaks reproducibility:**

| Change | Effect |
|---|---|
| Changing `random_state` | All splits and crops change |
| Changing `batch_size` | nc/nq values change (batch compositions shift, so the first-sample seed differs) |
| Changing `samples_per_epoch` | nc/nq seed offsets shift (aliasing between seed namespaces) |

---

### Iterable Datasets (Synthetic, ERA5)

Reproducibility is controlled by `deterministic=True` and `deterministic_seed`. When `deterministic=False` (default), batches are generated fresh each epoch using the live global RNG — suitable for training.

**How it works when `deterministic=True`:**

- On the **first** `__iter__` call, the current global RNG state is saved, everything is seeded with `deterministic_seed` via `pl.seed_everything`, all `_global_num_batches` batches are generated sequentially, then the original RNG state is restored. The batches are cached.
- On **subsequent epochs**, the cache is replayed — no new generation occurs.
- For DDP, all batches are generated globally first and then sliced by rank/worker, ensuring each rank sees the same subset across runs.

Crucially, batches are generated by **sequential RNG consumption**: batch 0 consumes some random numbers, then batch 1, and so on. This means the content of every batch depends on the full generation sequence.

**What is fixed:**

| Property | Fixed? |
|---|---|
| Batch content across epochs (same run) | ✅ Yes — cache is replayed |
| Batch content across runs (same config) | ✅ Yes — deterministic from `deterministic_seed` |
| Results across DDP ranks | ✅ Yes — generated globally then sliced |
| Global training RNG unaffected | ✅ Yes — RNG state saved and restored around generation |

**What breaks reproducibility:**

| Change | Effect |
|---|---|
| Changing `deterministic_seed` | All batches change |
| Changing `batch_size` | RNG consumption per batch changes → all batches from batch 0 onward change |
| Changing `samples_per_epoch` | Total number of batches changes → RNG consumption shifts → all batches change |
| Changing `drop_last` | Alters the final batch size, shifting RNG consumption |

> **Key difference from map-style datasets:** in iterable datasets, all batch content (not just nc/nq) is affected by changes to `batch_size` or `samples_per_epoch`, because generation is sequential rather than index-seeded. Any config change invalidates the entire cached sequence.

## Dataset Implementations

### Image (CIFAR-10, DTD, SVHN)

| Layer | Class |
|---|---|
| Processor | `CIFARDataProcessor`, `DTDDataProcessor`, `SVHNDataProcessor` |
| Dataset | `ImageDataset` (shared) |
| Batch type | `ImageBatch` |
| Base class | `BaseMapDataset` |

**Data flow:** Processor downloads/loads images → Dataset stores image tensors in `__init__`, drops processor → `__getitem__` returns a single image tensor → `collate_fn` stacks images, samples random context/query masks.

### Kolmogorov Flow

| Layer | Class |
|---|---|
| Processor | `KolmogorovDataProcessor` |
| Dataset | `KolmogorovDataset` |
| Batch type | `KolmogorovBatch` |
| Base class | `BaseMapDataset` |

**Data flow:** Processor generates/loads fluid trajectories via `KolmogorovFlow` simulation, computes statistics, normalizes → Dataset extracts trajectory, vorticity, and grid tensors, drops processor → `__getitem__` applies the crop strategy and returns a dict of tensors → `collate_fn` stacks and samples context/query masks.

**Crop strategies** (`KolmogorovCropStrategy`): The dataset requires a `crop_strategy` argument that controls how samples are cropped each call to `__getitem__`.

| Strategy | Use | Behaviour |
|---|---|---|
| `KolmogorovRandomCropStrategy` | Train | One random crop per sample per call. Stateless — safe for `num_workers > 0`. |
| `KolmogorovStridedCropStrategy` | Validation / Test | Enumerates all non-overlapping (or overlapping) tiles across the full sample. Deterministic — `__len__` expands by the tile count. |

`crop_size` and `strides` in `KolmogorovStridedCropStrategy` accept a scalar (broadcast to all dims) or a per-dimension tuple. Setting `strides == crop_size` gives non-overlapping tiling; `strides < crop_size` gives overlapping tiling.

**Modes:** The processor and dataset support two modes controlled by `KolmogorovDataProcessor(mode=...)`:

| Mode | `x_dim` | `grid` channels | Tiling dims |
|---|---|---|---|
| `"spatio-temporal"` (default) | 3 | `[3, T, H, W]` — (t, x, y) | (T, H, W) |
| `"spatial"` | 2 | `[2, H, W]` — (x, y) only | (H, W) per time step |

### ERA5 Climate

| Layer | Class |
|---|---|
| Processor | `ERA5DataProcessor` |
| Dataset | `ERA5Dataset` |
| Batch type | `ERA5Batch` |
| Base class | `BaseIterableDataset` |

**Data flow:** Processor loads NetCDF files, normalizes, exports coordinates and data variables as `.npy` files → Dataset loads coordinates as numpy arrays and stores paths to data variable files, drops processor → `generate_batch` accesses data variables via lazy `np.load(path, mmap_mode='r')`. Each DataLoader worker re-opens its own mmap via `__getstate__`/`__setstate__`.

### Synthetic

| Layer | Class |
|---|---|
| Generators | `SyntheticInputGenerator`, `BaseSyntheticOutputGenerator` subclasses |
| Dataset | `SyntheticDataset` |
| Batch type | `SyntheticBatch` |
| Base class | `BaseIterableDataset` |

**Data flow:** Input and output generators are composed in the dataset → `generate_batch` calls `input_generator.sample()` then `output_generator.sample()`, splits into context/query.

### Predator-Prey

| Layer | Class |
|---|---|
| Processor | — (no processor; data is generated or downloaded directly) |
| Dataset (sim) | `PredPreySimDataset` |
| Dataset (real) | `PredPreyRealDataset` |
| Batch type | `PredPreyBatch` |
| Base class | `BaseIterableDataset` |

**`PredPreySimDataset`** generates synthetic Lotka-Volterra trajectories on the fly from a pre-sampled pool. Each `generate_batch` call draws random initial conditions, integrates the ODE, then samples random context/query splits.

**`PredPreyRealDataset`** loads the Hudson Bay Company hare-lynx census dataset (1845–1935, 91 annual observations). The data file (`LynxHare.txt`) is downloaded automatically on first use and cached in `data_path` (typically `dataset-files/predprey/`). In DDP training, rank 0 performs the download behind a `torch.distributed.barrier()`. Each `generate_batch` call draws random context indices from the fixed 91-point time series; the remaining points become the held-out query set.

Both datasets use the same `PredPreyBatch` type (`xc`, `yc`, `xq`, `yq`, `x_dense`, `y_dense`), enabling zero-shot sim-to-real transfer: a model trained on `PredPreySimDataset` can be evaluated directly on `PredPreyRealDataset` with no changes to the forward pass, metrics, or plotter.

**Sim-to-real evaluation** swaps the `benchmark/predprey` group member on the CLI:
```bash
python eval.py +experiment=predprey/default model/predprey=<model> benchmark/predprey=test_real
```
`conf/benchmark/predprey/test_real.yaml` chains through `base` via its own
`defaults: [base, _self_]` list, so it overlays `data.datasets.test` (plus
`plot_fn.xc_range_eval` / `xq_range_eval` to match the real data range
[0, 9.0] and `misc.metrics_dirpath` → `metrics/real/`) while keeping
the train/validation datasets from `base.yaml`. Use `benchmark/predprey=test_sim`
for the simulated test split.

## Multiprocessing Safety

All datasets are designed to work with `num_workers > 0`:

- **Map-style datasets** store only plain tensors — inherently picklable and fork-safe.
- **ERA5** exports data to `.npy` files and uses memory-mapped reads. The mmap handle is cleared on pickle (`__getstate__`) and lazily re-opened per worker (`__setstate__`), making it safe with both `spawn` and `fork` start methods.
- **Synthetic** generators are stateless and picklable.

The key rule: **never store unpicklable objects** (xarray Datasets, open file handles, HDF5 references) on the dataset after `__init__` completes.

## Adding a New Dataset

1. **Create a processor** in `utils/data/<name>/processor.py` that handles raw data loading, normalization, and caching.

2. **Create a dataset** in `utils/data/<name>/dataset.py`:
   - Choose `BaseMapDataset` if your data fits in memory as tensors.
   - Choose `BaseIterableDataset` if data must be generated or streamed.
   - Extract everything you need from the processor in `__init__`, then drop the processor reference.

3. **Create a batch dataclass** extending `BaseBatch` with your dataset's fields.

4. **Register an ``as_batch`` handler** for your batch type in the same file (singledispatch) so ``eval_on="context"`` swaps its query slot from its context slot. **Register a forward wrapper** in `utils/experiment/forward_wrappers.py` only if your new batch's attribute shape doesn't match the existing ``cnp_forward_wrapper`` registration (which reads ``batch.xc`` / ``batch.yc`` / ``batch.xq`` uniformly). The metric path needs no per-batch wrapper — ``metric_fn(likelihood, batch)`` reads ``batch.yq`` on every batch type.

5. **Add a plotter** in `utils/plot_fn/<name>.py` if visualization is needed.

6. **Export** your classes from `utils/data/<name>/__init__.py` and `utils/data/__init__.py`.
