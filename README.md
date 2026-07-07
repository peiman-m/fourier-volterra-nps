# [Revisiting Neural Processes via Fourier Transform and Volterra Series](https://openreview.net/forum?id=UEeBGrOGa8)

The official code accompanying the ICML 2026 paper of the same name.

A research codebase for training and evaluating **neural process (NP)** models with advanced convolutional architectures, including Fourier spectral convolutions, Volterra nonlinear convolutions, and their hybrids. Experiments span five domains: synthetic functions, images (CIFAR-10, DTD, SVHN), climate data (ERA5), fluid dynamics (Kolmogorov flow), and predator-prey population dynamics (sim-to-real: simulated Lotka-Volterra → Hudson Bay hare-lynx data).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration System](#configuration-system)
5. [Models](#models)
6. [Datasets](#datasets)
7. [Training Framework](#training-framework)
8. [Evaluation](#evaluation)
9. [Extending the Codebase](#extending-the-codebase)

---

## Project Structure

```
fourier-volterra-nps/
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── requirements.txt            # pinned dependency versions
│
├── nps/                        # Core neural process library
│   ├── models/                 # High-level NP model classes
│   ├── core/                   # Building blocks (convolutions, encoders, decoders, etc.)
│   ├── likelihoods/            # Output likelihood functions
│   └── utils/                  # NPS-specific utilities (grids, distances, etc.)
│
├── utils/                      # Training and data utilities
│   ├── data/                   # Data pipeline (Processor → Dataset → DataLoader)
│   │   ├── base.py             # BaseBatch, BaseMapDataset, BaseIterableDataset
│   │   ├── image/              # CIFAR-10, DTD, SVHN
│   │   ├── synthetic/          # GP and synthetic function generation
│   │   ├── era5/               # ERA5 climate data (NetCDF → numpy mmap)
│   │   ├── kolmogorov/         # Kolmogorov flow PDE simulation
│   │   └── predprey/           # Predator-prey (Lotka-Volterra sim + Hudson Bay real data)
│   ├── experiment/             # PyTorch Lightning training framework
│   │   ├── lightning_wrapper.py
│   │   ├── forward_wrappers.py
│   │   ├── metrics/             # MetricSpec + accumulators + reducers + functions + losses
│   │   ├── callbacks/           # PlotterCallback, LogPerformanceCallback
│   │   └── registry/
│   └── plot_fn/                # Per-dataset visualization helpers
│
├── conf/                       # Hydra config tree (all experiment composition)
│   ├── config.yaml             # top-level composer; pins hydra.job.chdir=false
│   ├── _resolvers.py           # eval/product/range custom resolvers
│   ├── misc/base.yaml          # default misc block (bare-load fallback)
│   ├── optimizer/adamw.yaml
│   ├── benchmark/              # monolithic per-benchmark spec (data + misc + plot + params)
│   │   ├── synthetic/{base,base-translation-test}.yaml
│   │   ├── synthetic/{input_generator,output_generator}/*.yaml
│   │   ├── image/{base-cifar10,base-dtd,base-svhn}.yaml
│   │   ├── predprey/{base,test_real,test_sim}.yaml
│   │   ├── kolmogorov/base.yaml
│   │   └── era5/base.yaml
│   ├── model/                  # architecture fragments, nested by benchmark
│   │   ├── synthetic/*.yaml
│   │   ├── image/*.yaml
│   │   ├── predprey/*.yaml
│   │   ├── kolmogorov/*.yaml
│   │   └── era5/*.yaml
│   ├── metrics/                # MetricSpec defaults groups
│   │   ├── train/nll.yaml
│   │   ├── val/{standard,synthetic}.yaml
│   │   └── test/{standard,synthetic}.yaml
│   └── experiment/             # thin composers, one per runnable experiment
│       ├── synthetic/*.yaml
│       ├── image/*.yaml
│       ├── predprey/*.yaml
│       ├── kolmogorov/*.yaml
│       └── era5/*.yaml
│
├── scripts/                    # SLURM job templates + submit drivers
│   ├── synthetic/jobs/         # submit_jobs.sh, job.job, eval variants
│   ├── image/jobs/
│   ├── predprey/jobs/
│   ├── kolmogorov/jobs/
│   └── era5/3d/jobs/
│
├── tools/                      # Standalone analysis / sanity-check scripts
│   ├── count_params.py         # walk the experiment tree, instantiate + count parameters
│   ├── hydra_smoke.py          # validate Hydra scaffolding without instantiating models
│   ├── analyze_receptive_field.py  # receptive-field analysis
│   └── analyze_erf.py          # effective receptive field (ERF) analysis
│
└── dataset-files/              # Raw and cached data (not tracked by git)
    ├── era5/
    ├── image/
    └── pde/
```

For deeper documentation on specific subsystems:
- Data pipeline: [`utils/data/README.md`](utils/data/README.md)
- Training framework: [`utils/experiment/README.md`](utils/experiment/README.md)

---

## Installation

All results in the paper were produced on a Linux HPC cluster (NVIDIA A100,
CUDA 12.4) under **Python 3.11.5**. `requirements.txt` pins the exact package
versions from that environment. On an HPC system, load the matching toolchain
modules first, then create a fresh virtualenv:

```bash
module purge
module load GCCcore/13.2.0 Python/3.11.5 git/2.42.0
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

(The `module load` lines are specific to the cluster used here; adapt them to
your own site. See the `scripts/**/jobs/job.job` templates for the full module
list used at submission time.)

**CPU-only / local development.** The code was developed and debugged on macOS,
which has no NVIDIA GPU. To set up a CPU-only environment, install a CPU build of
`torch`/`torchvision`/`torchaudio` and omit the `nvidia-*-cu12` and `triton`
lines in `requirements.txt`.

The codebase does **not** require a `setup.py` install — run all scripts from the project root.

---

## Quick Start

Training is configured via [Hydra](https://hydra.cc/) structured configs. Each
benchmark ships a single parameterized composer at
`conf/experiment/<bench>/default.yaml`; the model is selected via the
`model/<bench>` group override:

```bash
python train.py +experiment=<benchmark>/default model/<benchmark>=<model> [key=value ...]
```

The composer already wires up its data, optimizer, and default generators;
add `key=value` overrides to tweak fields or swap config groups (e.g.
`benchmark/synthetic/output_generator=sawtooth`).

### Discovering available experiments

Models live under `conf/model/<bench>/<model>.yaml`; the name you pass to
`model/<bench>=` is the file stem (no `.yaml`). To list everything:

```bash
# All models, grouped by benchmark
ls conf/model/*/*.yaml

# Python: {benchmark: [model_name, ...]}
python -c "from tools.count_params import discover_experiments; \
            print(discover_experiments())"
```

`tools/count_params.py` is a worked example that walks the experiment tree and
instantiates each composer — useful as a reference for compose-from-Python
and as a quick sanity check that every composer still loads.

### Synthetic functions

```bash
# Default (gp-rbf + uniform); pick the model via the model/synthetic group
python train.py +experiment=synthetic/default model/synthetic=convcnp-unet

# GP with Matern-5/2 kernel (switch the output-generator group)
python train.py +experiment=synthetic/default model/synthetic=convcnp-unet \
    benchmark/synthetic/output_generator=gp-matern52

# Sawtooth functions
python train.py +experiment=synthetic/default model/synthetic=sf-convcnp-f4.9-e288 \
    benchmark/synthetic/output_generator=sawtooth
```

### Image datasets

```bash
# CIFAR-10 is the default; switch model via model/image and dataset via benchmark/image
python train.py +experiment=image/default model/image=convcnp-resnet
python train.py +experiment=image/default model/image=sf-convcnp-f4.8-e384 benchmark/image=base-dtd
python train.py +experiment=image/default model/image=cnp                benchmark/image=base-svhn
```

### ERA5 climate data

```bash
python train.py +experiment=era5/default model/era5=convcnp-resnet
```

### Kolmogorov flow (PDE)

```bash
python train.py +experiment=kolmogorov/default model/kolmogorov=sf-convcnp-f4.25-e176
```

### Predator-prey (sim-to-real)

```bash
# Train on simulated Lotka-Volterra trajectories (default: benchmark/predprey=base)
python train.py +experiment=predprey/default model/predprey=te-eqtnp

# Evaluate on the simulated test split
python eval.py +experiment=predprey/default model/predprey=te-eqtnp benchmark/predprey=test_sim

# Sim-to-real: evaluate on Hudson Bay hare-lynx data
# (LynxHare.txt is downloaded automatically on first use)
python eval.py +experiment=predprey/default model/predprey=te-eqtnp benchmark/predprey=test_real
```

`benchmark/predprey=test_real` and `test_sim` are overlays that chain through
`base` via their own `defaults:` list, so they replace only
`data.datasets.test` (plus plot-range + metrics-dir overrides) while
keeping the train/validation datasets from `base.yaml`.

### Sweeps

Hydra's `--multirun` (`-m`) launches one job per combination:

```bash
# 3 seeds × 2 output generators = 6 runs
python train.py --multirun +experiment=synthetic/default model/synthetic=cnp \
    misc.seed=0,1,2 \
    benchmark/synthetic/output_generator=gp-rbf,sawtooth
```

### Cluster submission (SLURM)

On a SLURM cluster, use the per-benchmark submit drivers instead of
calling `python train.py` directly:

```
scripts/<benchmark>/jobs/
├── submit_jobs.sh   # driver — edit the MODEL_TYPES / OUTPUT_GENERATORS arrays
└── job.job          # SLURM batch template — invokes train.py / eval.py
```

To add or remove models from the matrix, edit the array variables at
the top of `submit_jobs.sh` (`MODEL_TYPES`, `OUTPUT_GENERATORS`,
`INPUT_GENERATORS`, `BASES`) and run `./submit_jobs.sh`. The driver
builds an `EXPERIMENT_NAME` + `HYDRA_OVERRIDES` tuple per cell and
passes them to `job.job` via `sbatch --export=...`; `job.job` then
invokes:

```bash
python ${OPERATION}.py "+experiment=${EXPERIMENT_NAME}" ${HYDRA_OVERRIDES} "misc.seed=${seed}"
```

Cluster runs log to the real W&B project resolved from each benchmark's
`misc.project` (e.g. `FVNP-${output_generator}-x=1d-y=1d` for synthetic).
To suppress W&B for a one-off run, prepend `WANDB_MODE=offline` to the
`python` invocation in `job.job` or export it in your local shell.

---

## Configuration System

Configs use [Hydra](https://hydra.cc/) with a defaults-list composer. Each
benchmark ships a single parameterized experiment composer at
`conf/experiment/<bench>/default.yaml` that pulls in the right
`benchmark/<bench>`, `model/<bench>`, `optimizer`, and (for synthetic)
generator fragments via its `defaults:` list. The model is selected at
the CLI via `model/<bench>=<name>`, so any entry under
`conf/model/<bench>/` is runnable without adding a per-model composer
file. The composed config is currently untyped at the root (plain
`DictConfig`); structured-schema validation is a possible future
addition.

### CLI override grammar

| Purpose | Syntax | Example |
|---|---|---|
| Pick an experiment composer | `+experiment=<bench>/default` | `+experiment=image/default` |
| Pick a model within the composer | `model/<bench>=<name>` | `model/image=convcnp-resnet` |
| Override an existing key | `key.path=value` | `misc.seed=42` |
| Add a new key | `+key.path=value` | `+misc.extra_tag=debug` |
| Delete a key | `~key.path` | `~misc.plot_fn` |
| Switch a config group | `group=member` | `benchmark/synthetic/output_generator=sawtooth` |
| Multirun sweep | `--multirun` + comma lists | `misc.seed=0,1,2` |

Nested keys use dots (`misc.checkpointing.local=false`), not slashes.

### Config groups and valid members

Group overrides select one of several sibling files in a `conf/<group>/` directory. The useful ones per benchmark:

| Group | Members (from `conf/<group>/`) | Example override |
|---|---|---|
| `benchmark/synthetic` | `base`, `base-translation-test` | `benchmark/synthetic=base-translation-test` |
| `benchmark/synthetic/output_generator` | `gp-rbf`, `gp-matern12`, `gp-matern32`, `gp-matern52`, `gp-periodic`, `sawtooth`, `squarewave` (plus the shared `gp` fragment) | `benchmark/synthetic/output_generator=sawtooth` |
| `benchmark/synthetic/input_generator` | `uniform`, `uniform-shift-{0,3,6,9,12,15}`, `mixturebeta` | `benchmark/synthetic/input_generator=uniform-shift-6` |
| `benchmark/image` | `base-cifar10`, `base-svhn`, `base-dtd` | `benchmark/image=base-svhn` |
| `benchmark/predprey` | `base`, `test_real`, `test_sim` | `benchmark/predprey=test_real` |
| `benchmark/kolmogorov` | `base` | — |
| `benchmark/era5` | `base` | — |
| `model/<bench>` | One file per architecture variant; see `ls conf/model/<bench>/` | `model/synthetic=sf-volterra-convcnp-f4.9-e128-l5-vr4` |
| `optimizer` | `adamw` | `optimizer=adamw` |

Run `ls conf/<group>/` to see the full member list for any group.

### Key top-level config fields

| Field | Description |
|---|---|
| `data.datasets.{train,validation,test}` | Dataset class + per-split arguments (`_target_` + kwargs) |
| `data.dataloaders.{train,validation,test}` | Per-split `batch_size` / `shuffle` / `drop_last` |
| `model` | Model class + architecture hyperparameters |
| `optimizer` | Optimizer class (`_target_`, `_partial_: true`, `lr`, etc.) |
| `phase_configs.{train,validation,test}` | One or a list of `PhaseConfig`(s) per phase. Train phases set `loss_fn`; validation/test phases set a list of `MetricSpec`. Wired by Hydra defaults groups from `conf/metrics/{train,val,test}/` — see [`utils/experiment/README.md`](utils/experiment/README.md). |
| `params.{x_dim,y_dim,embed_dim,num_layers,…}` | Model hyperparameters referenced via `${params.*}` interpolation |
| `misc.epochs` | Total training epochs |
| `misc.num_workers` / `misc.num_eval_workers` | DataLoader worker counts |
| `misc.seed` | Random seed |
| `misc.gradient_clip_val` | Gradient clipping (default 0.5) |
| `misc.wandb_logging_enabled` | Toggle W&B experiment tracking |
| `misc.wandb_run_id` | Resume / attach to an existing W&B run |
| `misc.checkpointing.local` | Save checkpoints to local disk |
| `misc.checkpointing.wandb` | Upload checkpoints as W&B artifacts |
| `misc.checkpointing.save_weights_only` | Lightning `ModelCheckpoint` flag |
| `misc.checkpointing.save_last` | Keep the most recent checkpoint in addition to best |
| `misc.checkpointing.save_periodic_every_n_epochs` | Nullable int for periodic saves |
| `misc.checkpointing.resume_from` | `null`, local path, or `wandb:<artifact>` |
| `misc.checkpointing.best.monitor` | Metric name to track for the best checkpoint (e.g. `validation_query_loglik_epoch`) |
| `misc.checkpointing.best.mode` | `max` or `min` depending on the monitored metric |
| `misc.checkpointing.best.save_top_k` | Number of best checkpoints to keep |

The checkpointing block uses null-sentinel scalars and a single
`resume_from` string rather than the nested
`enabled`/`resume.enabled`/`last.enabled`/`periodic.enabled` flag pattern.

### Checkpoint metric naming

Epoch metrics are logged as `{phase}_{cfg.name}_{spec.name}_epoch` when
`PhaseConfig.name` is set (e.g. `validation_query_loglik_epoch`), or as
`{phase}_{spec.name}_epoch` when it's not. `misc.checkpointing.best.monitor`
must match this exact string. See [`utils/experiment/README.md`](utils/experiment/README.md)
for the full naming and DDP-correctness story.

---

## Models

All models live in `nps/models/` and extend `BaseNeuralProcess` (`nps/models/base.py`). The forward interface is:

```python
output = model(xc, yc, xq)  # returns a torch.distributions.Distribution
```

### Available models

| Model | Class | Description |
|---|---|---|
| CNP | `CNP` | Conditional Neural Process — DeepSet encoder, MLP decoder |
| ACNP | `ACNP` | Attentive CNP — cross-attention encoder |
| TNP | `TNP` | Transformer Neural Process |
| TETNP | `TETNP` | Translation-Equivariant Transformer NP (full self-attention) |
| TE-Perceiver TNP | `TETNP` + `TEPerceiverEncoder` | Pseudo-token TETNP variant; Perceiver-style one-way cross-attention from fixed-size latent tokens to context |
| TE-IST TNP | `TETNP` + `TEISTransformerEncoder` | Pseudo-token TETNP variant; Induced Set Transformer-style — also updates context via cross-attention with the pseudo tokens |
| ConvCNP | `ConvCNP` | Convolutional CNP (off-the-grid inputs) |
| GridConvCNP | `GridConvCNP` | ConvCNP with mask inputs for image grids |
| SetFourierConvCNP | `SetFourierConvCNP` | Hybrid Set-Fourier convolutional NP |

`ConvCNP`/`GridConvCNP` accept different CNN backbones (`ConvNet`, `UNet`,
`FNO`) wired through config; the spectral/FNO-backbone variants
are obtained by swapping `cnn` rather than instantiating a separate
class (see e.g. `conf/model/image/convcnp-fno.yaml`). The TE-Perceiver
and TE-IST rows above are likewise encoder swaps on the same `TETNP`
class, selected via the model config.

Model configs live under `conf/model/<benchmark>/<name>.yaml`, grouped by
benchmark to avoid the cross-benchmark name collisions that would occur
with a flat layout (every benchmark has its own `cnp.yaml`, `acnp.yaml`,
…). Each file specifies the model class and its hyperparameters (hidden
dimensions, number of layers, CNN type, etc.) and uses `# @package _global_`
so `${params.*}` interpolations resolve against the composed root.

**Variant naming**: the filename under `conf/model/<bench>/` is a *config
variant name*, not a class name. Multiple variants can share the same
`_target_`: e.g. `eqtnp.yaml` and `te-eqtnp.yaml` both instantiate
`nps.models.TNP`/`TETNP` with different encoder/decoder/convolution
wiring, and every `sf-volterra-convcnp-*` variant instantiates
`nps.models.SetFourierConvCNP` with a different frequency-grid choice.
The model name (what you pass to `model/<bench>=`) tracks the variant,
not the class.

### Core building blocks (`nps/core/`)

**Convolutions** (`nps/core/convolutions/`):

| Class | Description |
|---|---|
| `ConvNd` | Standard convolution |
| `VolterraConvNd` | 1st + 2nd order Volterra (nonlinear) convolution |
| `SpectralConv` | FFT-based spectral convolution |
| `SetConv` | Permutation-invariant set convolution |
| `SetFourierConv` | Set convolution + Fourier spectral convolution |
| `SetFourierVolterraConv` | Set + Fourier + Volterra hybrid |


**CNNs** (`nps/core/cnns/`): `ConvNet`, `FNO` (Fourier Neural Operator), `SetFourierConvNet` (Set-Fourier), `UNet`

**Encoders** (`nps/core/encoders/`): `CNPEncoder`, `ConvCNPEncoder`, `GridConvCNPEncoder`, `ACNPEncoder`, `TNPEncoder`, `TETNPEncoder`, `SetFourierConvCNPEncoder`

**Transformers** (`nps/core/transformers/`): `PerceiverEncoder`, `ISTransformerEncoder` (inducing-set), `TETransformerEncoder` (translation-equivariant)

---

## Datasets

All datasets follow a **Processor → Dataset → DataLoader** pattern. See [`utils/data/README.md`](utils/data/README.md) for full details and multiprocessing safety notes.

### Dataset summary

| Domain | Processor | Dataset | Batch type | Style |
|---|---|---|---|---|
| Image (CIFAR-10, DTD, SVHN) | `CIFARDataProcessor`, etc. | `ImageDataset` | `ImageBatch` | Map-style |
| Kolmogorov flow | `KolmogorovDataProcessor` | `KolmogorovDataset` | `KolmogorovBatch` | Map-style |
| ERA5 climate | `ERA5DataProcessor` | `ERA5Dataset` | `ERA5Batch` | Iterable |
| Synthetic functions | `SyntheticInputGenerator` + output generator | `SyntheticDataset` | `SyntheticBatch` | Iterable |
| Predator-prey (sim) | — | `PredPreySimDataset` | `PredPreyBatch` | Iterable |
| Predator-prey (real) | — (auto-download) | `PredPreyRealDataset` | `PredPreyBatch` | Iterable |

### Standard batch fields

Every batch type extends `BaseBatch` and provides:

| Field | Description |
|---|---|
| `xc` | Context input coordinates `(B, Nc, x_dim)` |
| `yc` | Context observations `(B, Nc, y_dim)` |
| `xq` | Query input coordinates `(B, Nq, x_dim)` |
| `yq` | Query observations `(B, Nq, y_dim)` |

Batches support `.to(device)` to move all tensor fields.

### Synthetic output generators

Available in `conf/benchmark/synthetic/output_generator/` (select via the
`benchmark/synthetic/output_generator=<name>` group override):

- Gaussian processes: `gp-rbf`, `gp-matern12`, `gp-matern32`, `gp-matern52`, `gp-periodic`
- Deterministic waveforms: `sawtooth`, `squarewave`
- Shared fragment: `gp` (base Gaussian-process scaffold pulled in by the `gp-*` variants via their own defaults list)

Input generators live at `conf/benchmark/synthetic/input_generator/` — `uniform`,
`mixturebeta`, plus the translation-shift variants
used by the rebuttal eval scripts (`uniform-shift-{0,3,6,9,12,15}`).

### ERA5 notes

- Raw data must be placed in `dataset-files/era5/` as NetCDF files.
- On first run the processor exports normalized `.npy` cache files (coordinate-hash-named).
- Subsequent runs load from cache. On a cluster with multiple ranks, rank 0 creates the cache behind a barrier.
- `num_workers > 0` is safe — the dataset uses `mmap_mode='r'` with `__getstate__`/`__setstate__` for pickle safety.

---

## Training Framework

The training loop lives in `utils/experiment/lightning_wrapper.py` as `LitWrapper` (a `pl.LightningModule`).

### PhaseConfig

Each training phase (train / validation / test) is described by one or
more `PhaseConfig` objects:

```python
@dataclass
class PhaseConfig:
    metric_specs: list[MetricSpec]            # validation/test specs
    loss_fn: Callable | None                  # (model, batch) -> scalar; required on train
    name: str | None                          # prefix for logged metric names
```

Training presets wire `loss_fn` (typically
`utils.experiment.metrics.losses.nll_loss`); validation/test presets
wire `metric_specs`. Each `MetricSpec` pairs a per-sample
`metric_fn(likelihood, batch) -> Tensor` with an `accumulator`
(`TaskMeanAccumulator`, `TaskRMSEAccumulator`, etc.) and pins
`eval_on: "query"` or `"context"`. The Lightning loop groups specs by
`eval_on`, runs `as_batch(batch, eval_on=...)` and a single forward per
group, then dispatches each spec's `metric_fn` over the group's
likelihood. The accumulator's epoch-end `compute()` runs the canonical
task-weighted reduction (DDP-correct via `torchmetrics`'
`dist_reduce_fx="sum"` on its `sum` and `count` buffers).

The two `eval_on` slots support evaluating both held-out targets and
reconstructed context in one pass — the validation presets under
`conf/metrics/val/` ship a `[query, context]` split out of the box. See
[`utils/experiment/README.md`](utils/experiment/README.md) for the full
metric / accumulator / reducer / forward-wrapper layout.

Hydra wires phase configs via defaults groups:

```yaml
defaults:
  - /metrics/train: nll
  - /metrics/val: synthetic       # or "standard" for non-synthetic benchmarks
  - /metrics/test: synthetic
  - _self_
```

Each preset under `conf/metrics/{train,val,test}/` carries a
`# @package phase_configs.<phase>` directive so the composed shape lands
in the right slot of the training config.

### Logging and checkpointing

- **Weights & Biases**: set `misc.wandb_logging_enabled=true`. Run name,
  project, and entity default values live in each benchmark's
  `conf/benchmark/<bench>/base.yaml` under `misc.project`, `misc.name`,
  `misc.wandb_user`. Override per-run via
  `misc.project=my-project misc.name=my-run`. The logger is initialized
  *before* Hydra instantiate fires, so dataset/model construction stdout
  is captured by wandb's console redirect.
- **Local checkpoints**: set `misc.checkpointing.local=true` (default on most
  benchmark bases).
- **W&B artifact upload**: set `misc.checkpointing.wandb=true`.
- **Resuming**: set `misc.checkpointing.resume_from=<path-or-artifact>`
  (local path or `wandb:<artifact>` URI). It's a single nullable scalar
  — `null` for fresh training, a string to resume.
- **Best checkpoint**: `misc.checkpointing.best.monitor`,
  `misc.checkpointing.best.mode` (`max` / `min`), and
  `misc.checkpointing.best.save_top_k`.

### Where artifacts land

Every run writes its checkpoints, metrics, plots, and Hydra's own
resolved-config snapshot under `misc.artifacts_dir`, which is pinned as
`hydra.run.dir` in `conf/config.yaml` (so `os.getcwd()` stays at the repo
root). The default path template in each benchmark's base is something
like:

```
artifacts/<experiment_name>/x=<x_dim>d_y=<y_dim>d/model=<model_name>/seed=<seed>/
├── .hydra/           # config.yaml, hydra.yaml, overrides.yaml
├── checkpoints/      # Lightning ModelCheckpoint output
├── metrics/          # per-checkpoint test results (JSON)
├── plots/            # matplotlib figures if plot_fn is wired
└── train.log         # stdout redirect
```

Multi-run sweeps land under `${misc.artifacts_dir}/sweeps/<hydra.job.num>/`.
Override `misc.artifacts_dir=/some/other/path` at the CLI if you want to
redirect a specific run.

---

## Evaluation

`eval.py` loads a trained checkpoint and runs evaluation on the test set:

```bash
# Attach to an existing W&B run (was --wandb-run-id in the legacy CLI)
python eval.py +experiment=synthetic/default model/synthetic=convcnp-unet \
    benchmark/synthetic/output_generator=sawtooth \
    misc.wandb_run_id=entity/project/run_id

# CIFAR-10
python eval.py +experiment=image/default model/image=convcnp-resnet \
    misc.wandb_run_id=entity/project/run_id
```

The legacy `--wandb-run-id` argparse flag is gone — it's a normal Hydra
override now (`misc.wandb_run_id=<id>`). The script evaluates each
available checkpoint type (best, last, periodic) and logs results back
to W&B or prints them locally.

---

## Extending the Codebase

### Adding a new dataset

**Python side:**

1. Create `utils/data/<name>/processor.py` — handles raw data loading, normalization, caching.
2. Create `utils/data/<name>/dataset.py` — extracts tensors from the processor in `__init__`, drops the processor reference. Extend `BaseMapDataset` (finite data) or `BaseIterableDataset` (generated/streamed).
3. Define a batch dataclass extending `BaseBatch` with your dataset's fields.
4. Register a forward wrapper in `utils/experiment/forward_wrappers.py` if your new model or batch combination isn't already covered by the existing `cnp_forward_wrapper` registration. The metric path is uniform — `metric_fn(likelihood, batch)` works for any batch type that exposes `xc/yc/xq/yq`.
5. Add a plotter in `utils/plot_fn/<name>.py` if needed.
6. Export classes from `utils/data/<name>/__init__.py` and `utils/data/__init__.py`.

**Config side:**

7. Create `conf/benchmark/<name>/base.yaml` with the `# @package _global_`
   header. Populate `data.datasets.{train,validation,test}` with
   `_target_` references to your new dataset class, plus any
   per-split `data.dataloaders.*` settings. Also set the benchmark-wide
   `misc.experiment_name`, `misc.artifacts_dir`, optional `plot_fn`,
   and W&B defaults (`misc.project`, `misc.wandb_user`) — mirror one of
   the existing benchmarks (e.g. `conf/benchmark/kolmogorov/base.yaml`) as
   a template.
8. Create model fragments at `conf/model/<name>/<arch>.yaml` for each
   architecture variant you want to run.
9. Create a single parameterized experiment composer at
   `conf/experiment/<name>/default.yaml` whose `defaults:` list pulls in
   `- /benchmark/<name>: base`, `- /optimizer: adamw`, and a default
   `- /model/<name>: <arch>`. See the "Adding a new model" template
   below for exact syntax.
10. Smoke the composition: `python train.py +experiment=<name>/default
    model/<name>=<arch> misc.epochs=1`. To validate the Hydra scaffolding
    (cwd, run-dir interpolation, custom resolvers) without instantiating
    anything, run `python tools/hydra_smoke.py`.

### Adding a new model

1. Implement the model in `nps/models/<name>.py` extending `BaseNeuralProcess`.
2. Implement `forward(xc, yc, xq) -> Distribution`.
3. Register a forward wrapper in `utils/experiment/forward_wrappers.py` only if the model has a non-standard call signature (the default `cnp_forward_wrapper` already covers any `(xc, yc, xq)` shape).
4. Add a model fragment `conf/model/<benchmark>/<name>.yaml` with the
   `# @package _global_` header. Populate `_target_` wiring + any
   `${params.*}` / `${eval:…}` interpolations the architecture needs.
5. Add (or update) the parameterized experiment composer
   `conf/experiment/<benchmark>/default.yaml`. The benchmark's composer
   already pulls in the benchmark base, the optimizer, and a default
   `model/<benchmark>` entry — your new model fragment is selected via
   the CLI `model/<benchmark>=<name>` override, no new composer file
   needed. The composer for synthetic looks like:
   ```yaml
   # @package _global_
   defaults:
     - /benchmark/synthetic: base
     - /optimizer: adamw
     - /model/synthetic: cnp
     - /benchmark/synthetic/output_generator: gp-rbf
     - /benchmark/synthetic/input_generator: uniform
     - _self_
   ```
6. Run `python train.py +experiment=<benchmark>/default model/<benchmark>=<name>`
   to test the composition.

### Adding a new convolution variant

1. Implement in `nps/core/convolutions/<name>.py`.
2. Add a convolution block wrapper in `nps/core/convolution_blocks/`.
3. Integrate into a CNN class in `nps/core/cnns/`.
4. No new config fragment is needed — convolutions are nested
   `_target_` entries inside existing `conf/model/<bench>/<arch>.yaml`
   files. To use the new variant, grep for the old convolution class
   in `conf/model/` and swap the `_target_` (or create a new model
   fragment that references it).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{
  mohseni2026revisiting,
  title={Revisiting Neural Processes via Fourier Transform and Volterra Series},
  author={Peiman Mohseni and Nick Duffield and Raymond K. W. Wong},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=UEeBGrOGa8}
}
```

## Acknowledgments

This codebase adapts ideas and code from several open-source projects, which we
gratefully acknowledge:

- [**TETNP** — Translation-Equivariant Transformer Neural Processes](https://github.com/cambridge-mlg/tetnp)
  (Cambridge MLG): the translation-equivariant transformer / attention layers and
  the TE-TNP model family.
- [**SDA** — Score-based Data Assimilation](https://github.com/francois-rozet/sda)
  (François Rozet): the Kolmogorov-flow data generation (the `MarkovChain` /
  `KolmogorovFlow` simulation).
- [**neuralprocesses**](https://github.com/wesselb/neuralprocesses)
  (Wessel Bruinsma): convolutional NP building blocks (UNet / `ConvBlock` coders)
  and the predator–prey (Lotka–Volterra → Hudson Bay hare–lynx) data pipeline.

Each project remains under its own license; please consult the linked repositories.

## License

This project is released under the [MIT License](LICENSE).
