# Neural Process Experiment Framework

This directory contains the experiment framework — a PyTorch Lightning
training loop wrapped around a registry-dispatched forward call and a
``MetricSpec``-driven metric pipeline. The metrics subsystem is
task-weighted by design: the shipped val/test configs use
``TaskMeanAccumulator`` / ``TaskStdAccumulator`` / ``TaskRMSEAccumulator``,
which reduce each batch to one value per task and then weight every task
equally — the meta-learning "expected metric on a new task", reported as
mean ± across-task std. All accumulators keep a running ``sum`` / ``count``
(reduced with ``dist_reduce_fx="sum"``) so the epoch value is exact under
DDP and multi-batch accumulation rather than a biased mean-of-batch-means.
Sample-weighted (``SampleMeanAccumulator``, every point equal) and
batch-weighted (``BatchMeanAccumulator``) families are also available for
configs that want them.

## Layout

```
utils/experiment/
├── README.md                    # This file
├── lightning_wrapper.py         # LitWrapper + PhaseConfig
├── forward_wrappers.py          # (model, batch) → likelihood registry
├── metrics/
│   ├── spec.py                  # MetricSpec dataclass
│   ├── accumulators.py          # Sample/Batch/Task {Mean,Std} + Cat + Sample/Task RMSE
│   ├── reducers.py              # Mean / RMSEStep / Std / Median / Quantile / NoReducer
│   ├── functions.py             # log_likelihood, neg_log_likelihood, squared_error, absolute_error, gaussian_crps_closed_form, gt_log_likelihood
│   └── losses.py                # nll_loss, mse_loss (PhaseConfig.loss_fn callables)
├── callbacks/
│   ├── plotter.py               # PlotterCallback — runs BaseNeuralProcessPlotter every-N validation epochs
│   └── performance.py           # LogPerformanceCallback — throughput + memory timing
├── registry/                    # Generic (model_cls × batch_cls) wrapper registry
├── helpers.py                   # ReductionType + TensorProcessor (tensor-shaped reductions)
├── setup.py                     # Hydra-side initialization (experiment, callbacks, loggers, dataloaders)
└── utils.py                     # WandB / checkpoint / NumpyEncoder helpers
```

## Core Components

### `PhaseConfig`

Per-phase configuration: a list of ``MetricSpec`` plus an optional
``loss_fn``. Training phases set ``loss_fn``; validation/test phases set
``metric_specs``.

```python
class PhaseConfig:
    def __init__(
        self,
        metric_specs: list[MetricSpec] | None = None,    # validation/test specs
        loss_fn: Callable[[nn.Module, BaseBatch], Tensor] | None = None,
        name: str | None = None,                          # logged-key prefix (e.g. "query", "context")
    ) -> None: ...
```

A list of ``PhaseConfig`` per phase is supported — useful for the
``[query, context]`` split that lets one validation pass score both
held-out targets and reconstructed context with one model forward per
group.

### `LitWrapper`

The Lightning module. ``_run_step`` does the work:

1. Group ``cfg.metric_specs`` by ``eval_on`` (``"query"`` / ``"context"``).
2. For each group, run ``as_batch(batch, eval_on=...)`` once, then a
   single forward through the registered wrapper.
3. For each spec, ``raw = spec.metric_fn(likelihood, batch)`` produces a
   per-sample tensor (e.g. ``[B, N, Dy]``). The result feeds into the
   spec's ``accumulator`` which Lightning auto-``compute``/``reset``s at
   epoch end.
4. Training phases additionally call ``cfg.loss_fn(model, batch)`` and
   ``.backward()`` on the returned scalar.

Metric accumulators are stored in an ``nn.ModuleDict`` so ``.to(device)``
reaches their state buffers and ``torchmetrics``' built-in DDP all-reduce
on ``sum`` + ``count`` delivers a true global mean across ranks.

### `MetricSpec`

```python
MetricFn = Callable[[Distribution, BaseBatch], Tensor]

@dataclass
class MetricSpec:
    metric_fn: MetricFn                       # (predictive Distribution, batch) -> per-sample raw
    name: str
    accumulator: BaseAccumulator
    step_reducer: BaseReducer | None = None
    eval_on: EvalOn = "query"
    prog_bar: bool = False
    sync_dist: bool = False
```

``metric_fn`` receives the ``torch.distributions.Distribution`` returned
by the model's forward pass (constructed from ``BaseLikelihood``), not a
``BaseLikelihood`` instance — the forward wrapper has already called
the likelihood on the model output by the time the metric sees it.

``metric_fn`` returns the rawest honest per-sample shape — no reduction.
Reduction happens at two layers below it:

- ``accumulator`` does the epoch reduction (a ``torchmetrics.Metric``
  that updates a running ``sum`` and ``count`` per call).
- ``step_reducer`` (optional) turns each batch's raw into a single
  scalar for step-level logging — only required when you want
  ``on_step=True`` telemetry.

Pairings used by the shipped val/test configs (each ``*_std`` is the
across-task error bar for the metric above it):

| Logged metric             | ``metric_fn``                       | ``accumulator``               |
|---------------------------|-------------------------------------|-------------------------------|
| ``loglik``                | ``log_likelihood``                  | ``TaskMeanAccumulator``       |
| ``loglik_std``            | ``log_likelihood``                  | ``TaskStdAccumulator``        |
| ``rmse``                  | ``squared_error``                   | ``TaskRMSEAccumulator``       |
| ``crps``                  | ``gaussian_crps_closed_form``       | ``TaskMeanAccumulator``       |
| ``crps_std``              | ``gaussian_crps_closed_form``       | ``TaskStdAccumulator``        |
| ``gt_loglik`` (synth.)    | ``gt_log_likelihood``               | ``TaskMeanAccumulator``       |

Swap in the ``Sample*`` (point-weighted) or ``Batch*`` (batch-weighted)
accumulators for a metric if you need a different weighting.

### Forward Wrappers

Dispatched on ``(type(model), type(batch))``. The caller in
``_run_step`` has already routed the batch through
``as_batch(batch, eval_on=...)``, so wrappers no longer parse a
``query_subset`` kwarg — ``batch.xq`` / ``batch.yq`` /
``batch.mq_grid`` already point at the correct slot.

```python
@register_forward_wrapper(
    (CNP, ACNP, TNP, TETNP, ConvCNP, SetFourierConvCNP),
    (Batch, SyntheticBatch, PredPreyBatch, ImageBatch, KolmogorovBatch, ERA5Batch),
)
def cnp_forward_wrapper(model, batch):
    return model(xc=batch.xc, yc=batch.yc, xq=batch.xq)


@register_forward_wrapper(GridConvCNP, (ImageBatch, KolmogorovBatch, ERA5Batch))
def grid_xy_cnp_forward_wrapper(model, batch):
    return model(y_mc=batch.y_mc_grid, y=batch.y_grid, y_mq=batch.mq_grid)
```

Every batch type carries ``xc`` / ``yc`` / ``xq`` / ``yq``; the
``_grid``-bearing batches (``ImageBatch``, ``KolmogorovBatch``,
``ERA5Batch``) carry ``x_grid`` / ``y_grid`` / ``mc_grid`` / ``mq_grid``
in addition. The per-sample fields share one naming convention across
every batch type, so a single forward wrapper covers every non-grid
model call.

### Callbacks

- ``PlotterCallback`` — wraps a ``BaseNeuralProcessPlotter``, caches the
  first ``num_batches`` validation batches per epoch (rank 0), and
  invokes the plotter every ``every_n_val_epochs``.
- ``LogPerformanceCallback`` — emits throughput / memory telemetry.

Both are wired by ``setup.initialize_callbacks`` from ``misc.plot_fn``,
``misc.plot_interval``, and ``misc.num_plots``.

## Hydra wiring

Validation/test presets live under ``conf/metrics/{train,val,test}/``
and use Hydra's ``# @package phase_configs.<phase>`` directive so a
model yaml's ``defaults`` block collapses to four lines:

```yaml
defaults:
  - /metrics/train: nll
  - /metrics/val: synthetic      # or "standard" for non-synthetic benchmarks
  - /metrics/test: synthetic
  - _self_
```

``train/nll.yaml`` wires ``loss_fn: utils.experiment.metrics.losses.nll_loss``;
``val/synthetic.yaml`` and ``val/standard.yaml`` ship the
``[query, context]`` two-``PhaseConfig`` split with loglik / rmse / crps
(plus ``gt_loglik`` in synthetic's query phase).

## Logged-metric naming

Epoch metrics log as ``f"{phase}_{prefix}_{spec.name}_epoch"`` where
``prefix = cfg.name`` (empty if ``cfg.name`` is None). For a
``cfg.name = "query"`` and ``spec.name = "loglik"`` on the validation
phase, the W&B key is ``validation_query_loglik_epoch``.

When wiring ``misc.checkpointing.best.monitor``, match the exact key:

```yaml
checkpointing:
  best:
    monitor: "validation_query_loglik_epoch"
    mode: "max"
```

## DDP notes

Accumulators are ``torchmetrics.Metric`` subclasses with
``dist_reduce_fx="sum"`` on ``sum`` and ``count``, so ``.compute()``
all-reduces both then divides — the true global mean (over tasks for the
default ``Task*`` accumulators, over points for ``Sample*``). Pass
``sync_dist=False`` (the default) on the epoch log call; passing
``sync_dist=True`` on top of a ``torchmetrics.Metric`` causes a
double-reduce and Lightning warns.

The ``DistributedSampler`` tail-padding (Lightning's
``use_distributed_sampler=True`` default) duplicates a handful of
samples to balance ranks; the accumulators count those
duplicates, biasing the mean by ``O(world_size / N)`` when the dataset
size isn't divisible by ``world_size × batch_size``. Tolerate (typical),
or pass a custom sampler with ``use_distributed_sampler=False``.

``train_loss`` itself is still logged via Lightning's ``sync_dist=True``
mean-of-batch-means; replacing that with an explicit
``SampleMeanAccumulator`` would be a future improvement.

## Extending

To support a new model:

1. Register a forward wrapper if the new model's ``forward`` signature
   differs from the ones above; otherwise reuse ``cnp_forward_wrapper``.
2. The metric path needs no per-model wrapper — ``metric_fn(likelihood,
   batch)`` is uniform.

To support a new batch type:

1. Define the dataclass with ``xc`` / ``yc`` / ``xq`` / ``yq`` (plus
   ``x_grid`` / ``y_grid`` / ``mc_grid`` / ``mq_grid`` if the data has a
   grid form).
2. Register an ``as_batch`` singledispatch handler that swaps
   ``xq``/``yq``/``mq_grid`` ← ``xc``/``yc``/``mc_grid`` for the
   ``eval_on="context"`` case.
3. Register a forward wrapper for ``(model_cls × batch_cls)`` only if
   the existing ``cnp_forward_wrapper`` registration doesn't already
   cover it.

To add a new metric:

1. Write a function ``(likelihood, batch) -> Tensor`` that returns
   the per-sample raw shape, no reduction.
2. Pair it with an accumulator (``TaskMean`` for mean-style,
   ``TaskRMSE`` for root-mean-square, ``Cat`` for median/quantile).
3. Wire it via a ``MetricSpec`` in the relevant ``conf/metrics/`` preset.

## Dependencies

- PyTorch Lightning, PyTorch, torchmetrics
- Hydra (for configuration)
- ``nps.models`` / ``nps.likelihoods``
