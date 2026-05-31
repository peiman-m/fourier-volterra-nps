"""Metrics subsystem.

Pure metric functions live in ``functions``; batch-local step reductions
in ``reducers``; epoch-level ``torchmetrics.Metric`` accumulators in
``accumulators``; the composed ``MetricSpec`` dataclass in ``spec``;
optimizer-facing loss callables in ``losses``.
"""
from .accumulators import (
    BaseAccumulator,
    BatchMeanAccumulator,
    BatchStdAccumulator,
    CatAccumulator,
    SampleMeanAccumulator,
    SampleRMSEAccumulator,
    SampleStdAccumulator,
    TaskMeanAccumulator,
    TaskRMSEAccumulator,
    TaskStdAccumulator,
)
from .reducers import (
    BaseReducer,
    MeanReducer,
    MedianReducer,
    NoReducer,
    QuantileReducer,
    RMSEStepReducer,
    StdReducer,
)
from .spec import MetricFn, MetricSpec

__all__ = [
    "MetricSpec",
    "MetricFn",
    "BaseAccumulator",
    "SampleMeanAccumulator",
    "SampleStdAccumulator",
    "BatchMeanAccumulator",
    "BatchStdAccumulator",
    "TaskMeanAccumulator",
    "TaskStdAccumulator",
    "CatAccumulator",
    "SampleRMSEAccumulator",
    "TaskRMSEAccumulator",
    "BaseReducer",
    "MeanReducer",
    "RMSEStepReducer",
    "StdReducer",
    "MedianReducer",
    "QuantileReducer",
    "NoReducer",
]
