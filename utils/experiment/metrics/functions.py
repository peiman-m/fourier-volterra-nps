"""Pure metric functions — ``(Distribution, BaseBatch) -> raw tensor``.

Stateless. Each returns the rawest honest granularity (per-sample,
no reduction). The Lightning loop hands the function the predictive
distribution that came out of the forward wrapper plus the batch the
metric is scoring against; ``batch.yq`` is the held-out query slot.

Return-shape contract:

- ``log_likelihood``       → ``[B, N, Dy]`` (or whatever shape ``Distribution.log_prob(yq)`` returns)
- ``neg_log_likelihood``   → same shape, sign flipped
- ``squared_error``        → ``[B, N, Dy]``
- ``absolute_error``       → ``[B, N, Dy]``
- ``gaussian_crps_closed_form`` → ``[B, N, Dy]``
- ``gt_log_likelihood``    → ``[B, N]`` from the synthetic GP ground truth (Dy=1 collapsed)

``gt_log_likelihood`` reads ``batch.gt_pred`` directly. It raises
``NotImplementedError`` when no ground-truth predictor is attached, so
specs only land on synthetic-data experiments.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution

from utils.data.base import BaseBatch


def log_likelihood(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample log-likelihood of ``batch.yq`` under ``likelihood``."""
    return likelihood.log_prob(batch.yq)


def neg_log_likelihood(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample negative log-likelihood — sign-flipped ``log_likelihood``."""
    return -likelihood.log_prob(batch.yq)


def squared_error(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample ``(mean - yq) ** 2``. Pair with ``SampleRMSEAccumulator`` for RMSE."""
    return (likelihood.mean - batch.yq) ** 2


def absolute_error(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample ``|mean - yq|``."""
    return (likelihood.mean - batch.yq).abs()


def gaussian_crps_closed_form(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample closed-form Gaussian CRPS — no reduction.

    ``CRPS(N(μ, σ²), y) = σ * [z(2Φ(z) - 1) + 2φ(z) - 1/√π]`` where
    ``z = (y - μ) / σ``, ``Φ`` the standard-normal CDF, ``φ`` its PDF.
    """
    mu = likelihood.mean
    # ``stddev`` is the base-Distribution property; for the Gaussian
    # predictive it equals ``scale`` (Normal.stddev returns scale).
    sigma = likelihood.stddev
    z = (batch.yq - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def gt_log_likelihood(likelihood: Distribution, batch: BaseBatch) -> Tensor:
    """Per-sample ground-truth log-likelihood from ``batch.gt_pred``.

    ``likelihood`` is unused — kept in the signature so every metric
    function shares the same shape. Raises ``NotImplementedError`` when
    no ground-truth predictor is attached, so the spec must only be wired
    onto synthetic experiments.
    """
    del likelihood
    gt_pred = getattr(batch, "gt_pred", None)
    if gt_pred is None:
        raise NotImplementedError(
            f"gt_log_likelihood requires batch.gt_pred; got {type(batch).__name__} "
            "with no attached ground-truth predictor (synthetic-data only)."
        )
    _, _, gt_loglik = gt_pred(xc=batch.xc, yc=batch.yc, xq=batch.xq, yq=batch.yq)
    return gt_loglik
