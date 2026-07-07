from typing import cast

import gpytorch
import torch

from ..base import GroundTruthPredictor


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        train_inputs: torch.Tensor | None = None,
        train_targets: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # gpytorch module __call__ leaks Tensor | Distribution | LinearOperator;
        # ConstantMean returns a Tensor mean.
        return gpytorch.distributions.MultivariateNormal(
            cast(torch.Tensor, mean_x), covar_x
        )


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        self.kernel = kernel
        self.likelihood = likelihood

        self._result_cache: dict[str, torch.Tensor] | None = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xq: torch.Tensor,
        yq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        # Move devices.
        old_device = xc.device
        device = self.kernel.device
        xc = xc.to(device)
        yc = yc.to(device)
        xq = xq.to(device)
        if yq is not None:
            yq = yq.to(device)

        if yq is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xq_) in enumerate(zip(xc, yc, xq)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            likelihood = gp_model.likelihood
            assert likelihood is not None
            gp_model.eval()
            likelihood.eval()
            with torch.no_grad():

                dist = gp_model(xq_)
                pred_dist = likelihood.marginal(dist)
                if yq is not None:
                    gt_loglik = pred_dist.to_data_independent_dist().log_prob(
                        yq[i, ..., 0]
                    )
                    gt_loglik_list.append(gt_loglik)

                mean_list.append(pred_dist.mean)
                try:
                    std_list.append(pred_dist.stddev)
                except RuntimeError:
                    std_list.append(
                        cast(torch.Tensor, pred_dist.covariance_matrix).diagonal() ** 0.5
                    )

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        # Cache for deterministic validation batches.
        # Cache only when yq is supplied; the x_plot path passes yq=None.
        if yq is not None:
            # gt_loglik is a Tensor here (yq is not None => gt_loglik_list is
            # populated above); cast to the cache's declared value type.
            self._result_cache = cast(
                "dict[str, torch.Tensor]",
                {"mean": mean, "std": std, "gt_loglik": gt_loglik},
            )

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xq = xq.to(old_device)
        if yq is not None:
            yq = yq.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:

        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=self.kernel,
        )
        likelihood = gp_model.likelihood
        assert likelihood is not None
        gp_model.eval()
        likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = likelihood(f)
            y = dist.sample()
            return y[..., None]
