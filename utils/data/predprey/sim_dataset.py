from dataclasses import dataclass

import torch

from ..base import BaseIterableDataset, Batch


@dataclass
class PredPreyBatch(Batch):
    """Batch class for PredPrey data."""

    x_dense: torch.Tensor  # Dense version of input features
    y_dense: torch.Tensor  # Dense version of output values


class PredPreySimDataset(BaseIterableDataset):
    """
    A PyTorch-based data generator for the stochastic Lotka-Volterra predator-prey model.

    This class simulates trajectories of interacting predator and prey populations governed by
    the following stochastic differential equations (SDEs):

        dX_t = α·X_t·dt - β·X_t·Y_t·dt + σ·X_t^ν·dW_t^(1)
        dY_t = -γ·Y_t·dt + δ·X_t·Y_t·dt + σ·Y_t^ν·dW_t^(2)

    where:
        - X_t: Prey population at time t
        - Y_t: Predator population at time t
        - α (alpha): Prey growth rate
        - β (beta): Rate at which prey are killed by predators
        - γ (gamma): Predator death rate
        - δ (delta): Predator growth rate from consuming prey
        - σ (sigma): Noise intensity
        - ν (nu): Noise exponent, scaling the population-dependent noise
        - dW_t^(i): Independent Wiener processes (Brownian motions)

    The generator supports dense trajectory simulation and flexible sampling of context and query
    sets for use in meta-learning or neural process training.

    Reference:
        - https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/data/predprey.py
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nq: int,
        max_nq: int,
        xc_range: tuple[float, float] | None = None,
        xq_range: tuple[float, float] | None = None,
        trajectory_pool_size: int | None = None,
        # Model parameters ranges
        alpha_range: tuple[float, float] = (0.2, 0.8),
        beta_range: tuple[float, float] = (0.04, 0.08),
        gamma_range: tuple[float, float] = (0.8, 1.2),
        delta_range: tuple[float, float] = (0.04, 0.08),
        sigma_range: tuple[float, float] = (0.5, 10.0),
        scale_range: tuple[float, float] = (1.0, 5.0),
        noise_exponent: float = 1.0 / 6.0,
        # Initial conditions and time ranges
        prey_init_range: tuple[float, float] = (5.0, 100.0),
        pred_init_range: tuple[float, float] = (5.0, 100.0),
        t_simulation_range: tuple[float, float] = (-10.0, 100.0),
        # Scaling factors
        population_rescale_factor: float = 1.0,
        time_rescale_factor: float = 1.0,
        # Simulation parameters
        sim_resolution: float = 0.05,
        max_sim_steps: int = 5000,
        max_prey_population: float | None = None,
        max_pred_population: float | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the stochastic predator-prey data generator.

        Args:
            min_nc: Minimum number of context points per sample.
            max_nc: Maximum number of context points per sample.
            min_nq: Minimum number of query points per sample.
            max_nq: Maximum number of query points per sample.
            xc_range: Optional time interval from which to sample context points, in raw (unrescaled) time.
            xq_range: Optional time interval from which to sample query points, in raw (unrescaled) time.
            trajectory_pool_size: Number of trajectories to simulate in parallel and cache for reuse.

            alpha_range: Range of values for prey growth rate α.
            beta_range: Range of values for predation rate β.
            gamma_range: Range of values for predator death rate γ.
            delta_range: Range of values for predator growth rate δ.
            sigma_range: Range of values for noise intensity σ.
            scale_range: Range for global scaling of trajectory magnitudes.
            noise_exponent: Exponent ν used in population-dependent noise scaling (e.g., 1/6 for weak noise).

            prey_init_range: Range for initial prey population.
            pred_init_range: Range for initial predator population.
            t_simulation_range: Time span of simulation before rescaling.

            population_rescale_factor: Scaling factor applied to prey and predator populations.
            time_rescale_factor: Scaling factor applied to time values (after simulation).

            sim_resolution: Step size for numerical integration.
            max_sim_steps: Maximum number of steps to simulate.
            max_prey_population: Optional cap on prey population during simulation to prevent divergence.
            max_pred_population: Optional cap on predator population during simulation.

            **kwargs: Additional keyword arguments passed to the base class.

        Raises:
            ValueError: If any provided argument is invalid or inconsistent.
        """
        super().__init__(**kwargs)

        # Validate and initialize point count parameters
        if None in (min_nc, max_nc, min_nq, max_nq):
            raise ValueError(
                "All point count parameters (min_nc, max_nc, min_nq, max_nq) must be specified"
            )

        if min_nc > max_nc:
            raise ValueError(f"min_nc ({min_nc}) must be <= max_nc ({max_nc})")

        if min_nq > max_nq:
            raise ValueError(f"min_nq ({min_nq}) must be <= max_nq ({max_nq})")

        self.min_nc = int(min_nc)
        self.max_nc = int(max_nc)
        self.min_nq = int(min_nq)
        self.max_nq = int(max_nq)

        # Store optional raw-time sampling ranges
        self.xc_range = xc_range
        self.xq_range = xq_range

        # SDE parameters
        self.noise_exponent = noise_exponent
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.delta_range = delta_range
        self.sigma_range = sigma_range
        self.scale_range = scale_range

        # Trajectory parameters
        self.prey_init_range = prey_init_range
        self.pred_init_range = pred_init_range
        self.max_prey_population = max_prey_population
        self.max_pred_population = max_pred_population
        self.t_simulation_range = t_simulation_range

        self.population_rescale_factor = population_rescale_factor
        self.time_rescale_factor = time_rescale_factor

        self.sim_resolution = sim_resolution
        self.max_sim_steps = max_sim_steps

        # Trajectory pool management
        self.trajectory_pool_size = trajectory_pool_size or self.batch_size
        self._trajectory_pool_x: torch.Tensor | None = None
        self._trajectory_pool_y: torch.Tensor | None = None
        self._trajectory_pool_num_left: int = 0

        print(
            f'[{type(self).__name__}] '
            f'nc=[{min_nc}, {max_nc}], nq=[{min_nq}, {max_nq}], '
            f't={t_simulation_range}, pool={self.trajectory_pool_size}'
        )

    def _sample_lotka_volterra_params(
        self, batch_size: int = 16
    ) -> dict[str, torch.Tensor]:
        """
        Generate random parameters for the predator-prey model.

        Args:
            batch_size: Number of parameter sets to generate

        Returns:
            Dictionary containing parameter tensors (alpha, beta, delta, gamma, sigma, scale)
        """
        # Generate all random values in one operation
        rand = torch.rand(6, batch_size)

        # Apply ranges to parameters
        return {
            "alpha": self._shift_scale(rand[0], self.alpha_range),
            "beta": self._shift_scale(rand[1], self.beta_range),
            "gamma": self._shift_scale(rand[2], self.gamma_range),
            "delta": self._shift_scale(rand[3], self.delta_range),
            "sigma": self._shift_scale(rand[4], self.sigma_range),
            "scale": self._shift_scale(rand[5], self.scale_range),
        }

    def _predprey_step(
        self,
        x_y: torch.Tensor,
        t: float,
        dt: float,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Perform one step of the stochastic Lotka-Volterra simulation using Euler-Maruyama method.

        Args:
            x_y: Current population state [batch_size, 2] (prey, predator)
            t: Current time
            dt: Time step
            alpha: Prey growth rate
            beta: Prey death rate due to predation
            delta: Predator growth rate from predation
            gamma: Predator death rate
            sigma: Noise intensity

        Returns:
            Tuple of (updated population state, updated time)
        """
        x = x_y[..., 0]  # Prey population
        y = x_y[..., 1]  # Predator population

        # Generate random noise for Brownian motion (scaled by sqrt(dt))
        sqrt_dt = torch.sqrt(torch.tensor(dt))
        dw = torch.randn(2, *x.shape) * sqrt_dt

        # Calculate deterministic derivatives
        deriv_x = x * (alpha - beta * y)
        deriv_y = y * (delta * x - gamma)

        # Apply an exponent 1/6 to emphasise the noise at lower population levels and
        # prevent the populations from dying out.
        noise_x = (x**self.noise_exponent) * sigma * dw[0]
        noise_y = (y**self.noise_exponent) * sigma * dw[1]

        # Update populations with Euler-Maruyama method
        x = x + deriv_x * dt + noise_x
        y = y + deriv_y * dt + noise_y

        # Make sure that the populations never become negative. Mathematically, the
        # populations should remain positive. Note that if we were to `max(x, 0)`, then
        # `x` could become zero. We therefore take the absolute value.
        x = torch.abs(x)
        y = torch.abs(y)

        # Make sure that the populations do not become overflow
        # (This caused some issues in training).
        if self.max_prey_population:
            x = torch.clamp(x, max=self.max_prey_population)
        if self.max_pred_population:
            y = torch.clamp(y, max=self.max_pred_population)

        # Update time
        t = t + dt

        return torch.stack([x, y], dim=-1), t

    def _predprey_simulate(
        self,
        t0: float,
        t1: float,
        dt: float,
        t_target: torch.Tensor,
        batch_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate the stochastic Lotka-Volterra model.

        Uses adaptive step size to balance computational efficiency and accuracy.

        Args:
            t0: Start time
            t1: End time
            dt: Initial time step
            t_target: Target times to record the population
            batch_size: Number of simulations to run in parallel

        Returns:
            Tuple of (times, trajectories)
        """
        # Sample model parameters
        params = self._sample_lotka_volterra_params(batch_size=batch_size)
        scale_param = params.pop("scale")

        # Sort the target times for efficient collection during simulation
        perm = torch.argsort(t_target)
        inv_perm = self._inverse_permutation(perm)
        t_target_sorted = t_target[perm]

        # Initialize populations with random values
        x_y = torch.cat(
            (
                self._shift_scale(torch.rand(batch_size, 1), self.prey_init_range),
                self._shift_scale(torch.rand(batch_size, 1), self.pred_init_range),
            ),
            dim=-1,
        )
        t = t0
        traj: list[tuple[float, torch.Tensor]] = []

        # Define a collect function to gather trajectory points at target times
        def _collect(t_current, t_targets, remainder=False):
            result_targets = t_targets.clone()
            idx = 0
            while idx < len(result_targets) and (
                t_current >= result_targets[idx] or remainder
            ):
                traj.append((result_targets[idx].item(), x_y.clone()))
                idx += 1
            return (
                result_targets[idx:]
                if idx < len(result_targets)
                else result_targets.new_empty(0)
            )

        # Run the simulation
        t_target_remaining = _collect(t, t_target_sorted)
        while t < t1:
            x_y, t = self._predprey_step(x_y, t, dt, **params)
            t_target_remaining = _collect(t, t_target_remaining)
        # Collect any remaining points
        _ = _collect(t, t_target_remaining, remainder=True)

        # Convert lists to tensors
        t_vals, traj_values = zip(*traj) if traj else ([], [])
        times = torch.tensor(t_vals)
        trajectories = torch.stack(traj_values, dim=1)  # [batch_size, num_points, 2]

        # Apply scale to the trajectories
        trajectories = trajectories * scale_param[:, None, None]

        # Undo the sorting
        times = times[inv_perm]
        trajectories = trajectories[:, inv_perm, :]

        return times, trajectories

    def _get_from_trajectory_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples from the trajectory pool.
        Regenerates the pool if necessary.

        Returns:
            Tuple of (times, trajectory values)
        """
        if self._trajectory_pool_num_left > 0:
            # There is still some available from the trajectory pool
            assert self._trajectory_pool_x is not None and self._trajectory_pool_y is not None
            x = self._trajectory_pool_x
            y = self._trajectory_pool_y[: self.batch_size]
            self._trajectory_pool_y = self._trajectory_pool_y[self.batch_size :]
            self._trajectory_pool_num_left -= 1
            return x, y
        else:
            # Create a new trajectory pool
            x = torch.arange(self.t_simulation_range[0], self.t_simulation_range[1], self.sim_resolution)

            # Calculate how many batches to generate at once for efficiency
            multiplier = max(self.trajectory_pool_size // self.batch_size, 1)
            effective_batch_size = multiplier * self.batch_size

            # Simulate the equations
            x, y = self._predprey_simulate(
                t0=self.t_simulation_range[0],
                t1=self.t_simulation_range[1],
                # Use a budget of 5000 steps
                dt=(self.t_simulation_range[1] - self.t_simulation_range[0]) / self.max_sim_steps,
                t_target=x,
                batch_size=effective_batch_size,
            )

            # Save the trajectory pool and rerun generation to return the first slice
            self._trajectory_pool_x = x
            self._trajectory_pool_y = y
            self._trajectory_pool_num_left = multiplier
            return self._get_from_trajectory_pool()

    def generate_batch(self) -> PredPreyBatch:
        """
        Generate a batch of predator-prey data with optional
        range-based sampling for context and query points.

        Returns:
            PredPreyBatch containing:
                - x: All sampled input points [B, n, 1]
                - y: All sampled output points [B, n, D]
                - xc: Context input points [B, nc, 1]
                - yc: Context output points [B, nc, D]
                - xq: Query input points [B, nq, 1]
                - yq: Query output points [B, nq, D]
                - x_dense: Dense input points [B, T, 1]
                - y_dense: Dense output values [B, T, D]
        """
        # Sample number of context and query points
        nc, nq = self._sample_point_counts()

        # Retrieve raw trajectories from pool
        x_dense, y_dense = self._get_from_trajectory_pool()  # x_dense: [T], y_dense: [B, T, D]
        B, T, D = y_dense.shape

        # Prepare masks for eligible indices based on raw time ranges
        device = x_dense.device
        if self.xc_range is not None:
            xc_min, xc_max = self.xc_range
            mask_c = (x_dense >= xc_min) & (x_dense <= xc_max)
        else:
            mask_c = torch.ones(T, dtype=torch.bool, device=device)
        if self.xq_range is not None:
            xq_min, xq_max = self.xq_range
            mask_q = (x_dense >= xq_min) & (x_dense <= xq_max)
        else:
            mask_q = torch.ones(T, dtype=torch.bool, device=device)

        # Initialize index tensors
        idc = torch.empty(B, nc, dtype=torch.long, device=device)
        idq = torch.empty(B, nq, dtype=torch.long, device=device)

        # Sample per batch element
        for b in range(B):
            # Sample context indices
            probs_c = mask_c.float()
            if probs_c.sum() < nc:
                raise RuntimeError(
                    f"Not enough points in xc_range to sample {nc} context points"
                )
            idc_b = torch.multinomial(probs_c, nc, replacement=False)

            # Sample query indices
            probs_q = mask_q.float()
            # Exclude any context indices in overlap
            probs_q[idc_b] = 0
            if probs_q.sum() < nq:
                raise RuntimeError(
                    f"Not enough points in xq_range excluding context to sample {nq} query points"
                )
            idq_b = torch.multinomial(probs_q, nq, replacement=False)

            idc[b] = idc_b
            idq[b] = idq_b

        # Concatenate context and query indices
        idx = torch.cat([idc, idq], dim=1)  # [B, n]

        # Rescale x and y
        x_dense_rescaled = x_dense * self.time_rescale_factor
        y_dense_rescaled = y_dense * self.population_rescale_factor

        # Gather sampled points
        x_dense_expanded = x_dense_rescaled[None].expand(B, -1)  # [B, T]
        x_sampled = torch.gather(x_dense_expanded, dim=1, index=idx)  # [B, n]
        x_sampled = x_sampled[..., None]  # [B, n, 1]

        idx_exp = idx[..., None].expand(-1, -1, D)  # [B, n, D]
        y_sampled = torch.gather(y_dense_rescaled, dim=1, index=idx_exp)  # [B, n, D]

        # Split into context and query (held-out) sets
        xc = x_sampled[:, :nc]
        yc = y_sampled[:, :nc]
        xq = x_sampled[:, nc:]
        yq = y_sampled[:, nc:]

        # Prepare dense trajectories
        x_dense = x_dense_expanded[..., None]  # [B, T, 1]
        y_dense = y_dense_rescaled  # [B, T, D]

        return PredPreyBatch(
            x=x_sampled,
            y=y_sampled,
            xc=xc,
            yc=yc,
            xq=xq,
            yq=yq,
            x_dense=x_dense,
            y_dense=y_dense,
        )

    def _sample_point_counts(self) -> tuple[int, int]:
        """
        Sample the number of context and query points.

        Returns:
            Tuple of (number of context points, number of query points)
        """
        nc = torch.randint(
            low=self.min_nc,
            high=self.max_nc + 1,
            size=(),
        )
        nq = torch.randint(
            low=self.min_nq,
            high=self.max_nq + 1,
            size=(),
        )
        return int(nc.item()), int(nq.item())

    @staticmethod
    def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse permutation.

        Args:
            perm: Permutation tensor

        Returns:
            Inverse permutation tensor
        """
        inv_perm = torch.empty_like(perm)
        arange = torch.arange(perm.size(0))
        inv_perm[perm] = arange
        return inv_perm

    @staticmethod
    def _shift_scale(
        tensor: torch.Tensor,
        range_: tuple[float, float],
    ) -> torch.Tensor:
        """Shift and scale values from [0, 1] to target range."""
        return tensor * (range_[1] - range_[0]) + range_[0]
