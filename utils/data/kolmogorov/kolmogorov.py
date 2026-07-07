import abc
import math
import random

import einops
import jax
import jax.numpy as jnp
import jax.random as rng
import numpy as np
import torch

try:
    # Import submodules by name: jax_cfd is untyped, and the type checker does
    # not treat `import jax_cfd.base as cfd` as binding the submodules.
    from jax_cfd.base import (
        boundaries,
        equations,
        forcings,
        funcutils,
        grids,
        initial_conditions,
    )
except ImportError as e:
    raise ImportError(
        "jax_cfd is required for KolmogorovFlow. "
        "Please install it with: "
        "pip install jax_cfd==0.2.1"
    ) from e

# WARNING: If you encounter the error 
# "AttributeError: module 'jax.numpy' has no attribute 'DeviceArray'",
# you need to downgrade your JAX version. 
# Run: pip install jax==0.4.27 jaxlib==0.4.27

from torch import Size, Tensor


class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(
        self,
        shape: Size | tuple[int, ...] = (),
    ) -> Tensor:
        r"""x_0 ~ p(x_0)"""
        pass

    @abc.abstractmethod
    def transition(
        self,
        x: Tensor,
    ) -> Tensor:
        r"""x_i ~ p(x_i | x_{i-1})"""
        pass

    def trajectory(
        self,
        x: Tensor,
        length: int,
        last: bool = False,
    ) -> Tensor:
        r"""(x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1})"""

        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class KolmogorovFlow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,
        dt: float = 0.01,
        reynolds: float = 1e3,
    ):
        super().__init__()

        # Store parameters for coordinate generation
        self.size = size
        self.dt = dt
        self.domain = ((0, 2 * math.pi), (0, 2 * math.pi))

        self.grid = grids.Grid(
            shape=(size, size),
            domain=self.domain,
        )

        bc = boundaries.periodic_boundary_conditions(2)

        forcing = forcings.simple_turbulence_forcing(
            grid=self.grid,
            constant_magnitude=1.0,
            constant_wavenumber=4,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )

        self.dt_max = equations.stable_time_step(
            grid=self.grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )  # Maximum stable time step

        if self.dt_max > dt:
            self.inner_steps = 1
        else:
            print(
                f"[WARNING] Time step dt ({dt:.5f}) exceeds maximum stable limit ({self.dt_max:.5f}). "
                f"Adjusting to {self.dt_max:.5f} and increasing integration steps."
            )
            self.inner_steps = math.ceil(dt / self.dt_max)

        self.effective_dt = dt / self.inner_steps

        step = funcutils.repeated(
            f=equations.semi_implicit_navier_stokes(
                grid=self.grid,
                forcing=forcing,
                dt=self.effective_dt,
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=self.inner_steps,
        )

        def prior(key: jax.Array) -> jax.Array:
            u, v = initial_conditions.filtered_velocity_field(
                key,
                grid=self.grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )

            return jnp.stack((u.data, v.data))

        def transition(uv: jax.Array) -> jax.Array:
            u, v = initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=self.grid,
                bcs=(bc, bc),
            )

            u, v = step((u, v))

            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(prior, signature="(K)->(C,H,W)"))
        self._transition = jax.jit(
            jnp.vectorize(transition, signature="(C,H,W)->(C,H,W)")
        )

    def prior(
        self,
        shape: Size | tuple[int, ...] = (),
        seed: int | None = None,
    ) -> Tensor:
        if seed is None:
            seed = random.randrange(2**32)

        key = rng.PRNGKey(seed)
        keys = rng.split(key, Size(shape).numel())
        keys = keys.reshape(*shape, -1)

        x = self._prior(keys)
        x = torch.tensor(np.asarray(x))

        return x

    def transition(
        self,
        x: Tensor,
    ) -> Tensor:
        original_device = x.device
        original_dtype = x.dtype

        x_np = x.detach().cpu().numpy()
        x_np = self._transition(x_np)
        return torch.tensor(
            np.asarray(x_np),
            dtype=original_dtype,
            device=original_device,
        )

    @staticmethod
    def vorticity(
        x: Tensor,
    ) -> Tensor:
        """
        Compute vorticity from velocity field.
        Args:
            x: Input tensor of shape [T, B, C, H, W] where C=2 (u, v components)
        Returns:
            Vorticity tensor of shape [T, B, H, W]
        """
        *batch, _, h, w = x.shape

        y = x.reshape(-1, 2, h, w)
        y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode="circular")

        (du,) = torch.gradient(y[:, 0], dim=-1)
        (dv,) = torch.gradient(y[:, 1], dim=-2)

        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, h, w)

        return y

    def get_time_points(
        self,
        length: int,
        start_time: float = 0.0,
    ) -> Tensor:
        """
        Get time points for a trajectory.

        Args:
            length: Number of time steps
            start_time: Starting time value

        Returns:
            Time points tensor of shape [length]
        """
        return (
            torch.arange(length, dtype=torch.float32)
            * self.effective_dt
            * self.inner_steps
            + start_time
        )

    def get_grid(
        self,
        num_time_steps: int,
        start_time: float = 0.0,
    ) -> Tensor:
        """
        Get the grid coordinates for the Kolmogorov flow.

        Args:
            num_time_steps: Number of time steps to include.
            start_time: Starting time value for temporal coordinates.

        Returns:
            Returns coordinates as a single tensor
                - Spatiotemporal: shape [T, 3, H, W]
        """
        # Get spatial mesh
        mesh = self.grid.mesh()
        mesh = jnp.stack(mesh, axis=0)
        mesh = torch.tensor(np.asarray(mesh))

        # Include temporal coordinates using get_time_points
        time_coords = self.get_time_points(num_time_steps, start_time)

        # Create spatiotemporal grid of shape [T, 3, H, W]
        time_coords = einops.repeat(
            time_coords, "T -> T 1 H W", H=self.size, W=self.size
        )
        mesh = einops.repeat(mesh, "C H W -> T C H W", T=num_time_steps)

        return torch.cat((time_coords, mesh), dim=1)  # [T, 3, H, W]

    @staticmethod
    def coarsen(
        x: Tensor,
        spatial_r: int = 2,
        temporal_r: int = 1,
    ) -> Tensor:
        """
        Coarsen trajectories along spatial and/or temporal dimensions.

        Args:
            x: Input tensor. If temporal coarsening is applied (temporal_r > 1),
               expects shape [T, B, C, H, W] where T is time dimension.
               Otherwise expects shape [..., H, W] for spatial-only coarsening.
            spatial_r: Spatial coarsening factor (default: 2)
            temporal_r: Temporal coarsening factor (default: 1, no temporal coarsening)

        Returns:
            Coarsened tensor with reduced spatial and/or temporal resolution
        """
        x = einops.rearrange(
            x,
            "(t tr) ... (h hr) (w wr) -> t ... h w (tr hr wr)",
            tr=temporal_r,
            hr=spatial_r,
            wr=spatial_r,
        )
        x = x.mean(dim=-1)  # Average over temporal and spatial dimensions

        return x

    # @staticmethod
    # def upsample(
    #     x: Tensor,
    #     r: int = 2,
    #     mode: str = 'bilinear',
    # ) -> Tensor:
    #     *batch, h, w = x.shape

    #     x = x.reshape(-1, 1, h, w)
    #     x = torch.nn.functional.pad(
    #         x, pad=(1, 1, 1, 1), mode='circular'
    #     )
    #     x = torch.nn.functional.interpolate(
    #         x, scale_factor=(r, r), mode=mode
    #     )
    #     x = x[..., r:-r, r:-r]
    #     x = x.reshape(*batch, r * h, r * w)

    #     return x
