import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn


class BaseEmbedding(nn.Module, ABC):
    """
    Base class for an embedding.

    Args:
        active_dims (int | list[int] | tuple[int] | set[int]): The indices of input
            dimensions to be embedded.
    """

    def __init__(self, active_dims: int | Iterable[int]):
        super().__init__()

        # Normalize `active_dims` to a tuple for consistent handling. Accept any
        # non-string iterable of ints (list/tuple/set/range) as well as an
        # OmegaConf ``ListConfig`` passed straight from a Hydra config.
        if isinstance(active_dims, int):
            self.active_dims = (active_dims,)
        elif isinstance(active_dims, Iterable) and not isinstance(
            active_dims, (str, bytes)
        ):
            dims = tuple(active_dims)
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("All `active_dims` entries must be integers.")
            self.active_dims = tuple(set(dims))
        else:
            raise ValueError(
                "`active_dims` must be an integer or "
                "a collection of integers (e.g., list, tuple, set)."
            )

        self.in_dim = None  # Placeholder for input dimension
        self.out_dim = None  # Placeholder for output dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, ..., D).

        Returns:
            torch.Tensor: Embedded tensor.
        """
        if x.ndim < 2:
            raise ValueError(
                "Input tensor must have at least 2 dimensions "
                "(batch size and feature dimensions)."
            )

        D = x.shape[-1]

        # Validate active_dims range
        if max(self.active_dims) >= D or min(self.active_dims) < -D:
            raise ValueError(
                f"`active_dims` indices must be in range [{-D}, {D - 1}], "
                f"got {self.active_dims}."
            )

        # Adjust negative indices to positive equivalents and remove duplicates
        adjusted_active_dims = tuple(
            set(dim if dim >= 0 else D + dim for dim in self.active_dims)
        )

        # Extract active dimensions for embedding
        # Shape: (B, ..., len(active_dims))
        x_active = x[..., adjusted_active_dims]

        # Perform embedding on active dimensions
        # Shape: (B, ..., out_dim)
        x_embedded = self._embed(x_active)

        # Concatenate non-active dimensions with embedded dimensions
        non_active_dims = [i for i in range(D) if i not in adjusted_active_dims]
        x_non_active = x[..., non_active_dims]  # Shape: (B, ..., len(non_active_dims))

        return torch.cat((x_non_active, x_embedded), dim=-1)

    @abstractmethod
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to define the embedding operation.

        Args:
            x (torch.Tensor): Tensor with active dimensions of
                shape (B, ..., len(active_dims)).

        Returns:
            torch.Tensor: Embedded tensor.
        """
        pass


class FourierFeaturesEmbedding(BaseEmbedding):
    """
    Fourier feature embedding: maps the active input dimensions into a
    higher-dimensional space using sin and cos of a fixed (non-trained)
    frequency bank.

    The frequency bank ``B`` of shape ``(num_active_dims, num_frequencies)`` is
    produced by ``init_frequencies_fn(num_active_dims, num_frequencies)`` and
    registered as a buffer, so it is frozen (no gradients) and saved with the
    model. Subclasses choose the frequency-generation strategy:

      - ``RandomFourierFeaturesEmbedding``: random frequencies.
      - ``LogSpacedFourierFeaturesEmbedding``: deterministic, log-spaced
        wavelengths (positional-encoding style).

    Args:
        in_dim (int): Input dimensionality of the data (raw width the embedding
            receives).
        num_frequencies (int): Number of frequencies; the active dims expand to
            ``2 * num_frequencies`` (sin and cos) features.
        init_frequencies_fn (Callable): ``fn(num_active_dims, num_frequencies)``
            returning the frequency bank tensor.
        active_dims (int | list[int] | tuple[int, ...] | set[int] | None): Indices
            of input dimensions to embed. If None, all dimensions are used.
    """

    # Registered as a buffer in __init__; declared here so attribute access is
    # typed as a Tensor rather than nn.Module.__getattr__'s Tensor | Module.
    frequencies: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        init_frequencies_fn: Callable[[int, int], torch.Tensor],
        active_dims: int | list[int] | tuple[int, ...] | set[int] | None = None,
    ):
        # If active_dims is None, use all dimensions
        if active_dims is None:
            active_dims = tuple(range(in_dim))

        super().__init__(active_dims=active_dims)

        # Validate inputs
        if num_frequencies <= 0:
            raise ValueError("`num_frequencies` must be a positive integer.")
        if in_dim <= 0:
            raise ValueError("`in_dim` must be a positive integer.")
        if max(self.active_dims) >= in_dim or min(self.active_dims) < -in_dim:
            raise ValueError(
                "`active_dims` indices must be in range " f"[{-in_dim}, {in_dim - 1}]."
            )

        # Adjust negative indices to positive equivalents and remove duplicates
        self.active_dims = tuple(
            sorted(set(dim if dim >= 0 else in_dim + dim for dim in self.active_dims))
        )

        # Build the (non-trained) frequency bank via the chosen strategy
        weights = init_frequencies_fn(len(self.active_dims), num_frequencies)

        # Register frequencies as a buffer to avoid training updates
        self.register_buffer("frequencies", weights)

        # Set input and output dimensions
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies

        # Calculate out_dim: 2 modes (sin and cos) for each frequency,
        # plus non-transformed dimensions
        self.out_dim = 2 * num_frequencies + (in_dim - len(self.active_dims))

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds the active dimensions of the input tensor using Fourier features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, ..., len(active_dims)).

        Returns:
            torch.Tensor: Tensor of shape (B, ..., 2 * num_frequencies) with
                Fourier features.
        """
        # Ensure the input tensor matches the expected dimensions
        D = x.shape[-1]
        if D != len(self.active_dims):
            raise ValueError(
                f"Expected input tensor with {len(self.active_dims)} "
                f"dimensions, got {D}."
            )

        # Compute sin and cos projections
        # Shape: (B, ..., num_frequencies)
        prod = 2 * torch.pi * x @ self.frequencies
        encoding = torch.cat(
            (torch.sin(prod), torch.cos(prod)), dim=-1
        )  # Shape: (B, ..., 2 * num_frequencies)

        return encoding


class RandomFourierFeaturesEmbedding(FourierFeaturesEmbedding):
    """
    Random Fourier Feature embedding: the frequency bank is drawn from a uniform
    distribution ``U[0, 1)`` and frozen. This is the standard random-features
    variant (Rahimi & Recht; Tancik et al.).

    Args:
        in_dim (int): Input dimensionality of the data.
        num_frequencies (int): Number of frequencies for the Fourier features.
        active_dims (int | list[int] | tuple[int, ...] | set[int] | None): Indices of
            input dimensions to be embedded. If None, all dimensions are used.
        init_frequencies_fn (Callable | None): Optional override for the frequency
            initializer; accepts ``(num_active_dims, num_frequencies)``. Defaults
            to ``torch.rand`` (uniform ``U[0, 1)``).
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        active_dims: int | list[int] | tuple[int, ...] | set[int] | None = None,
        init_frequencies_fn: Callable[[int, int], torch.Tensor] | None = None,
    ):
        if init_frequencies_fn is None:

            def _random_freqs(num_active_dims: int, n_freq: int) -> torch.Tensor:
                return torch.rand(num_active_dims, n_freq)

            init_frequencies_fn = _random_freqs

        super().__init__(
            in_dim=in_dim,
            num_frequencies=num_frequencies,
            init_frequencies_fn=init_frequencies_fn,
            active_dims=active_dims,
        )


class LogSpacedFourierFeaturesEmbedding(FourierFeaturesEmbedding):
    """
    Deterministic Fourier embedding with log-spaced wavelengths
    (positional-encoding / "temporal Fourier embedding" style).

    The frequency bank is fixed (not random): ``f_i = 1 / lambda_i`` where the
    wavelengths ``lambda_i`` are log-spaced between ``lambda_min`` and
    ``lambda_max``. The same wavelength set is shared across active dims, so this
    is intended for a single active dimension (e.g. time), matching

        Emb(t) = [sin(2*pi*t / lambda_i), cos(2*pi*t / lambda_i)].

    Note: because the underlying ``_embed`` computes ``x @ frequencies``, with
    more than one active dim the coordinates are summed before the sin/cos, which
    is generally only meaningful for a single (e.g. temporal) active dim.

    Args:
        in_dim (int): Input dimensionality of the data.
        num_frequencies (int): Number of wavelengths (``L / 2``); expands the
            active dim to ``2 * num_frequencies`` (sin and cos) features.
        active_dims (int | list[int] | tuple[int, ...] | set[int] | None): Indices
            of input dimensions to embed. If None, all dimensions are used.
        lambda_min (float): Smallest (shortest) wavelength. Must be > 0.
        lambda_max (float): Largest (longest) wavelength. Must be >= lambda_min.
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        active_dims: int | list[int] | tuple[int, ...] | set[int] | None = None,
        lambda_min: float = 1.0,
        lambda_max: float = 8760.0,
    ):
        if lambda_min <= 0:
            raise ValueError("`lambda_min` must be a positive number.")
        if lambda_max < lambda_min:
            raise ValueError("`lambda_max` must be >= `lambda_min`.")

        def init_frequencies_fn(num_active_dims: int, n_freq: int) -> torch.Tensor:
            # Log-spaced wavelengths -> frequencies f_i = 1 / lambda_i, shared
            # across active dims. Shape: (num_active_dims, n_freq).
            wavelengths = torch.logspace(
                math.log10(lambda_min), math.log10(lambda_max), n_freq
            )
            freqs = 1.0 / wavelengths
            return freqs.unsqueeze(0).expand(num_active_dims, n_freq).contiguous()

        super().__init__(
            in_dim=in_dim,
            num_frequencies=num_frequencies,
            init_frequencies_fn=init_frequencies_fn,
            active_dims=active_dims,
        )

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
