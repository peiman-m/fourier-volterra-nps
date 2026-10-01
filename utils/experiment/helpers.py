from enum import Enum, auto
from typing import Any, Literal, cast, overload

import torch


class ReductionType(Enum):
    """Enumeration of supported reduction methods.

    NONE: No reduction is applied
    SUM: Sum the tensor along specified dimensions
    MEAN: Average the tensor along specified dimensions
    LOGMEANEXP: Log-mean-exp reduction
        (numerically stable averaging in log-space)
    """

    NONE = auto()
    SUM = auto()
    MEAN = auto()
    LOGMEANEXP = auto()


class TensorProcessor:
    """Utility class for tensor operations used in
    loss calculations and tensor manipulations."""

    @staticmethod
    def _normalize_dim(dim: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Normalize dimension parameter to tuple format.

        Args:
            dim: Dimension(s) to process.
                Can be a single integer or tuple/list of integers.

        Returns:
            tuple of dimensions
        """
        if isinstance(dim, int):
            return (dim,)
        elif isinstance(dim, list):
            return tuple(dim)
        return dim

    @staticmethod
    def _normalize_repeats(
        repeats: int | tuple[int, ...] | list[int],
    ) -> tuple[int, ...]:
        """Normalize repeats parameter to tuple format.

        Args:
            repeats: Repeat count(s).
                Can be a single integer or tuple/list of integers.

        Returns:
            tuple of repeat counts
        """
        if isinstance(repeats, int):
            return (repeats,)
        elif isinstance(repeats, list):
            return tuple(repeats)
        return repeats

    @overload
    @staticmethod
    def unsqueeze(
        tensor: torch.Tensor,
        dim: int | tuple[int, ...] | list[int],
        return_adjusted_dims: Literal[False] = ...,
    ) -> torch.Tensor: ...

    @overload
    @staticmethod
    def unsqueeze(
        tensor: torch.Tensor,
        dim: int | tuple[int, ...] | list[int],
        return_adjusted_dims: Literal[True],
    ) -> tuple[torch.Tensor, tuple[int, ...]]: ...

    @staticmethod
    def unsqueeze(
        tensor: torch.Tensor,
        dim: int | tuple[int, ...] | list[int],
        return_adjusted_dims: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, ...]]:
        """Unsqueeze a tensor along one or multiple dimensions.

        Args:
            tensor: Input tensor to unsqueeze
            dim: Dimension(s) to insert.
                Can be a single integer or tuple/list of integers.
                Negative indices are supported.
            return_adjusted_dims: Whether to return the dim of the
                newly inserted dims in the new tensor

        Returns:
            Tensor with added dimension(s),
                optionally with the adjusted dimensions

        Example:
            >>> x = torch.randn(3, 4)
            >>> TensorProcessor.unsqueeze(x, (0, 3))  # Adds dimensions at positions 0 and 3
        """
        dims = sorted(TensorProcessor._normalize_dim(dim))

        # Keep track of how many dimensions we've added
        result = tensor
        offset = 0
        adjusted_dims = []
        for d in dims:
            # Adjust the dimension index by the current offset
            adjusted_d = d + offset
            result = result.unsqueeze(adjusted_d)
            # Increment the offset for subsequent dimensions
            offset += 1
            adjusted_dims.append(adjusted_d)

        if return_adjusted_dims:
            return result, tuple(adjusted_dims)
        return result

    @staticmethod
    def repeat(
        tensor: torch.Tensor,
        repeats: int | tuple[int, ...] | list[int],
        dim: int | tuple[int, ...] | list[int],
    ) -> torch.Tensor:
        """Repeat a tensor along specified dimensions to specified sizes.

        Args:
            tensor: Input tensor to repeat
            repeats: The number of times to repeat.
                Can be a single integer or tuple/list of integers.
            dim: Dimension(s) to repeat.
                Can be a single integer or tuple/list of integers.

        Returns:
            Repeated tensor

        Raises:
            ValueError: If the number of dimensions to repeat
                doesn't match the number of provided reps or
                if dimension index is out of bounds

        Example:
            >>> x = torch.randn(3, 1, 5)
            >>> TensorProcessor.repeat(x, 4, 1)  # Repeats the first dimension 4 times
        """
        dims = TensorProcessor._normalize_dim(dim)
        repeats_tuple = TensorProcessor._normalize_repeats(repeats)

        # If a single repeat value was provided for multiple dimensions,
        # expand it to match the number of dimensions
        if len(repeats_tuple) == 1 and len(dims) > 1:
            repeats_tuple = repeats_tuple * len(dims)

        if len(repeats_tuple) != len(dims):
            raise ValueError(
                f"Number of expansion sizes ({len(repeats_tuple)}) must match "
                f"number of dimensions ({len(dims)})"
            )

        # Check dimension bounds
        tensor_ndim = tensor.ndim
        for d in dims:
            if d < -tensor_ndim or d >= tensor_ndim:
                raise ValueError(
                    f"Dimension {d} out of range for tensor with "
                    f"{tensor_ndim} dimensions"
                )

        # Create expansion size specification
        sizes = list(tensor.shape)
        for d, s in zip(dims, repeats_tuple):
            # Handle negative indices by converting to positive
            if d < 0:
                d = tensor_ndim + d
            sizes[d] = s

        return tensor.expand(*sizes)

    @classmethod
    def insert_and_expand(
        cls,
        tensor: torch.Tensor,
        dim: int | tuple[int, ...] | list[int],
        repeats: int | tuple[int, ...] | list[int],
    ) -> torch.Tensor:
        """Insert new dimensions and then repeat the tensor along those dimensions.

        This is a convenience method that combines unsqueeze and repeat.

        Args:
            tensor: Input tensor
            dim: Dimension(s) to insert
            repeats: Number of times to repeat in each inserted dimension

        Returns:
            Tensor with new dimensions inserted and expanded
        """
        result, adjusted_dims = cls.unsqueeze(tensor, dim, return_adjusted_dims=True)
        return cls.repeat(result, repeats, adjusted_dims)

    @staticmethod
    def logmeanexp(
        tensor: torch.Tensor,
        dim: int | tuple[int, ...] | list[int] | None = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Compute log(mean(exp(tensor))) in a numerically stable way.

        This is useful for computing means in log-space without numerical underflow/overflow.

        Args:
            tensor: Input tensor
            dim: Dimension(s) to reduce. If None, reduces over all dimensions.
            keepdim: Whether to keep the reduced dimensions

        Returns:
            Tensor after applying logmeanexp reduction

        Example:
            >>> x = torch.randn(5, 10)
            >>> TensorProcessor.logmeanexp(x, dim=1)  # logmeanexp over dimension 1
        """
        # Get the number of elements in the specified dimension(s)
        if dim is None:
            n_elements = torch.numel(tensor)
        else:
            dim_tuple = TensorProcessor._normalize_dim(dim)
            n_elements = 1
            for d in dim_tuple:
                d_adjusted = d if d >= 0 else tensor.ndim + d  # Handle negative indices
                n_elements *= tensor.shape[d_adjusted]

        # logmeanexp = logsumexp - log(n)
        return torch.logsumexp(tensor, dim=cast(Any, dim), keepdim=keepdim) - torch.log(
            torch.tensor(n_elements, dtype=tensor.dtype, device=tensor.device)
        )

    @classmethod
    def reduce(
        cls,
        tensor: torch.Tensor,
        reduction: str | ReductionType | None = ReductionType.MEAN,
        dim: int | tuple[int, ...] | list[int] | None = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Reduce a tensor along specified dimensions using the specified reduction method.

        Args:
            tensor: Input tensor to reduce
            reduction: Reduction method to use.
                Can be a ReductionType enum value, a string representation,
                or None (no reduction)
            dim: Dimension(s) to reduce along
            keepdim: Whether to keep the reduced dimensions

        Returns:
            Reduced tensor

        Example:
            >>> x = torch.randn(5, 10)
            >>> TensorProcessor.reduce(x, "mean", dim=1)  # Mean reduction over dimension 1
        """
        # Handle None case early
        if reduction is None:
            return tensor

        # Convert string reductions to enum if needed
        if isinstance(reduction, str):
            reduction = cls._parse_reduction_type(reduction)

        # Standardize dim to tuple for consistency. dim may be None here
        # (reduce-all), which _normalize_dim passes through to torch reductions.
        dim_tuple = cls._normalize_dim(cast(Any, dim))

        # Apply the specified reduction
        if reduction == ReductionType.LOGMEANEXP:
            return cls.logmeanexp(tensor, dim=dim_tuple, keepdim=keepdim)
        elif reduction == ReductionType.SUM:
            return torch.sum(tensor, dim=dim_tuple, keepdim=keepdim)
        elif reduction == ReductionType.MEAN:
            return torch.mean(tensor, dim=dim_tuple, keepdim=keepdim)
        elif reduction == ReductionType.NONE:
            return tensor
        else:
            raise ValueError(f"Unknown reduction type: {reduction}")

    @staticmethod
    def _parse_reduction_type(reduction_str: str) -> ReductionType:
        """Convert a string representation to a ReductionType enum value.

        Args:
            reduction_str: String representation of reduction type

        Returns:
            Corresponding ReductionType enum value

        Raises:
            ValueError: If the string doesn't match any known reduction type
        """
        mapping = {
            "none": ReductionType.NONE,
            "sum": ReductionType.SUM,
            "mean": ReductionType.MEAN,
            "logmeanexp": ReductionType.LOGMEANEXP,
        }

        lower_reduction_str = reduction_str.lower()
        if lower_reduction_str not in mapping:
            valid_options = ", ".join(f"'{k}'" for k in mapping.keys())
            raise ValueError(
                f"Unknown reduction type: '{reduction_str}'. "
                f"Valid options are: {valid_options}"
            )

        return mapping[lower_reduction_str]
