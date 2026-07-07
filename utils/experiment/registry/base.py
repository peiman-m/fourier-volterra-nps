"""Base registry implementation for model and batch type wrappers."""

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Generic, Type, TypeVar

import torch
import torch.nn as nn

from ...data import BaseBatch

# Generic type for wrapper functions
T = TypeVar("T", bound=Callable)


class BaseWrapperRegistry(Generic[T]):
    """Generic registry with support for class hierarchies.

    This registry maps combinations of model and batch types to wrapper functions.
    It supports both exact matches and hierarchical matching based on inheritance.
    """

    def __init__(self):
        # Registry for class-based wrappers
        self._class_registry: dict[tuple[Type[nn.Module], Type[BaseBatch]], T] = {}
        # Registry for instance-based wrappers (for specific model instances)
        self._instance_registry: dict[int, T] = {}
        # Cache for class hierarchy lookups
        self._cache = lru_cache(maxsize=128)(self._find_best_match_uncached)

    def register_class(
        self,
        model_cls: Type[nn.Module] | tuple[Type[nn.Module], ...],
        batch_cls: Type[BaseBatch] | tuple[Type[BaseBatch], ...],
        wrapper: T,
    ) -> None:
        """Register a wrapper for a model class and batch class combination."""
        model_types = model_cls if isinstance(model_cls, tuple) else (model_cls,)
        batch_types = batch_cls if isinstance(batch_cls, tuple) else (batch_cls,)

        for model_type in model_types:
            for batch_type in batch_types:
                self._class_registry[(model_type, batch_type)] = wrapper

        # Clear cache when registry changes
        self._cache.cache_clear()

    def register_instance(self, model: nn.Module, wrapper: T) -> None:
        """Register a wrapper for a specific model instance."""
        self._instance_registry[id(model)] = wrapper

    def get_wrapper(self, model: nn.Module, batch: BaseBatch) -> T | None:
        """Find the best matching wrapper for the given model and batch."""
        # Check for instance-specific wrapper first
        if id(model) in self._instance_registry:
            return self._instance_registry[id(model)]

        # Check for class-based wrapper
        model_type = type(model)
        batch_type = type(batch)

        # Check for exact match first (fast path)
        if (model_type, batch_type) in self._class_registry:
            return self._class_registry[(model_type, batch_type)]

        # Use cached search for class hierarchy matching
        return self._cache(model, batch)

    def _find_best_match_uncached(self, model: nn.Module, batch: BaseBatch) -> T | None:
        """Find the best wrapper by traversing class hierarchies (non-cached version)."""
        model_type = type(model)
        batch_type = type(batch)

        # 1. Find all registered model types that the model is an instance of
        matching_model_types = []
        for reg_model_type, _ in self._class_registry:
            if inspect.isclass(reg_model_type) and isinstance(model, reg_model_type):
                matching_model_types.append(reg_model_type)

        # 2. Find all registered batch types that the batch is an instance of
        matching_batch_types = []
        for _, reg_batch_type in self._class_registry:
            if inspect.isclass(reg_batch_type) and isinstance(batch, reg_batch_type):
                matching_batch_types.append(reg_batch_type)

        # 3. Find the best match based on specificity
        # More specific classes are those farther from the root (object)
        def class_depth(cls):
            """Calculate the inheritance depth of a class."""
            depth = 0
            for base in cls.__mro__:
                if base is not object:
                    depth += 1
            return depth

        best_match = None
        best_score = -1

        for m_type in matching_model_types:
            for b_type in matching_batch_types:
                if (m_type, b_type) in self._class_registry:
                    # Score based on combined specificity of both classes
                    score = class_depth(m_type) + class_depth(b_type)
                    if score > best_score:
                        best_score = score
                        best_match = self._class_registry[(m_type, b_type)]

        return best_match


def register_class_wrapper(
    registry: BaseWrapperRegistry[T],
    model_cls: Type[nn.Module] | tuple[Type[nn.Module], ...],
    batch_cls: Type[BaseBatch] | tuple[Type[BaseBatch], ...],
) -> Callable[[T], T]:
    """Decorator to register a wrapper for specific model and batch types.

    Args:
        registry: The registry to register with.
        model_cls: Model class or tuple of model classes.
        batch_cls: Batch class or tuple of batch classes.

    Returns:
        Decorator function.
    """

    def decorator(wrapper_fn: T) -> T:
        registry.register_class(model_cls, batch_cls, wrapper_fn)
        return wrapper_fn

    return decorator


def register_instance_wrapper(
    registry: BaseWrapperRegistry[T], model: nn.Module
) -> Callable[[T], T]:
    """Decorator to register a wrapper for a specific model instance.

    Args:
        registry: The registry to register with.
        model: Model instance.

    Returns:
        Decorator function.
    """

    def decorator(wrapper_fn: T) -> T:
        registry.register_instance(model, wrapper_fn)
        return wrapper_fn

    return decorator
