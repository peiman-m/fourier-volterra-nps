"""OmegaConf resolvers for the Hydra-based config system.

The ``${eval:...}`` resolver does not bind Python's built-in ``eval`` —
it routes through :func:`_safe_arithmetic_eval`, an AST-walker that
accepts only numeric literals and the four arithmetic operators. Every
yaml call site is arithmetic-only (``2 * ${params.embed_dim}``,
``1.0 / 64.0``, ``${params.y_dim} + 1``, …) so the surface area is the
same while the YAML-side code-injection vector is closed.
"""

from __future__ import annotations

import ast
import itertools
import operator
from collections.abc import Iterable
from typing import Any

from omegaconf import OmegaConf


# Whitelisted AST binary / unary operators. Any operator not in this map
# causes :func:`_safe_arithmetic_eval` to reject the expression.
_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_arithmetic_eval(expression: str) -> Any:
    """Evaluate a numeric-only arithmetic expression.

    Supports integer / float literals, parenthesized sub-expressions, and
    the four arithmetic operators (``+``, ``-``, ``*``, ``/``) plus unary
    sign. Any other AST node — function calls, attribute access, names,
    comparisons, subscripts, comprehensions — raises ``ValueError``.

    OmegaConf resolves interpolations inside-out, so ``${params.y_dim}``
    is already substituted with its concrete value by the time this
    function sees the string. That is why a pure-literal evaluator is
    sufficient for every ``${eval:...}`` call site in the conf tree.
    """
    if not isinstance(expression, str):
        raise TypeError(
            f"eval resolver expects a string expression, got {type(expression)}"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise ValueError(
            f"eval resolver could not parse expression {expression!r}: {err}"
        ) from err

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            # Accept int, float, and bool. Bools are deliberately in the
            # whitelist because era5 configs interpolate boolean flags
            # (``use_time``, ``use_surface_elevation``) into arithmetic
            # expressions such as ``2 + True + False`` — relying on
            # Python's ``True == 1`` / ``False == 0`` coercion. The
            # legacy ``eval`` handled this; rejecting bools here would
            # break those call sites.
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError(
                "eval resolver only accepts int/float/bool literals; "
                f"got {type(node.value).__name__} ({node.value!r})"
            )
        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(
                    f"eval resolver rejects binary operator {type(node.op).__name__}"
                )
            return op_fn(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(
                    f"eval resolver rejects unary operator {type(node.op).__name__}"
                )
            return op_fn(_walk(node.operand))
        raise ValueError(
            "eval resolver rejects AST node "
            f"{type(node).__name__}; only arithmetic on numeric literals is allowed"
        )

    return _walk(tree)


def _product_resolver(
    target: str, param_dict: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    if not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dictionary.")

    names = list(param_dict.keys())
    value_lists: list[list[Any]] = []
    for name in names:
        value = param_dict[name]
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise TypeError(
                "All values in param_dict must be iterable (excluding strings). "
                f"Got: {type(value)}"
            )
        value_lists.append(list(value) if not isinstance(value, list) else value)

    combinations = itertools.product(*value_lists)
    return [{"_target_": target, **dict(zip(names, combo))} for combo in combinations]


def _range_resolver(*args: Any) -> list[int]:
    if len(args) == 1:
        start, stop, step = 0, int(args[0]), 1
    elif len(args) == 2:
        start, stop, step = int(args[0]), int(args[1]), 1
    elif len(args) == 3:
        start, stop, step = int(args[0]), int(args[1]), int(args[2])
    else:
        raise ValueError(
            "range resolver expects 1, 2, or 3 arguments (stop, or start, stop, [step])."
        )
    return list(range(start, stop, step))


def register_resolvers() -> None:
    """Idempotently register ``eval``, ``product``, ``range`` with OmegaConf.

    Must be called *before* Hydra resolves any config that references these
    resolvers — in practice, at import time of the Hydra entry point, above
    the ``@hydra.main`` decorator.
    """
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", _safe_arithmetic_eval)
    if not OmegaConf.has_resolver("product"):
        OmegaConf.register_new_resolver("product", _product_resolver)
    if not OmegaConf.has_resolver("range"):
        OmegaConf.register_new_resolver("range", _range_resolver)


register_resolvers()
