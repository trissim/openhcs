"""Canonical OpenHCS import surface for arraybridge memory decorators.

The underlying memory conversion behavior belongs to arraybridge. OpenHCS adds
compiler/runtime metadata preservation at this boundary so memory declarations
and callable preparation contracts compose predictably.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import arraybridge as _arraybridge


def _with_openhcs_metadata(decorator: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an arraybridge decorator with optional OpenHCS prepare metadata."""

    def openhcs_decorator(*args: Any, prepare: Any = None, **kwargs: Any) -> Any:
        if args and callable(args[0]) and len(args) == 1:
            wrapped = decorator(args[0], **kwargs)
            if prepare is not None:
                from openhcs.core.callable_contract import (
                    attach_processing_prepare,
                )

                attach_processing_prepare(wrapped, prepare)
            return wrapped

        arraybridge_decorator = decorator(*args, **kwargs)

        def decorate(target: Any) -> Any:
            wrapped = arraybridge_decorator(target)
            if prepare is not None:
                from openhcs.core.callable_contract import (
                    attach_processing_prepare,
                )

                attach_processing_prepare(wrapped, prepare)
            return wrapped

        return decorate

    openhcs_decorator.__name__ = getattr(decorator, "__name__", "openhcs_decorator")
    openhcs_decorator.__doc__ = getattr(decorator, "__doc__", None)
    openhcs_decorator.__module__ = __name__
    return openhcs_decorator


memory_types = _with_openhcs_metadata(_arraybridge.memory_types)
numpy = _with_openhcs_metadata(_arraybridge.numpy)
cupy = _with_openhcs_metadata(_arraybridge.cupy)
torch = _with_openhcs_metadata(_arraybridge.torch)
tensorflow = _with_openhcs_metadata(_arraybridge.tensorflow)
jax = _with_openhcs_metadata(_arraybridge.jax)
pyclesperanto = _with_openhcs_metadata(_arraybridge.pyclesperanto)

__all__ = [
    "memory_types",
    "numpy",
    "cupy",
    "torch",
    "tensorflow",
    "jax",
    "pyclesperanto",
]
