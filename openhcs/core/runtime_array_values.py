"""Nominal runtime array payload protocol and NumPy interoperation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Self

import numpy as np


def runtime_array_ufunc_result(
    ufunc: Any,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Invoke a NumPy ``__array_ufunc__`` protocol method without dynamic lookup."""
    match method:
        case "__call__":
            return ufunc(*inputs, **kwargs)
        case "reduce":
            return ufunc.reduce(*inputs, **kwargs)
        case "accumulate":
            return ufunc.accumulate(*inputs, **kwargs)
        case "reduceat":
            return ufunc.reduceat(*inputs, **kwargs)
        case "outer":
            return ufunc.outer(*inputs, **kwargs)
        case "at":
            return ufunc.at(*inputs, **kwargs)
        case _:
            return NotImplemented


class RuntimeArrayPayload(ABC):
    """Nominal ABC for array payload types accepted by runtime artifacts."""

    __array_priority__ = 1000

    @property
    @abstractmethod
    def shape(self) -> Any: ...

    @abstractmethod
    def array_payload_data(self) -> Any: ...

    @abstractmethod
    def with_data(self, data: Any) -> Self: ...

    def array_ufunc_result(self, result: Any) -> Any:
        if isinstance(result, tuple):
            return tuple(self.array_ufunc_result(item) for item in result)
        if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.bool_):
            return result
        if isinstance(result, np.ndarray):
            return self.with_data(result)
        return result

    def compare_array_payload(self, other: Any, ufunc: Any) -> Any:
        return ufunc(np.asarray(self), runtime_array_operand(other))

    def __lt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less)

    def __le__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less_equal)

    def __gt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater)

    def __ge__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater_equal)

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        converted_inputs = tuple(runtime_array_operand(value) for value in inputs)
        if "out" in kwargs:
            kwargs = {
                **kwargs,
                "out": tuple(runtime_array_operand(value) for value in kwargs["out"]),
            }
        result = runtime_array_ufunc_result(ufunc, method, converted_inputs, kwargs)
        if result is NotImplemented:
            return NotImplemented
        return self.array_ufunc_result(result)


class DataBackedRuntimeArrayPayload(RuntimeArrayPayload):
    """Runtime array payload whose concrete array is stored in ``data``."""

    data: Any

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.data, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)


RuntimeArrayData = RuntimeArrayPayload | np.ndarray


def runtime_array_operand(value: Any) -> Any:
    """Return the ndarray operand for nominal runtime array payloads."""
    if isinstance(value, RuntimeArrayPayload):
        return value.array_payload_data()
    return value


def is_array_payload(data: Any) -> bool:
    """Return whether OpenHCS or ArrayBridge owns this array payload type."""
    if isinstance(data, RuntimeArrayPayload):
        return True
    from openhcs.core.memory import detect_memory_type

    try:
        detect_memory_type(data)
    except ValueError:
        return False
    return True
