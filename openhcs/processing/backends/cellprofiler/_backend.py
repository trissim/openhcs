"""Shared backend-selection helpers for CellProfiler-compatible processing."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, TypeVar

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import CallableContract


class CellProfilerBackendProvider(str, Enum):
    """Typed CellProfiler-compatible backend provider identifiers."""

    NATIVE = "native"
    NUMBA = "numba"
    CENTROSOME = "centrosome"
    LEGACY_FAST = "legacy_fast"
    EXACT = "exact"
    NUMBA_EXACT = "numba_exact"
    NATIVE_EXACT = "native_exact"
    CUCIM = "cucim"
    PYCLESPERANTO = "pyclesperanto"


DEFAULT_CELLPROFILER_BACKEND_PROVIDER = CellProfilerBackendProvider.NATIVE
_BACKEND_KEY_SEPARATOR = ":"

BackendStrategyT = TypeVar(
    "BackendStrategyT",
    bound="CellProfilerBackendStrategyMixin",
)
BackendProviderInput = CellProfilerBackendProvider


def normalize_cellprofiler_memory_type(
    memory_type: MemoryType | str = MemoryType.NUMPY,
) -> MemoryType:
    """Resolve a memory type value using OpenHCS' canonical enum."""
    return memory_type if isinstance(memory_type, MemoryType) else MemoryType(str(memory_type))


def normalize_cellprofiler_backend_provider(
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_PROVIDER,
) -> CellProfilerBackendProvider:
    """Resolve a backend provider using the closed typed provider enum."""
    if not isinstance(backend_provider, CellProfilerBackendProvider):
        raise TypeError(
            "CellProfiler backend provider must be a "
            "CellProfilerBackendProvider enum value"
        )
    return backend_provider


def cellprofiler_backend_key(
    memory_type: MemoryType | str = MemoryType.NUMPY,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_PROVIDER,
) -> str:
    """Return the registry key for one memory/provider backend implementation."""
    provider = normalize_cellprofiler_backend_provider(backend_provider)
    return (
        normalize_cellprofiler_memory_type(memory_type).value
        + _BACKEND_KEY_SEPARATOR
        + provider.value
    )


class CellProfilerBackendStrategyMixin:
    """Mixin for backend strategies keyed by OpenHCS memory type and provider.

    Concrete strategy families keep their own AutoRegisterMeta registry; this
    mixin only standardizes lookup semantics so adding providers does not copy
    boilerplate across morphology, thresholding, watershed, and future modules.
    """

    backend_key: ClassVar[str | None] = None
    memory_type: ClassVar[MemoryType | None] = None
    backend_provider: ClassVar[CellProfilerBackendProvider] = (
        DEFAULT_CELLPROFILER_BACKEND_PROVIDER
    )
    is_default_backend: ClassVar[bool] = False

    @classmethod
    def for_memory_type(
        cls: type[BackendStrategyT],
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput | None = None,
    ) -> BackendStrategyT:
        """Instantiate the exact backend for ``memory_type`` and provider.

        ``backend_provider=None`` resolves the single default provider for the
        memory type. Explicit providers never fall back to another backend.
        """
        return cls._resolve_backend_class(memory_type, backend_provider)()

    @classmethod
    def for_callable(
        cls: type[BackendStrategyT],
        func: object,
        *,
        backend_provider: BackendProviderInput | None = None,
    ) -> BackendStrategyT:
        """Instantiate a backend using a function's OpenHCS memory contract."""
        contract = CallableContract.from_callable(func)
        memory_type = (
            contract.input_memory_type
            or contract.output_memory_type
            or MemoryType.NUMPY.value
        )
        return cls.for_memory_type(memory_type, backend_provider=backend_provider)

    @classmethod
    def available_backend_providers(
        cls,
        memory_type: MemoryType | str | None = None,
    ) -> tuple[CellProfilerBackendProvider, ...]:
        """Return registered providers, optionally filtered by memory type."""
        resolved = (
            None
            if memory_type is None
            else normalize_cellprofiler_memory_type(memory_type)
        )
        providers: list[CellProfilerBackendProvider] = []
        for strategy_cls in getattr(cls, "__registry__", {}).values():
            if resolved is not None and strategy_cls.memory_type is not resolved:
                continue
            providers.append(
                normalize_cellprofiler_backend_provider(strategy_cls.backend_provider)
            )
        return tuple(sorted(set(providers), key=lambda provider: provider.value))

    @classmethod
    def _resolve_backend_class(
        cls: type[BackendStrategyT],
        memory_type: MemoryType | str,
        backend_provider: BackendProviderInput | None,
    ) -> type[BackendStrategyT]:
        resolved = normalize_cellprofiler_memory_type(memory_type)
        registry: dict[str, type[BackendStrategyT]] = getattr(cls, "__registry__", {})
        if backend_provider is not None:
            provider = normalize_cellprofiler_backend_provider(backend_provider)
            key = cellprofiler_backend_key(resolved, provider)
            try:
                return registry[key]
            except KeyError as exc:
                raise NotImplementedError(
                    f"No CellProfiler {cls.__name__} backend is registered for "
                    f"memory type {resolved.value!r} and provider "
                    f"{provider.value!r}. Registered providers for this memory "
                    f"type: {cls.available_backend_providers(resolved)!r}."
                ) from exc

        matches = [
            strategy_cls
            for strategy_cls in registry.values()
            if strategy_cls.memory_type is resolved
            and bool(strategy_cls.is_default_backend)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise NotImplementedError(
                f"No default CellProfiler {cls.__name__} backend is registered "
                f"for memory type {resolved.value!r}. Registered providers for "
                f"this memory type: {cls.available_backend_providers(resolved)!r}."
            )
        raise RuntimeError(
            f"Multiple default CellProfiler {cls.__name__} backends are "
            f"registered for memory type {resolved.value!r}: "
            f"{tuple(strategy.__name__ for strategy in matches)!r}."
        )


__all__ = [
    "DEFAULT_CELLPROFILER_BACKEND_PROVIDER",
    "BackendProviderInput",
    "CellProfilerBackendProvider",
    "CellProfilerBackendStrategyMixin",
    "cellprofiler_backend_key",
    "normalize_cellprofiler_backend_provider",
    "normalize_cellprofiler_memory_type",
]
