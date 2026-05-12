"""Shared backend-selection helpers for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import ClassVar, TypeAlias, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import (
    CallableContract,
    CompilerPreparedAutoRegisterFamily,
)


class CellProfilerBackendProvider(str, Enum):
    """Typed CellProfiler-compatible backend provider identifiers."""

    NATIVE = "native"
    NUMBA = "numba"
    CENTROSOME = "centrosome"
    OPENCV = "opencv"
    LEGACY_FAST = "legacy_fast"
    CUCIM = "cucim"
    PYCLESPERANTO = "pyclesperanto"

    @property
    def requires_compiler_prewarm(self) -> bool:
        """Return whether this provider compiles runtime-specialized kernels."""
        return self is CellProfilerBackendProvider.NUMBA


DEFAULT_CELLPROFILER_BACKEND_PROVIDER = CellProfilerBackendProvider.NATIVE
_BACKEND_KEY_SEPARATOR = ":"

BackendStrategyT = TypeVar(
    "BackendStrategyT",
    bound="CellProfilerBackendStrategyMixin",
)


@dataclass(frozen=True, slots=True)
class CellProfilerBackendRegistrySnapshot:
    """Immutable registry view used as the backend-selection cache key."""

    strategy_family: type["CellProfilerBackendStrategyMixin"]
    memory_type: MemoryType
    registry_keys: tuple[str, ...]

    @classmethod
    def for_family(
        cls,
        strategy_family: type["CellProfilerBackendStrategyMixin"],
        memory_type: MemoryType,
    ) -> "CellProfilerBackendRegistrySnapshot":
        registry = strategy_family.__registry__
        return cls(strategy_family, memory_type, tuple(sorted(registry)))

    @property
    def registry(self) -> dict[str, type[BackendStrategyT]]:
        return cast(dict[str, type[BackendStrategyT]], self.strategy_family.__registry__)

    def available_backend_providers(self) -> tuple[CellProfilerBackendProvider, ...]:
        providers = (
            normalize_cellprofiler_backend_provider(strategy_cls.backend_provider)
            for strategy_cls in self.registry.values()
            if strategy_cls.memory_type is self.memory_type
        )
        return tuple(sorted(set(providers), key=lambda provider: provider.value))


class CellProfilerBackendProviderSelection(ABC, metaclass=AutoRegisterMeta):
    """Nominal provider-selection policy for CellProfiler backend families."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    @abstractmethod
    def backend_class(
        self,
        snapshot: CellProfilerBackendRegistrySnapshot,
    ) -> type[BackendStrategyT]:
        """Return the backend implementation selected by this policy."""

    @abstractmethod
    def provider_or(
        self,
        default_provider: CellProfilerBackendProvider,
    ) -> CellProfilerBackendProvider:
        """Return the explicit provider or a caller-owned contextual default."""


@dataclass(frozen=True, slots=True)
class DefaultCellProfilerBackendProviderSelection(CellProfilerBackendProviderSelection):
    """Select the single declared default backend for the requested memory type."""

    registry_key = "default"

    def backend_class(
        self,
        snapshot: CellProfilerBackendRegistrySnapshot,
    ) -> type[BackendStrategyT]:
        matches = [
            strategy_cls
            for strategy_cls in snapshot.registry.values()
            if strategy_cls.memory_type is snapshot.memory_type
            and bool(strategy_cls.is_default_backend)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise NotImplementedError(
                f"No default CellProfiler {snapshot.strategy_family.__name__} backend "
                f"is registered for memory type {snapshot.memory_type.value!r}. "
                f"Registered providers for this memory type: "
                f"{snapshot.available_backend_providers()!r}."
            )
        raise RuntimeError(
            f"Multiple default CellProfiler {snapshot.strategy_family.__name__} "
            f"backends are registered for memory type {snapshot.memory_type.value!r}: "
            f"{tuple(strategy.__name__ for strategy in matches)!r}."
        )

    def provider_or(
        self,
        default_provider: CellProfilerBackendProvider,
    ) -> CellProfilerBackendProvider:
        return normalize_cellprofiler_backend_provider(default_provider)


@dataclass(frozen=True, slots=True)
class ExplicitCellProfilerBackendProviderSelection(CellProfilerBackendProviderSelection):
    """Select one explicit CellProfiler backend provider without fallback."""

    registry_key = "explicit"
    provider: CellProfilerBackendProvider

    def backend_class(
        self,
        snapshot: CellProfilerBackendRegistrySnapshot,
    ) -> type[BackendStrategyT]:
        key = cellprofiler_backend_key(snapshot.memory_type, self.provider)
        try:
            return snapshot.registry[key]
        except KeyError as exc:
            raise NotImplementedError(
                f"No CellProfiler {snapshot.strategy_family.__name__} backend is "
                f"registered for memory type {snapshot.memory_type.value!r} and "
                f"provider {self.provider.value!r}. Registered providers for this "
                f"memory type: {snapshot.available_backend_providers()!r}."
            ) from exc

    def provider_or(
        self,
        default_provider: CellProfilerBackendProvider,
    ) -> CellProfilerBackendProvider:
        del default_provider
        return self.provider


DEFAULT_CELLPROFILER_BACKEND_SELECTION = (
    DefaultCellProfilerBackendProviderSelection()
)
BackendProviderInput: TypeAlias = (
    CellProfilerBackendProvider | CellProfilerBackendProviderSelection | None
)


def normalize_cellprofiler_memory_type(
    memory_type: MemoryType | str = MemoryType.NUMPY,
) -> MemoryType:
    """Resolve a memory type value using OpenHCS' canonical enum."""
    return memory_type if isinstance(memory_type, MemoryType) else MemoryType(str(memory_type))


def normalize_cellprofiler_backend_provider(
    backend_provider: CellProfilerBackendProvider = DEFAULT_CELLPROFILER_BACKEND_PROVIDER,
) -> CellProfilerBackendProvider:
    """Resolve a backend provider using the closed typed provider enum."""
    if not isinstance(backend_provider, CellProfilerBackendProvider):
        raise TypeError(
            "CellProfiler backend provider must be a "
            "CellProfilerBackendProvider enum value"
        )
    return backend_provider


def cellprofiler_backend_provider_selection(
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> CellProfilerBackendProviderSelection:
    """Return the nominal backend-provider selection policy."""
    if backend_provider is None:
        return DEFAULT_CELLPROFILER_BACKEND_SELECTION
    if isinstance(backend_provider, CellProfilerBackendProviderSelection):
        return backend_provider
    return ExplicitCellProfilerBackendProviderSelection(
        normalize_cellprofiler_backend_provider(backend_provider)
    )


def cellprofiler_backend_key(
    memory_type: MemoryType | str = MemoryType.NUMPY,
    backend_provider: CellProfilerBackendProvider = DEFAULT_CELLPROFILER_BACKEND_PROVIDER,
) -> str:
    """Return the registry key for one memory/provider backend implementation."""
    provider = normalize_cellprofiler_backend_provider(backend_provider)
    return (
        normalize_cellprofiler_memory_type(memory_type).value
        + _BACKEND_KEY_SEPARATOR
        + provider.value
    )


class CellProfilerBackendStrategyMixin(CompilerPreparedAutoRegisterFamily):
    """Mixin for backend strategies keyed by OpenHCS memory type and provider.

    Concrete strategy families keep their own AutoRegisterMeta registry; this
    mixin only standardizes lookup semantics so adding providers does not copy
    boilerplate across morphology, thresholding, watershed, and future modules.
    """

    backend_key: ClassVar[str | None] = None
    __registry__: ClassVar[dict[str, type["CellProfilerBackendStrategyMixin"]]]
    memory_type: ClassVar[MemoryType | None] = None
    backend_provider: ClassVar[CellProfilerBackendProvider] = (
        DEFAULT_CELLPROFILER_BACKEND_PROVIDER
    )
    is_default_backend: ClassVar[bool] = False

    @classmethod
    def prepare_registered_family(cls) -> None:
        """Prepare every registered backend implementation for compiler warmup."""
        snapshot = CellProfilerBackendRegistrySnapshot.for_family(
            cls,
            MemoryType.NUMPY,
        )
        _prepare_cellprofiler_backend_family_cached(snapshot)

    def prepare_backend(self) -> None:
        """Prepare this concrete backend implementation."""
        return

    @classmethod
    def requires_explicit_prepare_backend(cls) -> bool:
        """Return whether this provider must prewarm runtime-specialized code."""
        return cls.backend_provider.requires_compiler_prewarm

    @classmethod
    def for_memory_type(
        cls: type[BackendStrategyT],
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    ) -> BackendStrategyT:
        """Instantiate the exact backend for ``memory_type`` and provider.

        The default selection resolves the single default provider for the
        memory type. Explicit providers never fall back to another backend.
        """
        return cls._resolve_backend_class(memory_type, backend_provider)()

    @classmethod
    def for_callable(
        cls: type[BackendStrategyT],
        func: object,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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
        for strategy_cls in cls.__registry__.values():
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
        backend_provider: BackendProviderInput,
    ) -> type[BackendStrategyT]:
        snapshot = CellProfilerBackendRegistrySnapshot.for_family(
            cls,
            normalize_cellprofiler_memory_type(memory_type),
        )
        selection = cellprofiler_backend_provider_selection(backend_provider)
        return _resolve_backend_class_cached(snapshot, selection)


@lru_cache(maxsize=None)
def _resolve_backend_class_cached(
    snapshot: CellProfilerBackendRegistrySnapshot,
    selection: CellProfilerBackendProviderSelection,
) -> type[BackendStrategyT]:
    return selection.backend_class(snapshot)


@lru_cache(maxsize=None)
def _prepare_cellprofiler_backend_family_cached(
    snapshot: CellProfilerBackendRegistrySnapshot,
) -> None:
    for strategy_cls in snapshot.registry.values():
        if (
            strategy_cls.requires_explicit_prepare_backend()
            and strategy_cls.prepare_backend
            is CellProfilerBackendStrategyMixin.prepare_backend
        ):
            raise RuntimeError(
                f"{strategy_cls.__module__}.{strategy_cls.__name__} uses the "
                "NUMBA CellProfiler backend provider but does not implement "
                "prepare_backend(). Numba specializations must be compiled "
                "during OpenHCS compiler preparation, not first timed execution."
            )
        strategy_cls().prepare_backend()


__all__ = [
    "DEFAULT_CELLPROFILER_BACKEND_PROVIDER",
    "DEFAULT_CELLPROFILER_BACKEND_SELECTION",
    "BackendProviderInput",
    "CellProfilerBackendProvider",
    "CellProfilerBackendProviderSelection",
    "CellProfilerBackendStrategyMixin",
    "DefaultCellProfilerBackendProviderSelection",
    "ExplicitCellProfilerBackendProviderSelection",
    "cellprofiler_backend_key",
    "cellprofiler_backend_provider_selection",
    "normalize_cellprofiler_backend_provider",
    "normalize_cellprofiler_memory_type",
]
