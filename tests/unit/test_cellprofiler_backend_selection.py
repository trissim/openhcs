import pytest
import importlib
import inspect
import pkgutil

from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    DefaultCellProfilerBackendProviderSelection,
    ExplicitCellProfilerBackendProviderSelection,
    CellProfilerBackendAuthority,
)


def test_absent_backend_provider_uses_default_selection_policy() -> None:
    assert isinstance(
        CellProfilerBackendAuthority.provider_selection(None),
        DefaultCellProfilerBackendProviderSelection,
    )


def test_explicit_backend_provider_uses_explicit_selection_policy() -> None:
    selection = CellProfilerBackendAuthority.provider_selection(
        CellProfilerBackendProvider.NUMBA
    )

    assert isinstance(selection, ExplicitCellProfilerBackendProviderSelection)
    assert selection.provider is CellProfilerBackendProvider.NUMBA


def test_backend_provider_selection_identity_is_stable_and_nominal() -> None:
    assert CellProfilerBackendAuthority.selection_identity(None) == (
        ("selection", "default"),
    )
    assert CellProfilerBackendAuthority.selection_identity(
        CellProfilerBackendProvider.NUMBA
    ) == (
        ("selection", "explicit"),
        ("provider", "numba"),
    )


def test_backend_provider_normalizer_rejects_absent_provider() -> None:
    with pytest.raises(TypeError, match="CellProfiler backend provider"):
        CellProfilerBackendAuthority.provider(None)  # type: ignore[arg-type]


def test_all_numba_cellprofiler_backends_define_compiler_prewarm() -> None:
    import openhcs.processing.backends.cellprofiler as cellprofiler_backends

    missing: list[str] = []
    for module_info in pkgutil.iter_modules(cellprofiler_backends.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(
            f"{cellprofiler_backends.__name__}.{module_info.name}"
        )
        for _name, strategy_cls in inspect.getmembers(module, inspect.isclass):
            if strategy_cls.__module__ != module.__name__:
                continue
            if (
                not issubclass(strategy_cls, CellProfilerBackendStrategyMixin)
                or strategy_cls is CellProfilerBackendStrategyMixin
            ):
                continue
            if (
                strategy_cls.requires_explicit_prepare_backend()
                and strategy_cls.prepare_backend
                is CellProfilerBackendStrategyMixin.prepare_backend
            ):
                missing.append(f"{strategy_cls.__module__}.{strategy_cls.__name__}")

    assert missing == []
