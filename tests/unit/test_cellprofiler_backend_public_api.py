"""Public API tests for the absorbed CellProfiler backend package."""

import inspect
import importlib
import pickle

from python_introspect import SignatureAnalyzer

from openhcs.processing.backends.cellprofiler.function_documentation import (
    CELLPROFILER_FUNCTION_DOCUMENTATION_ATTR,
)


_SIGNATURE_TRANSPORT_PARAMETER_KINDS = frozenset(
    {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
)


def test_cellprofiler_backend_from_import_returns_function_for_submodule_name():
    from openhcs.processing.backends.cellprofiler import crop

    assert callable(crop)
    assert crop.__name__ == "crop"
    assert crop.__module__ == "openhcs.processing.backends.cellprofiler"


def test_cellprofiler_backend_function_export_survives_submodule_import():
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    importlib.import_module("openhcs.processing.backends.cellprofiler.crop")

    assert callable(cellprofiler_backend.crop)
    assert cellprofiler_backend.crop.__name__ == "crop"


def test_cellprofiler_backend_function_identity_survives_catalog_rebuild():
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    crop = cellprofiler_backend.crop
    cellprofiler_backend._cellprofiler_function_maps.cache_clear()

    assert crop is cellprofiler_backend.crop
    assert pickle.loads(pickle.dumps(crop)) is cellprofiler_backend.crop


def test_cellprofiler_backend_functions_expose_parameter_documentation():
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    threshold = cellprofiler_backend.threshold
    params = SignatureAnalyzer.analyze(threshold)

    assert hasattr(threshold, CELLPROFILER_FUNCTION_DOCUMENTATION_ATTR)
    assert params["threshold_method"].description
    assert "CellProfiler" in params["threshold_method"].description
    assert params["threshold_scope"].description
    assert params["smoothing"].description


def test_cellprofiler_backend_catalog_has_no_undocumented_configurable_parameters():
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    missing: list[str] = []
    for function_name in cellprofiler_backend.list_cellprofiler_functions():
        func = cellprofiler_backend.get_cellprofiler_function(function_name)
        analyzed = SignatureAnalyzer.analyze(func)
        for parameter_name, parameter in inspect.signature(func).parameters.items():
            if parameter.kind in _SIGNATURE_TRANSPORT_PARAMETER_KINDS:
                continue
            if parameter_name not in analyzed or not analyzed[parameter_name].description:
                missing.append(f"{function_name}.{parameter_name}")

    assert missing == []
