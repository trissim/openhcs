"""Public API tests for the absorbed CellProfiler backend package."""

import importlib
import pickle


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
