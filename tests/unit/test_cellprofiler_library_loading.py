import importlib
import numpy as np

from benchmark.cellprofiler_library import (
    canonical_module_name,
    get_contract,
    get_function,
    list_modules,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    correct_illumination_calculate,
)
from benchmark.cellprofiler_library.functions.opening import opening
from openhcs.core.config import DtypeConfig
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def test_absorbed_registry_resolves_every_declared_function():
    unresolved_modules = tuple(
        module_name
        for module_name in list_modules()
        if get_contract(module_name) is not None and get_function(module_name) is None
    )

    assert unresolved_modules == ()


def test_active_absorbed_cellprofiler_functions_import_cleanly():
    function_names = (
        "ConvertObjectsToImage",
        "GrayToColor",
        "Opening",
        "OverlayOutlines",
    )

    loaded_functions = {name: get_function(name) for name in function_names}

    assert all(func is not None for func in loaded_functions.values())


def test_examplefly_absorbed_functions_import_cleanly():
    function_names = (
        "IdentifyPrimaryObjects",
        "IdentifySecondaryObjects",
        "IdentifyTertiaryObjects",
        "MeasureObjectSizeShape",
        "MeasureObjectIntensity",
        "MeasureTexture",
        "MeasureObjectNeighbors",
        "MeasureColocalization",
        "MeasureImageIntensity",
    )

    loaded_functions = {name: get_function(name) for name in function_names}

    assert all(func is not None for func in loaded_functions.values())


def test_legacy_cellprofiler_module_aliases_resolve_to_canonical_functions():
    assert canonical_module_name("MeasureCorrelation") == "MeasureColocalization"
    assert get_contract("MeasureCorrelation") == get_contract("MeasureColocalization")
    assert get_function("MeasureCorrelation") is get_function("MeasureColocalization")


def test_export_to_spreadsheet_module_imports_cleanly():
    module = importlib.import_module(
        "benchmark.cellprofiler_library.functions.exporttospreadsheet"
    )

    assert module is not None


def test_absorbed_processing_contract_metadata_does_not_act_as_validator():
    image = np.ones((8, 8), dtype=np.float32)

    result, stats = correct_illumination_calculate(image, dtype_config=DtypeConfig())

    assert result.shape == image.shape
    assert stats.calculation_type == "regular"
    assert (
        correct_illumination_calculate.__processing_contract__
        is ProcessingContract.PURE_2D
    )
    assert opening.__processing_contract__ is ProcessingContract.PURE_2D


def test_pure_2d_contract_wrapper_aggregates_tuple_outputs_per_slice():
    registry = OpenHCSRegistry()
    wrapped = registry.apply_contract_wrapper(
        correct_illumination_calculate,
        ProcessingContract.PURE_2D,
    )
    image = np.stack(
        (
            np.full((8, 8), 1.0, dtype=np.float32),
            np.full((8, 8), 2.0, dtype=np.float32),
        )
    )

    result, stats = wrapped(image, dtype_config=DtypeConfig())

    assert result.shape == image.shape
    assert len(stats) == 2
    assert [item.slice_index for item in stats] == [0, 1]
    assert all(item.mean_value > 0 for item in stats)
