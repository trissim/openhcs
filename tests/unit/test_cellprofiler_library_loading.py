import importlib

from benchmark.cellprofiler_library import get_function


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


def test_export_to_spreadsheet_module_imports_cleanly():
    module = importlib.import_module(
        "benchmark.cellprofiler_library.functions.exporttospreadsheet"
    )

    assert module is not None
