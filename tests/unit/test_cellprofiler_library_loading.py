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
