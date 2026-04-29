import importlib
import numpy as np

from benchmark.cellprofiler_library import (
    canonical_module_name,
    get_contract,
    get_function,
    list_modules,
)
from benchmark.cellprofiler_library.functions.align import align
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    correct_illumination_calculate,
)
from benchmark.cellprofiler_library.functions.measureimageareaoccupied import (
    measure_image_area_occupied,
)
from benchmark.cellprofiler_library.functions.opening import opening
from benchmark.cellprofiler_library.functions.overlayoutlines import overlay_outlines
from benchmark.cellprofiler_library.functions.unmixcolors import unmix_colors
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


def test_unmix_colors_returns_one_output_per_stain_row():
    image = np.full((8, 9, 3), 0.5, dtype=np.float32)

    outputs = unmix_colors(
        image,
        stain_names=("Hematoxylin", "Eosin", "Custom"),
        custom_absorbances=(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.1, 0.2, 0.3),
        ),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(outputs, tuple)
    assert [output.shape for output in outputs] == [(8, 9), (8, 9), (8, 9)]
    assert all(output.dtype == np.float32 for output in outputs)
    assert unmix_colors.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_measure_image_area_occupied_runs_mixed_rows():
    binary = np.zeros((5, 6), dtype=np.float32)
    binary[1:3, 1:4] = 1.0
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[2:4, 2:5] = 1

    retained, measurements = measure_image_area_occupied(
        binary,
        operand_choices=("binary_image", "objects"),
        input_names=("DNA", "Nuclei"),
        retained_image_names=(None, "OccupiedNuclei"),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert retained.shape == labels.shape
    assert [measurement.slice_index for measurement in measurements] == [0, 1]
    assert all(measurement.area_occupied == 6.0 for measurement in measurements)
    assert measure_image_area_occupied.__processing_contract__ is (
        ProcessingContract.FLEXIBLE
    )


def test_measure_image_area_occupied_reduces_label_stacks_as_2d_planes():
    image = np.zeros((2, 5, 6), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 1:3, 1:4] = 1
    labels[1, 2:4, 2:5] = 1

    retained, measurements = measure_image_area_occupied(
        image,
        operand_choices=("objects",),
        input_names=("Nuclei",),
        retained_image_names=("OccupiedNuclei",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert retained.shape == labels.shape
    assert len(measurements) == 1
    assert measurements[0].area_occupied == 12.0
    assert measurements[0].total_area == 60.0
    assert measurements[0].perimeter > 0


def test_align_returns_two_registered_images():
    first = np.zeros((8, 8), dtype=np.float32)
    first[2:5, 2:5] = 1.0
    second = np.zeros_like(first)
    second[3:6, 2:5] = 1.0

    aligned_first, aligned_second = align(
        np.stack((first, second)),
        crop_mode="Keep size",
        dtype_config=DtypeConfig(),
    )

    assert aligned_first.shape == first.shape
    assert aligned_second.shape == second.shape
    assert align.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_overlay_outlines_runs_mixed_image_and_object_rows():
    base = np.zeros((8, 8), dtype=np.float32)
    outline_image = np.zeros_like(base)
    outline_image[1:6, 1] = 1.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1

    output = overlay_outlines(
        np.stack((base, outline_image)),
        outline_source_kinds=("image", "objects"),
        outline_colors=("Red", "Green"),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (8, 8, 3)
    assert output[..., 0].max() > 0
    assert output[..., 1].max() > 0
    assert overlay_outlines.__processing_contract__ is ProcessingContract.FLEXIBLE
