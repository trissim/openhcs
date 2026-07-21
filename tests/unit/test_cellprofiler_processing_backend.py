from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from python_introspect import Enableable, is_enableable

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.pipeline.function_contracts import execution_scope
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.runtime_object_labels import ObjectLabelVariantData
from openhcs.processing.backends.cellprofiler.morphology import (
    resize_objects,
    resize_objects_3d,
)
from openhcs.processing.backends.cellprofiler.spreadsheet_export import (
    export_to_spreadsheet,
)


def _processing_contract(function):
    return vars(function)[FunctionContractAttribute.processing_contract]


def test_openhcs_product_code_does_not_import_benchmark_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[tuple[str, int, str]] = []
    for file_path in sorted((repo_root / "openhcs").rglob("*.py")):
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "benchmark" or alias.name.startswith("benchmark."):
                        offenders.append(
                            (
                                str(file_path.relative_to(repo_root)),
                                node.lineno,
                                alias.name,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name == "benchmark" or module_name.startswith("benchmark."):
                    offenders.append(
                        (
                            str(file_path.relative_to(repo_root)),
                            node.lineno,
                            module_name,
                        )
                    )

    assert offenders == []


def test_cellprofiler_processing_backend_exports_declared_function() -> None:
    from openhcs.processing.backends import cellprofiler

    function = cellprofiler.identify_primary_objects
    module_type = CellProfilerModule.require_module("IdentifyPrimaryObjects")

    assert function.__module__ == module_type.__module__
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert _processing_contract(function) is ProcessingContract.PURE_2D
    assert function is module_type.require_callable("identify_primary_objects")


def test_display_data_on_image_treats_hwc_color_image_as_slice() -> None:
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.display_modules import (
        DisplayMode,
        ObjectsOrImage,
        display_data_on_image,
    )

    image = np.ones((8, 9, 3), dtype=np.float32)
    labels = np.zeros((8, 9), dtype=np.int32)
    labels[2:5, 3:7] = 1

    result = display_data_on_image(
        image,
        labels=ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        measurements=np.array([0.5], dtype=np.float32),
        objects_or_image=ObjectsOrImage.OBJECTS,
        display_mode=DisplayMode.TEXT,
    )

    assert result.shape == image.shape


def test_display_data_on_image_rejects_unprojected_nhwc_label_stack() -> None:
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.display_modules import (
        DisplayMode,
        ObjectsOrImage,
        display_data_on_image,
    )

    image = np.ones((2, 8, 9, 3), dtype=np.float32)
    labels = np.zeros((2, 8, 9), dtype=np.int32)
    labels[:, 2:5, 3:7] = 1

    with pytest.raises(ValueError, match="labels already projected"):
        display_data_on_image(
            image,
            labels=ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=labels)
            ),
            measurements=np.array([0.5], dtype=np.float32),
            objects_or_image=ObjectsOrImage.OBJECTS,
            display_mode=DisplayMode.TEXT,
        )


def test_classify_objects_aligns_dense_label_indexed_measurements_to_sparse_labels() -> (
    None
):
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.classification import (
        classify_objects_single_measurement,
    )

    image = np.ones((3, 4), dtype=np.float32)
    labels = np.array(
        [
            [0, 2, 2, 0],
            [5, 5, 0, 9],
            [0, 9, 9, 0],
        ],
        dtype=np.int32,
    )
    measurement_values = np.full(9, np.nan, dtype=np.float64)
    measurement_values[1] = 100.0
    measurement_values[4] = 500.0
    measurement_values[8] = 900.0

    _classified, result = classify_objects_single_measurement(
        image,
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        measurement_values=measurement_values,
        bin_choice="even",
        bin_count=3,
        low_threshold=0.0,
        high_threshold=900.0,
        bin_names="Small,Medium,Large",
    )

    result_row = result.row_mappings()[0]
    assert result_row["total_objects"] == 3
    assert result_row["object_classes"] == (
        '{"2": "Small", "5": "Medium", "9": "Large"}'
    )


def test_classify_objects_measurement_vector_does_not_consume_rgb_main_flow() -> None:
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.classification import (
        classify_objects_single_measurement,
    )

    image = np.ones((3, 4, 3), dtype=np.float32)
    labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    classified, result = classify_objects_single_measurement(
        image,
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        measurement_values=np.array([0.25, 0.75], dtype=np.float64),
        bin_choice="custom",
        custom_thresholds="0,0.5,1",
        bin_names="Low,High",
    )

    np.testing.assert_array_equal(classified, labels)
    assert result.row_mappings()[0]["object_classes"] == (
        '{"1": "Low", "2": "High"}'
    )


def test_classify_objects_variants_reject_unprojected_label_stacks() -> None:
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.classification import (
        classify_objects_by_intensity_bins,
        classify_objects_two_measurements,
    )

    image = np.ones((3, 4), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 3, 4), dtype=np.int32))
    )

    for function in (
        classify_objects_two_measurements,
        classify_objects_by_intensity_bins,
    ):
        with pytest.raises(ValueError, match="already projected to one 2-D plane"):
            function(image, labels)


def test_cellprofiler_processing_backend_exports_canonical_grid_callable() -> None:
    from openhcs.processing.backends import cellprofiler

    function = CellProfilerModule.require_module(
        "IdentifyObjectsInGrid"
    ).require_callable()

    assert function is cellprofiler.identify_objects_in_grid
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert _processing_contract(function) is ProcessingContract.PURE_2D


def test_cellprofiler_processing_backend_exports_resize_volumetric_variant() -> None:
    from openhcs.processing.backends import cellprofiler

    function = CellProfilerModule.require_module("Resize").require_callable(
        "resize_volumetric"
    )

    assert function is cellprofiler.resize_volumetric
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert _processing_contract(function) is ProcessingContract.PURE_3D


def test_cellprofiler_processing_backend_exports_resize_objects_volumetric_variant() -> (
    None
):
    from openhcs.processing.backends import cellprofiler

    function = CellProfilerModule.require_module("ResizeObjects").require_callable(
        "resize_objects_3d"
    )

    assert function is cellprofiler.resize_objects_3d
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert _processing_contract(function) is ProcessingContract.PURE_3D


@pytest.mark.parametrize(
    ("function", "shape"),
    ((resize_objects, (4, 5)), (resize_objects_3d, (2, 4, 5))),
)
def test_resize_objects_replaces_source_spatial_domain(
    function,
    shape: tuple[int, ...],
) -> None:
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
        object_label_dense_array,
    )
    from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
    from openhcs.core.source_spatial_domain import SourceSpatialDomain

    labels = np.zeros(shape, dtype=np.int32)
    labels[(..., slice(1, 3), slice(1, 3))] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(3, 4),
            source_shape_yx=(10, 12),
        ),
    )
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__

    _image, _stats, resized, _relationship = function(
        np.zeros(shape, dtype=np.float32),
        payload,
        factor_x=2.0,
        factor_y=2.0,
        factor_z=1.0,
    )

    assert object_label_dense_array(resized).shape[-2:] == (8, 10)
    assert resized.source_spatial_domain.origin_yx == (0, 0)
    assert resized.source_spatial_domain.source_shape_yx == (8, 10)


def test_measure_object_size_shape_retains_dense_label_plane_coordinates() -> None:
    from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.core.source_spatial_domain import SourceSpatialDomain
    from openhcs.processing.backends.cellprofiler.shape import measure_object_size_shape

    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:3, 2:4] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(10, 20),
            source_shape_yx=(100, 100),
        ),
    )

    _image, rows = measure_object_size_shape.__wrapped__(
        np.zeros_like(labels, dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert len(rows) == 1
    assert rows[0]["Center_X"] == 2.5
    assert rows[0]["Center_Y"] == 1.5
    assert payload.source_spatial_domain.origin_yx == (10, 20)


def test_measure_object_size_shape_restores_only_sparse_local_patch_coordinates() -> (
    None
):
    from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
        ObjectLabelRepresentation,
    )
    from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
    from openhcs.core.source_spatial_domain import SourceSpatialDomain
    from openhcs.processing.backends.cellprofiler.shape import measure_object_size_shape

    sparse_labels = SparseIJVLabelRows(
        np.asarray(
            (
                (1, 2, 1),
                (1, 3, 1),
                (2, 2, 1),
                (2, 3, 1),
            ),
            dtype=np.int32,
        )
    )
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=sparse_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(10, 20),
            source_shape_yx=(100, 100),
        ),
    )

    _image, rows = measure_object_size_shape.__wrapped__(
        np.zeros((5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert len(rows) == 1
    assert rows[0]["Center_X"] == 2.5
    assert rows[0]["Center_Y"] == 1.5
    assert payload.source_spatial_domain.origin_yx == (10, 20)


def test_openhcs_registry_discovers_cellprofiler_backend_contracts() -> None:
    registry = OpenHCSRegistry()
    registry.MODULES_TO_SCAN = ["openhcs.processing.backends.cellprofiler"]

    functions = registry.discover_functions()

    metadata = functions["cellprofiler_identify_primary_objects"]
    assert metadata.contract is ProcessingContract.PURE_2D
    assert metadata.func.input_memory_type == MemoryType.NUMPY.value
    assert "cellprofiler" in metadata.tags
    assert "cellprofiler_identify_objects_in_grid" in functions
    spreadsheet = functions["cellprofiler_export_to_spreadsheet"]
    assert spreadsheet.original_name == "export_to_spreadsheet"
    assert spreadsheet.func is not export_to_spreadsheet
    assert spreadsheet.func.__wrapped__ is export_to_spreadsheet
    assert is_enableable(spreadsheet.func)
    assert CallableContract.from_callable(
        spreadsheet.func
    ).resolve_canonical_raw_callable() is export_to_spreadsheet
    assert (
        Enableable.require_parameter_name()
        in inspect.signature(spreadsheet.func).parameters
    )


def test_openhcs_registry_cache_invalidates_when_scanned_modules_change(
    tmp_path,
) -> None:
    registry = OpenHCSRegistry()
    registry._cache_path = tmp_path / "openhcs_function_metadata.json"
    registry.MODULES_TO_SCAN = []

    assert registry.load_or_discover_functions() == {}

    registry.MODULES_TO_SCAN = ["openhcs.processing.backends.cellprofiler"]
    functions = registry.load_or_discover_functions()

    assert "cellprofiler_identify_primary_objects" in functions


def test_openhcs_registry_cache_preserves_plate_scoped_callable_contract(
    tmp_path,
) -> None:
    cache_path = tmp_path / "openhcs_function_metadata.json"
    discovery_registry = OpenHCSRegistry()
    discovery_registry._cache_path = cache_path
    discovery_registry.MODULES_TO_SCAN = [
        "openhcs.processing.backends.cellprofiler"
    ]
    discovery_registry.load_or_discover_functions()

    cache_registry = OpenHCSRegistry()
    cache_registry._cache_path = cache_path
    cache_registry.MODULES_TO_SCAN = [
        "openhcs.processing.backends.cellprofiler"
    ]
    cached = cache_registry.load_or_discover_functions()[
        "cellprofiler_export_to_spreadsheet"
    ]
    contract = CallableContract.from_callable(cached.func)

    assert cached.func is not export_to_spreadsheet
    assert cached.func.__wrapped__ is export_to_spreadsheet
    assert is_enableable(cached.func)
    assert contract.execution_scope is FunctionStepExecutionScope.PLATE
    assert contract.runtime_bound_parameters == ("artifact_batch",)
    assert contract.resolve_canonical_raw_callable() is export_to_spreadsheet


def test_shared_registry_wrapper_preserves_plate_call_semantics() -> None:
    @execution_scope(FunctionStepExecutionScope.PLATE)
    def plate_probe(*, value: int) -> int:
        return value

    wrapped = OpenHCSRegistry().apply_contract_wrapper(
        plate_probe,
        ProcessingContract.FLEXIBLE,
    )

    assert wrapped is not plate_probe
    assert wrapped(value=3, enabled=False) == 3
    assert is_enableable(wrapped)
    assert CallableContract.from_callable(wrapped).execution_scope is (
        FunctionStepExecutionScope.PLATE
    )
    assert CallableContract.from_callable(wrapped).resolve_canonical_raw_callable() is (
        plate_probe
    )


def test_cellprofiler_threshold_diagnostics_backend_resolves_numpy() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumbaNumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
    )

    strategy = ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    binary = image > 0.5

    weighted_variance, sum_of_entropies = strategy.diagnostics(image, mask, binary)

    assert type(strategy) is NumbaNumpyThresholdDiagnosticsBackendStrategy
    assert weighted_variance >= 0.0
    assert sum_of_entropies <= 0.0


def test_cellprofiler_threshold_diagnostics_handles_nd_images_as_one_domain() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
    )

    strategy = ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    binary = image > 0.5

    weighted_variance, sum_of_entropies = strategy.diagnostics(image, None, binary)
    reference = NumpyThresholdDiagnosticsBackendStrategy()
    flat_image = image.reshape(-1, 1)
    flat_binary = binary.reshape(-1, 1)
    flat_mask = np.ones(flat_image.shape, dtype=bool)
    reference_weighted_variance = reference.weighted_variance(
        flat_image,
        flat_mask,
        flat_binary,
    )
    reference_sum_of_entropies = reference.sum_of_entropies(
        flat_image,
        flat_mask,
        flat_binary,
    )

    assert isinstance(weighted_variance, float)
    assert isinstance(sum_of_entropies, float)
    np.testing.assert_allclose(
        weighted_variance,
        reference_weighted_variance,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        sum_of_entropies,
        reference_sum_of_entropies,
        rtol=1e-6,
    )


def test_cellprofiler_threshold_diagnostics_prewarm_covers_runtime_paths() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
    )

    ThresholdDiagnosticsBackendStrategy.prepare_registered_family()
    strategy = ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    reference = NumpyThresholdDiagnosticsBackendStrategy()
    image = np.array(
        [
            [[0.1, 0.0, 0.3], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    binary = image > 0.5
    partial_mask = np.ones(image.shape, dtype=bool)
    partial_mask[:, :, :1] = False

    for mask in (None, partial_mask):
        weighted_variance, sum_of_entropies = strategy.diagnostics(image, mask, binary)
        reference_mask = (
            np.ones(image.shape, dtype=bool) if mask is None else partial_mask
        )
        reference_weighted_variance = reference.weighted_variance(
            image.reshape(-1, 1),
            reference_mask.reshape(-1, 1),
            binary.reshape(-1, 1),
        )
        reference_sum_of_entropies = reference.sum_of_entropies(
            image.reshape(-1, 1),
            reference_mask.reshape(-1, 1),
            binary.reshape(-1, 1),
        )
        np.testing.assert_allclose(
            weighted_variance,
            reference_weighted_variance,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            sum_of_entropies,
            reference_sum_of_entropies,
            rtol=1e-6,
        )


def test_threshold_application_smoothing_matches_cellprofiler_mask_formula() -> None:
    from scipy import ndimage as ndi

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdApplicationRequest,
    )

    image = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape((5, 7))
    mask = np.ones(image.shape, dtype=bool)
    mask[:1, :2] = False
    smoothing = 1.3488
    sigma = smoothing / 0.6744 / 2.0
    expected_weight = ndi.gaussian_filter(
        mask.astype(float),
        sigma,
        mode="constant",
        cval=0,
    )
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    expected_smoothed = ndi.gaussian_filter(
        masked_image,
        sigma,
        mode="constant",
        cval=0,
    ) / (expected_weight + np.finfo(float).eps)
    expected = (expected_smoothed >= 0.5) & mask

    actual, actual_sigma = ThresholdApplicationRequest(
        image=image,
        threshold=0.5,
        mask=mask,
        smoothing=smoothing,
    ).apply()

    assert actual_sigma == sigma
    np.testing.assert_array_equal(actual, expected)


def test_threshold_selection_scale_policy_accepts_uint16_unit_interval_proof() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        unit_interval_scale_for_threshold_selection,
    )

    image = np.asarray([[0, 65535]], dtype=np.uint16)
    metadata = ImagePayloadMetadata.for_array(image).with_unit_interval_intensity_scale(
        65535
    )

    assert unit_interval_scale_for_threshold_selection(image, metadata) == 65535


def test_cellprofiler_uint8_source_normalization_matches_native_pixel_domain() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        MaskedImagePayload,
        normalize_image_payload_intensity,
    )
    from openhcs.interop.cellprofiler.image_normalization import (
        normalize_cellprofiler_image_payload,
    )

    image = np.asarray([[7, 98, 128, 254, 255]], dtype=np.uint8)
    core_normalized = np.asarray(normalize_image_payload_intensity(image))
    pathless_normalized = np.asarray(normalize_cellprofiler_image_payload(image))
    jpg_payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=bool),
        metadata=ImagePayloadMetadata.for_array(image, source_path="source.jpg"),
    )
    png_payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=bool),
        metadata=ImagePayloadMetadata.for_array(image, source_path="source.png"),
    )
    jpg_normalized = np.asarray(normalize_cellprofiler_image_payload(jpg_payload))
    cellprofiler_normalized = np.asarray(
        normalize_cellprofiler_image_payload(png_payload)
    )

    np.testing.assert_array_equal(pathless_normalized, core_normalized)
    np.testing.assert_array_equal(jpg_normalized, core_normalized)
    assert cellprofiler_normalized.dtype == np.float32
    assert cellprofiler_normalized[0, 0].view(np.uint32) == np.nextafter(
        core_normalized[0, 0], np.float32(0.0), dtype=np.float32
    ).view(np.uint32)
    assert cellprofiler_normalized[0, 1].view(np.uint32) == np.nextafter(
        core_normalized[0, 1], np.float32(0.0), dtype=np.float32
    ).view(np.uint32)
    assert cellprofiler_normalized[0, 2].view(np.uint32) == core_normalized[0, 2].view(
        np.uint32
    )
    assert cellprofiler_normalized[0, 3].view(np.uint32) == core_normalized[0, 3].view(
        np.uint32
    )
    assert cellprofiler_normalized[0, 4] == np.float32(1.0)

    normalized_payload = MaskedImagePayload(
        data=core_normalized,
        mask=np.ones(core_normalized.shape, dtype=bool),
        metadata=ImagePayloadMetadata.for_array(
            core_normalized, source_path="source.png"
        ).with_unit_interval_intensity_scale(255),
    )
    remapped = np.asarray(normalize_cellprofiler_image_payload(normalized_payload))

    np.testing.assert_array_equal(remapped, cellprofiler_normalized)


@pytest.mark.parametrize(
    ("virtual_path", "filter_path", "uses_png_domain"),
    (
        ("virtual/source.tif", "/physical/source.png", True),
        ("virtual/source.png", "/physical/source.tif", False),
    ),
)
def test_cellprofiler_uint8_normalization_uses_authoritative_filter_paths(
    monkeypatch,
    virtual_path: str,
    filter_path: str,
    uses_png_domain: bool,
) -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        MaskedImagePayload,
        normalize_image_payload_intensity,
    )
    from openhcs.core.source_metadata import (
        SOURCE_FILTER_PATHS_METADATA_FIELD,
        SourceFilterPathMetadata,
    )
    from openhcs.interop.cellprofiler.image_normalization import (
        normalize_cellprofiler_image_payload,
    )

    image = np.asarray([[7, 98, 128, 254, 255]], dtype=np.uint8)
    png_payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=bool),
        metadata=ImagePayloadMetadata.for_array(image, source_path="source.png"),
    )
    png_domain = np.asarray(normalize_cellprofiler_image_payload(png_payload))
    numpy_domain = np.asarray(normalize_image_payload_intensity(image))
    metadata = ImagePayloadMetadata.for_array(
        image,
        source_path=virtual_path,
    ).replace_fields(
        source_component_metadata={
            SOURCE_FILTER_PATHS_METADATA_FIELD: SourceFilterPathMetadata.from_paths(
                (filter_path,)
            ).as_dict()
        }
    )
    payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=bool),
        metadata=metadata,
    )

    def fail_source_image_path_reconstruction(_metadata) -> tuple[str, ...]:
        raise AssertionError("authoritative filter paths must bypass source identities")

    monkeypatch.setattr(
        ImagePayloadMetadata,
        "source_image_paths",
        property(fail_source_image_path_reconstruction),
    )

    observed = np.asarray(normalize_cellprofiler_image_payload(payload))

    np.testing.assert_array_equal(
        observed,
        png_domain if uses_png_domain else numpy_domain,
    )


def test_cellprofiler_backend_selection_is_memory_provider_keyed() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
        CellProfilerBackendAuthority,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        CentrosomeNumpyMorphologyBackendStrategy,
        MorphologyBackendStrategy,
        NumbaNumpyMorphologyBackendStrategy,
        NumpyMorphologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        NumbaNumpyObjectIntensityBackendStrategy,
        ObjectIntensityBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity_distribution import (
        NativeNumpyRadialDistributionBackendStrategy,
        NumbaNumpyRadialDistributionBackendStrategy,
        RadialDistributionBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NeighborTopologyBackendStrategy,
        NumbaNumpyNeighborTopologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        LegacyFastNumpyShapeMeasurementBackendStrategy,
        NumbaNumpyShapeMeasurementBackendStrategy,
        ShapeMeasurementBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        CentrosomeNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdDiagnosticsBackendStrategy,
        NumbaNumpyThresholdSmoothingBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
        ThresholdPrimitiveBackendStrategy,
        ThresholdSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
        NumbaNumpyLegacyWatershedBackendStrategy,
        NumpyLegacyWatershedBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        LegacyFastNumpyShapeZernikeBackendStrategy,
        NativeNumpyShapeZernikeBackendStrategy,
        ShapeZernikeBackendStrategy,
    )

    assert CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY) == (
        f"{MemoryType.NUMPY.value}:{CellProfilerBackendProvider.NATIVE.value}"
    )
    assert type(MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyMorphologyBackendStrategy
    )
    assert (
        type(
            MorphologyBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NATIVE,
            )
        )
        is NumpyMorphologyBackendStrategy
    )
    assert (
        type(
            MorphologyBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.CENTROSOME,
            )
        )
        is CentrosomeNumpyMorphologyBackendStrategy
    )
    assert (
        type(
            MorphologyBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NUMBA,
            )
        )
        is NumbaNumpyMorphologyBackendStrategy
    )
    assert type(
        ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    ) is (NumbaNumpyThresholdDiagnosticsBackendStrategy)
    assert type(
        ThresholdSmoothingBackendStrategy.for_memory_type(MemoryType.NUMPY)
    ) is (NumbaNumpyThresholdSmoothingBackendStrategy)
    assert (
        type(
            ThresholdSmoothingBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NUMBA,
            )
        )
        is NumbaNumpyThresholdSmoothingBackendStrategy
    )
    assert type(
        ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    ) is (NumbaNumpyThresholdPrimitiveBackendStrategy)
    assert type(
        ThresholdPrimitiveBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (NumbaNumpyThresholdPrimitiveBackendStrategy)
    assert (
        type(
            ThresholdPrimitiveBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.CENTROSOME,
            )
        )
        is CentrosomeNumpyThresholdPrimitiveBackendStrategy
    )
    assert type(LegacyWatershedBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyLegacyWatershedBackendStrategy
    )
    assert (
        type(
            LegacyWatershedBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NATIVE,
            )
        )
        is NumpyLegacyWatershedBackendStrategy
    )
    assert type(
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (NumbaNumpyLegacyWatershedBackendStrategy)
    assert type(ObjectIntensityBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyObjectIntensityBackendStrategy
    )
    assert (
        type(
            ObjectIntensityBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NUMBA,
            )
        )
        is NumbaNumpyObjectIntensityBackendStrategy
    )
    assert type(
        RadialDistributionBackendStrategy.for_memory_type(MemoryType.NUMPY)
    ) is (NativeNumpyRadialDistributionBackendStrategy)
    assert (
        type(
            RadialDistributionBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NATIVE,
            )
        )
        is NativeNumpyRadialDistributionBackendStrategy
    )
    assert (
        type(
            RadialDistributionBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NUMBA,
            )
        )
        is NumbaNumpyRadialDistributionBackendStrategy
    )
    assert type(NeighborTopologyBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyNeighborTopologyBackendStrategy
    )
    assert (
        type(
            NeighborTopologyBackendStrategy.for_memory_type(
                MemoryType.NUMPY,
                backend_provider=CellProfilerBackendProvider.NUMBA,
            )
        )
        is NumbaNumpyNeighborTopologyBackendStrategy
    )
    assert type(ShapeMeasurementBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        LegacyFastNumpyShapeMeasurementBackendStrategy
    )
    assert type(
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (NumbaNumpyShapeMeasurementBackendStrategy)
    with pytest.raises(NotImplementedError):
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
    assert type(ShapeZernikeBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NativeNumpyShapeZernikeBackendStrategy
    )
    assert type(
        ShapeZernikeBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        )
    ) is (LegacyFastNumpyShapeZernikeBackendStrategy)
    with pytest.raises(NotImplementedError):
        ShapeZernikeBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )


def test_watershed_sparse_peak_markers_match_mahotas_dense_connectivity() -> None:
    import mahotas
    from openhcs.processing.backends.cellprofiler.watershed import (
        _sparse_connected_peak_markers_3d,
    )

    peaks = np.zeros((12, 14, 16), dtype=bool)
    peak_coordinates = np.array(
        (
            (1, 1, 1),
            (1, 1, 9),
            (1, 1, 10),
            (8, 8, 8),
            (11, 13, 15),
        ),
        dtype=np.int64,
    )
    peaks[
        peak_coordinates[:, 0],
        peak_coordinates[:, 1],
        peak_coordinates[:, 2],
    ] = True
    connectivity = np.ones((16, 16, 16), dtype=bool)

    expected, expected_count = mahotas.label(peaks, connectivity)
    actual_tuple = _sparse_connected_peak_markers_3d(peaks, connectivity)

    assert actual_tuple is not None
    actual, actual_count = actual_tuple
    assert actual_count == expected_count
    np.testing.assert_array_equal(actual, expected)


def test_numba_neighbor_topology_outline_frontier_matches_dense_reference() -> None:
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NumbaNumpyNeighborTopologyBackendStrategy,
    )

    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[2:8, 11:17] = 2
    labels[12:18, 3:9] = 3
    perimeter = _labeled_perimeter(labels)
    footprint = _disk_footprint(4)
    touching_footprint = _disk_footprint(4.5)

    observed = NumbaNumpyNeighborTopologyBackendStrategy().measure_topology(
        labels,
        labels,
        perimeter,
        np.array([1, 2, 3], dtype=np.int32),
        distance=4,
        neighbors_are_same_objects=True,
        footprint=footprint,
        touching_footprint=touching_footprint,
        variant_object_count=3,
        variant_neighbor_count=3,
    )
    (
        expected_neighbor_count,
        expected_touching,
        expected_adjacency,
    ) = _dense_neighbor_topology_reference(
        labels,
        labels,
        perimeter,
        footprint,
        touching_footprint,
        neighbors_are_same_objects=True,
        variant_object_count=3,
        variant_neighbor_count=3,
    )

    np.testing.assert_array_equal(observed.neighbor_count, expected_neighbor_count)
    np.testing.assert_array_equal(observed.touching_pixel_count, expected_touching)
    expected_pairs = np.argwhere(expected_adjacency)
    np.testing.assert_array_equal(
        observed.source_variant_numbers,
        expected_pairs[:, 0] + 1,
    )
    np.testing.assert_array_equal(
        observed.target_variant_numbers,
        expected_pairs[:, 1] + 1,
    )


def test_numba_neighbor_topology_packed_edges_cross_word_boundary_in_order() -> None:
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NumbaNumpyNeighborTopologyBackendStrategy,
    )

    working_labels = np.zeros((5, 5), dtype=np.int32)
    working_labels[2, 2] = 1
    neighbor_labels = np.zeros_like(working_labels)
    neighbor_labels[2, 1] = 65
    neighbor_labels[2, 3] = 2
    footprint = _disk_footprint(1)

    observed = NumbaNumpyNeighborTopologyBackendStrategy().measure_topology(
        working_labels,
        neighbor_labels,
        working_labels,
        np.array([1], dtype=np.int32),
        distance=1,
        neighbors_are_same_objects=False,
        footprint=footprint,
        touching_footprint=footprint,
        variant_object_count=1,
        variant_neighbor_count=65,
    )

    np.testing.assert_array_equal(observed.neighbor_count, np.array([2.0]))
    np.testing.assert_array_equal(
        observed.source_variant_numbers,
        np.array([1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        observed.target_variant_numbers,
        np.array([2, 65], dtype=np.int32),
    )


def test_numba_closest_neighbors_reuses_exact_identity_mapping() -> None:
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NumbaNumpyNeighborTopologyBackendStrategy,
    )

    centers = np.array(((0.0, 0.0), (0.0, 2.0), (3.0, 0.0)))
    object_numbers = np.array([1, 2, 3], dtype=np.int32)
    has_pixels = np.ones(3, dtype=bool)

    observed = NumbaNumpyNeighborTopologyBackendStrategy().closest_neighbors(
        centers,
        centers,
        object_numbers,
        object_numbers,
        has_pixels,
        has_pixels,
        neighbors_are_same_objects=True,
        variant_object_count=3,
        variant_neighbor_count=3,
        final_object_count=3,
    )

    np.testing.assert_array_equal(
        observed.final_first_object_number,
        np.array([2, 1, 1]),
    )
    np.testing.assert_array_equal(
        observed.final_second_object_number,
        np.array([3, 3, 2]),
    )
    np.testing.assert_array_equal(
        observed.first_x_vector,
        np.array([2.0, -2.0, 0.0]),
    )
    np.testing.assert_array_equal(
        observed.first_y_vector,
        np.array([0.0, 0.0, -3.0]),
    )


def test_numba_closest_neighbors_keeps_nonidentity_final_mapping_search() -> None:
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NumbaNumpyNeighborTopologyBackendStrategy,
    )

    centers = np.array(((0.0, 0.0), (0.0, 10.0)))
    object_numbers = np.array([1, 1, 2], dtype=np.int32)
    has_pixels = np.ones(3, dtype=bool)

    observed = NumbaNumpyNeighborTopologyBackendStrategy().closest_neighbors(
        centers,
        centers,
        object_numbers,
        object_numbers,
        has_pixels,
        has_pixels,
        neighbors_are_same_objects=True,
        variant_object_count=2,
        variant_neighbor_count=2,
        final_object_count=3,
    )

    np.testing.assert_array_equal(
        observed.final_first_object_number,
        np.array([2, 1, 1]),
    )
    np.testing.assert_array_equal(
        observed.final_second_object_number,
        np.array([3, 3, 2]),
    )


def _disk_footprint(radius: float) -> np.ndarray:
    limit = int(np.ceil(radius))
    yy, xx = np.ogrid[-limit : limit + 1, -limit : limit + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _labeled_perimeter(labels: np.ndarray) -> np.ndarray:
    perimeter = np.zeros_like(labels)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            object_number = labels[y, x]
            if object_number == 0:
                continue
            for offset_y, offset_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                    or labels[neighbor_y, neighbor_x] != object_number
                ):
                    perimeter[y, x] = object_number
                    break
    return perimeter


def _dense_neighbor_topology_reference(
    working_labels: np.ndarray,
    neighbor_working_labels: np.ndarray,
    perimeter_outlines: np.ndarray,
    footprint: np.ndarray,
    touching_footprint: np.ndarray,
    *,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjacency = np.zeros((variant_object_count, variant_neighbor_count + 1), dtype=bool)
    touching_pixel_count = np.zeros(variant_object_count, dtype=float)
    height, width = working_labels.shape
    offsets = _footprint_offsets(footprint)
    touching_offsets = _footprint_offsets(touching_footprint)

    for y in range(height):
        for x in range(width):
            object_number = working_labels[y, x]
            if object_number <= 0 or object_number > variant_object_count:
                continue
            object_index = object_number - 1
            for offset_y, offset_x in offsets:
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                neighbor_number = neighbor_working_labels[neighbor_y, neighbor_x]
                if neighbor_number <= 0 or neighbor_number > variant_neighbor_count:
                    continue
                if neighbors_are_same_objects and neighbor_number == object_number:
                    continue
                adjacency[object_index, neighbor_number] = True

            if perimeter_outlines[y, x] != object_number:
                continue
            for offset_y, offset_x in touching_offsets:
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                if neighbors_are_same_objects:
                    touches = (
                        working_labels[neighbor_y, neighbor_x] != 0
                        and working_labels[neighbor_y, neighbor_x] != object_number
                    )
                else:
                    touches = neighbor_working_labels[neighbor_y, neighbor_x] != 0
                if touches:
                    touching_pixel_count[object_index] += 1.0
                    break

    return (
        adjacency[:, 1:].sum(axis=1).astype(float),
        touching_pixel_count,
        adjacency[:, 1:],
    )


def _footprint_offsets(footprint: np.ndarray) -> tuple[tuple[int, int], ...]:
    center_y = footprint.shape[0] // 2
    center_x = footprint.shape[1] // 2
    return tuple(
        (int(y - center_y), int(x - center_x)) for y, x in np.argwhere(footprint)
    )


def test_shape_distance_to_edge_handles_stacked_planes_planewise() -> None:
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeMeasurementBackendStrategy,
    )

    backend = ShapeMeasurementBackendStrategy.for_memory_type(MemoryType.NUMPY)
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 2:4, 2:4] = 2

    distances = backend.distance_to_edge(labels)

    np.testing.assert_allclose(distances[0], backend.distance_to_edge(labels[0]))
    np.testing.assert_allclose(distances[1], backend.distance_to_edge(labels[1]))


def test_default_shape_maximum_position_preserves_cellprofiler_tie_semantics() -> None:
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeMeasurementBackendStrategy,
    )

    default_backend = ShapeMeasurementBackendStrategy.for_memory_type(MemoryType.NUMPY)
    labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 2],
            [3, 0, 2, 2],
            [3, 3, 0, 0],
        ],
        dtype=np.int32,
    )
    image = default_backend.distance_to_edge(labels)
    label_ids = np.arange(1, int(labels.max()) + 1, dtype=np.int32)

    actual_i, actual_j = default_backend.maximum_position_of_labels(
        image,
        labels,
        label_ids,
    )

    np.testing.assert_array_equal(actual_i, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(actual_j, np.array([2.0, 3.0, 0.0]))


def test_medianfilter_preserves_unit_interval_scale_metadata() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        image_payload_data,
        image_payload_metadata,
        normalize_image_payload_intensity,
    )
    from openhcs.processing.backends.cellprofiler.median_filter import medianfilter

    raw = np.arange(25, dtype=np.uint16).reshape(5, 5)
    payload = ImagePayloadMetadata.for_array(raw).payload_with(raw, None)
    normalized = normalize_image_payload_intensity(payload, dtype=np.float32)

    filtered = medianfilter.__wrapped__(normalized, window_size=3)

    assert image_payload_metadata(filtered).unit_interval_intensity_scale == 65535
    assert image_payload_data(filtered).dtype == np.float32


def test_rescale_intensity_identity_preserves_unit_interval_scale_metadata() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        image_payload_metadata,
        normalize_image_payload_intensity,
    )
    from openhcs.processing.backends.cellprofiler.intensity import rescale_intensity

    raw = np.array([[0, 65535], [32768, 1]], dtype=np.uint16)
    payload = ImagePayloadMetadata.for_array(raw).payload_with(raw, None)
    normalized = normalize_image_payload_intensity(payload, dtype=np.float32)

    rescaled = rescale_intensity.__wrapped__(
        normalized,
        rescale_method="stretch",
        automatic_low="custom",
        automatic_high="custom",
        source_low=0.0,
        source_high=1.0,
        dest_low=0.0,
        dest_high=1.0,
    )

    assert image_payload_metadata(rescaled).unit_interval_intensity_scale == 65535


def test_rescale_intensity_nonidentity_clears_unit_interval_scale_metadata() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        image_payload_metadata,
        normalize_image_payload_intensity,
    )
    from openhcs.processing.backends.cellprofiler.intensity import rescale_intensity

    raw = np.array([[0, 65535], [32768, 1]], dtype=np.uint16)
    payload = ImagePayloadMetadata.for_array(raw).payload_with(raw, None)
    normalized = normalize_image_payload_intensity(payload, dtype=np.float32)

    rescaled = rescale_intensity.__wrapped__(
        normalized,
        rescale_method="manual_io_range",
        automatic_low="custom",
        automatic_high="custom",
        source_low=0.0,
        source_high=1.0,
        dest_low=0.0,
        dest_high=0.5,
    )

    assert image_payload_metadata(rescaled).unit_interval_intensity_scale is None


def test_resize_nearest_preserves_unit_interval_scale_metadata() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        image_payload_metadata,
        normalize_image_payload_intensity,
    )
    from openhcs.processing.backends.cellprofiler.image_geometry import (
        resize_volumetric,
    )

    raw = np.arange(2 * 4 * 4, dtype=np.uint16).reshape((2, 4, 4))
    payload = ImagePayloadMetadata.for_array(raw).payload_with(raw, None)
    normalized = normalize_image_payload_intensity(payload, dtype=np.float32)

    resized = resize_volumetric.__wrapped__(
        normalized,
        resize_method="by_factor",
        resizing_factor_x=1.0,
        resizing_factor_y=1.0,
        resizing_factor_z=1.0,
        interpolation="nearest_neighbor",
    )

    assert image_payload_metadata(resized).unit_interval_intensity_scale == 65535


def test_resize_interpolation_clears_unit_interval_scale_metadata() -> None:
    from openhcs.core.runtime_image_values import (
        ImagePayloadMetadata,
        image_payload_metadata,
        normalize_image_payload_intensity,
    )
    from openhcs.processing.backends.cellprofiler.image_geometry import (
        resize_volumetric,
    )

    raw = np.arange(2 * 4 * 4, dtype=np.uint16).reshape((2, 4, 4))
    payload = ImagePayloadMetadata.for_array(raw).payload_with(raw, None)
    normalized = normalize_image_payload_intensity(payload, dtype=np.float32)

    resized = resize_volumetric.__wrapped__(
        normalized,
        resize_method="by_factor",
        resizing_factor_x=1.0,
        resizing_factor_y=1.0,
        resizing_factor_z=1.0,
        interpolation="bilinear",
    )

    assert image_payload_metadata(resized).unit_interval_intensity_scale is None


def test_cellprofiler_backend_provider_rejects_raw_strings() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    raw_provider = CellProfilerBackendProvider.NUMBA.value

    with pytest.raises(TypeError, match="CellProfilerBackendProvider"):
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=raw_provider,
        )


def test_numba_threshold_smoothing_kernel_matches_centrosome_public_primitive() -> None:
    import centrosome.smooth
    import numpy as np

    from openhcs.processing.backends.cellprofiler.thresholding import (
        _threshold_smoothing_kernel,
    )

    sigma, kernel = _threshold_smoothing_kernel(1.3488, None)
    expected = centrosome.smooth.circular_gaussian_kernel(
        sigma,
        int(np.ceil(sigma * 4.0)),
    )

    np.testing.assert_allclose(kernel, expected, rtol=0.0, atol=1e-15)


def test_threshold_application_smoothing_is_mask_normalized() -> None:
    import centrosome.smooth
    import scipy.ndimage as ndi

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdApplicationSmoothing,
    )

    image = np.linspace(0.0, 1.0, 49, dtype=np.float64).reshape(7, 7)
    mask = np.ones((7, 7), dtype=bool)
    mask[:2, :] = False
    mask[:, :1] = False

    smoothed, sigma = ThresholdApplicationSmoothing(1.3488).smooth(image, mask)
    expected = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda array: ndi.gaussian_filter(
            array,
            sigma=1.0,
            mode="constant",
            cval=0,
            truncate=4.0,
        ),
        mask,
    )

    assert sigma == pytest.approx(1.0)
    np.testing.assert_allclose(smoothed, expected, rtol=0.0, atol=5e-15)


def test_threshold_application_smoothing_matches_scipy_unmasked() -> None:
    import scipy.ndimage as ndi

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdApplicationSmoothing,
    )

    image = np.linspace(0.0, 1.0, 121, dtype=np.float64).reshape(11, 11)

    smoothed, sigma = ThresholdApplicationSmoothing(1.3488).smooth(image, None)
    expected = ndi.gaussian_filter(
        image,
        sigma=1.0,
        mode="constant",
        cval=0,
        truncate=4.0,
    )
    expected /= ndi.gaussian_filter(
        np.ones(image.shape, dtype=np.float64),
        sigma=1.0,
        mode="constant",
        cval=0,
        truncate=4.0,
    )

    assert sigma == pytest.approx(1.0)
    np.testing.assert_allclose(smoothed, expected, rtol=0.0, atol=5e-15)


def test_numba_threshold_primitives_are_default_and_roundtrip_log_transform() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumbaNumpyThresholdPrimitiveBackendStrategy,
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.0, 0.2], [0.5, 1.0]], dtype=np.float32)

    transformed, conversion = strategy.log_transform(image)
    restored = strategy.inverse_log_transform(transformed, conversion)

    assert type(strategy) is NumbaNumpyThresholdPrimitiveBackendStrategy
    assert transformed.dtype == np.float64
    np.testing.assert_allclose(
        restored[image > 0],
        image[image > 0],
        rtol=0.0,
        atol=2e-7,
    )


def test_numba_log_transform_uses_dynamic_range_floor() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.0, 1.0 / 255.0], [0.5, 1.0]], dtype=np.float32)

    transformed, conversion = strategy.log_transform(image)
    restored_cut = strategy.inverse_log_transform(100.5 / 128.0, conversion)

    assert transformed[0, 1] > 0.0
    assert restored_cut == pytest.approx(0.3038138449192047, abs=1e-12)


def test_numba_threshold_primitives_cover_otsu_and_multiotsu() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    values = np.concatenate(
        (
            np.linspace(0.05, 0.20, 30),
            np.linspace(0.45, 0.60, 30),
            np.linspace(0.80, 0.95, 30),
        )
    ).astype(np.float64)
    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)

    otsu = strategy.otsu_threshold(values)
    multiotsu = strategy.multiotsu_thresholds(values, nbins=128)
    mce = strategy.minimum_cross_entropy_threshold(values.reshape(9, 10))

    assert 0.19 < otsu < 0.60
    assert multiotsu.shape == (2,)
    assert 0.20 < multiotsu[0] < 0.60
    assert 0.60 < multiotsu[1] < 0.90
    assert 0.19 < mce < 0.60


def test_numba_multiotsu_matches_public_tie_behavior() -> None:
    from skimage.filters import threshold_multiotsu

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    cases = (
        (np.repeat([0, 1, 2, 3], [10, 10, 10, 10]).astype(np.float64), 128),
        (np.repeat([0, 1, 2, 3, 4], [8, 8, 8, 8, 8]).astype(np.float64), 128),
        (
            np.repeat([0, 0.1, 0.2, 0.5, 0.9], [10, 20, 30, 20, 10]).astype(np.float64),
            5,
        ),
    )

    for values, nbins in cases:
        observed = strategy.multiotsu_thresholds(values, nbins=nbins)
        expected = threshold_multiotsu(values, classes=3, nbins=nbins)

        assert observed == pytest.approx(expected)


def test_numba_threshold_primitives_cover_histogram_methods() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    values = np.concatenate(
        (
            np.full(100, 0.25, dtype=np.float64),
            np.linspace(0.30, 0.35, 50, dtype=np.float64),
            np.linspace(0.70, 0.75, 50, dtype=np.float64),
            np.full(100, 0.80, dtype=np.float64),
        )
    )

    li = strategy.li_threshold(values)
    triangle = strategy.triangle_threshold(values)
    isodata = strategy.isodata_threshold(values)
    mean = strategy.mean_threshold(values)
    yen = strategy.yen_threshold(values)
    minimum = strategy.minimum_threshold(
        np.concatenate(
            (
                np.full(100, 0.30, dtype=np.float64),
                np.full(100, 0.70, dtype=np.float64),
            )
        )
    )
    sauvola = strategy.sauvola_threshold_image(values.reshape(20, 15), window_size=5)

    assert 0.25 <= li <= 0.80
    assert 0.25 <= triangle <= 0.80
    assert 0.25 <= isodata <= 0.80
    assert mean == pytest.approx(np.mean(values))
    assert 0.25 <= yen <= 0.80
    assert 0.25 <= minimum <= 0.80
    assert sauvola.shape == (20, 15)
    assert np.all(np.isfinite(sauvola))


def test_numba_minimum_cross_entropy_uses_li_positive_shift_semantics() -> None:
    from skimage.filters import threshold_li

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    values = np.array(
        [0.10, 0.11, 0.12, 0.30, 0.45, 0.46, 0.47, 0.90],
        dtype=np.float64,
    )
    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    expected = threshold_li(values)

    assert strategy.li_threshold(values) == pytest.approx(expected)
    assert strategy.minimum_cross_entropy_threshold(values.reshape(2, 4)) == (
        pytest.approx(expected)
    )

    values32 = np.array(
        [
            2.8775458e-04,
            1.6496025e-04,
            1.5679000e-04,
            2.0272080e-04,
            5.3595490e-05,
            1.7344829e-04,
            1.8645125e-04,
            1.7703290e-04,
            5.2483805e-04,
            9.2633464e-04,
            7.6701742e-04,
            1.2272450e-03,
            7.2445755e-04,
            9.1529643e-04,
            9.3011564e-04,
            1.1702714e-03,
        ],
        dtype=np.float32,
    )
    expected32 = threshold_li(values32.copy())

    assert strategy.li_threshold(values32) == pytest.approx(float(expected32))
    assert strategy.minimum_cross_entropy_threshold(values32.reshape(4, 4)) == (
        pytest.approx(float(expected32))
    )


def test_numba_minimum_cross_entropy_uses_quantized_scale_without_semantic_drift() -> (
    None
):
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    codes = np.array(
        [
            [0, 2, 4, 8, 16, 64],
            [1, 3, 5, 9, 32, 128],
            [0, 2, 4, 8, 16, 64],
            [1, 3, 5, 9, 32, 255],
        ],
        dtype=np.float32,
    )
    image = codes / np.float32(255)
    mask = np.ones(image.shape, dtype=bool)
    mask[:, 0] = False

    dense_full = strategy.minimum_cross_entropy_threshold(image)
    quantized_full = strategy.minimum_cross_entropy_threshold(
        image,
        proven_unit_interval_scale=255,
    )
    dense_masked = strategy.minimum_cross_entropy_threshold(image, mask=mask)
    quantized_masked = strategy.minimum_cross_entropy_threshold(
        image,
        mask=mask,
        proven_unit_interval_scale=255,
    )

    assert quantized_full == pytest.approx(dense_full, abs=1e-7)
    assert quantized_masked == pytest.approx(dense_masked, abs=1e-7)


def test_measure_colocalization_object_thresholds_use_pixel_dtype_boundary() -> None:
    from openhcs.processing.backends.cellprofiler.colocalization import (
        measure_colocalization_objects,
    )

    scale = np.float32(65535)
    image = (
        np.asarray(
            [
                [[1000, 1000, 1000]],
                [[105, 700, 700]],
            ],
            dtype=np.float32,
        )
        / scale
    )
    labels = np.asarray([[1, 1, 1]], dtype=np.int32)

    _, rows = measure_colocalization_objects(
        image,
        labels,
        channel_1=0,
        channel_2=1,
        threshold_percent=15.0,
        do_correlation=False,
        do_manders=True,
        do_rwc=False,
        do_overlap=True,
        do_costes=False,
    )
    row = next(iter(rows))

    assert row.manders_m1 == pytest.approx(2.0 / 3.0)


def test_measure_colocalization_object_correlation_uses_cellprofiler_two_pass_formula() -> (
    None
):
    from openhcs.processing.backends.cellprofiler.colocalization import (
        measure_colocalization_objects,
    )

    first = np.asarray([[0.001, 0.003, 0.008, 0.013, 0.021]], dtype=np.float32)
    second = np.asarray([[0.002, 0.005, 0.007, 0.017, 0.019]], dtype=np.float32)
    image = np.stack((first, second), axis=0)
    labels = np.ones(first.shape, dtype=np.int32)

    _, rows = measure_colocalization_objects(
        image,
        labels,
        channel_1=0,
        channel_2=1,
        threshold_percent=15.0,
        do_correlation=True,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        do_costes=False,
    )
    row = next(iter(rows))

    first_pixels = np.asarray(first.ravel(), dtype=float)
    second_pixels = np.asarray(second.ravel(), dtype=float)
    first_delta = first_pixels - np.mean(first_pixels)
    second_delta = second_pixels - np.mean(second_pixels)
    expected = np.sum(
        first_delta
        * second_delta
        / (
            np.sqrt(np.sum(first_delta * first_delta))
            * np.sqrt(np.sum(second_delta * second_delta))
        )
    )
    assert row.correlation == pytest.approx(float(expected), abs=1e-12)


def test_cellprofiler_backend_selection_does_not_silently_fallback() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        ObjectIntensityBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity_distribution import (
        RadialDistributionBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NeighborTopologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeMeasurementBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
        ThresholdSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        ShapeZernikeBackendStrategy,
    )

    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ThresholdSmoothingBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ThresholdPrimitiveBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ObjectIntensityBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        RadialDistributionBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        NeighborTopologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ShapeZernikeBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )


def test_object_intensity_inner_edges_match_cellprofiler_boundary_semantics() -> None:
    from skimage.segmentation import find_boundaries
    from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        object_intensity_backend,
    )

    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[:3, :3] = 1

    arrays = object_intensity_backend().measure(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    edge_mask = find_boundaries(labels, mode="inner") & (labels == 1)

    assert arrays.integrated_intensity_edge[0] == pytest.approx(
        float(np.sum(image[edge_mask]))
    )
    assert arrays.mean_intensity_edge[0] == pytest.approx(
        float(np.mean(image[edge_mask]))
    )


def test_object_intensity_3d_batch_matches_single_image_backend() -> None:
    from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
    from openhcs.core.runtime_object_labels import (
        ObjectLabelPayload,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        ObjectIntensityMeasurementRows,
        ObjectIntensityPreparedLabels,
        object_intensity_backend,
    )

    rng = np.random.default_rng(42)
    image_a = rng.random((4, 6, 5), dtype=np.float32)
    image_b = rng.random((4, 6, 5), dtype=np.float32)
    image_b[1, 2, 3] = np.nan
    labels = np.zeros((4, 6, 5), dtype=np.int32)
    labels[:, 1:4, 1:3] = 1
    labels[1:4, 3:5, 2:5] = 2

    prepared = ObjectIntensityPreparedLabels.from_source(
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        ),
        labels,
    )
    backend = object_intensity_backend()

    singles = tuple(
        backend.measure_prepared(image, prepared) for image in (image_a, image_b)
    )
    batch = backend.measure_prepared_batch((image_a, image_b), prepared)

    assert tuple(
        list(ObjectIntensityMeasurementRows.from_arrays(arrays, slice_index=index))
        for index, arrays in enumerate(singles)
    ) == tuple(
        list(ObjectIntensityMeasurementRows.from_arrays(arrays, slice_index=index))
        for index, arrays in enumerate(batch)
    )


def test_numba_declumping_smoothing_matches_native_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    rng = np.random.default_rng(11)
    image = rng.random((21, 24), dtype=np.float32)
    mask = rng.random((21, 24)) > 0.2
    mask[:2, :3] = False
    filter_size = 6.5

    native = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    ).smooth_image_for_declumping(image, mask, filter_size)
    numba = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).smooth_image_for_declumping(image, mask, filter_size)

    np.testing.assert_allclose(numba, native, rtol=0.0, atol=1e-7)
    assert numba.dtype == native.dtype


def test_numba_seed_shrink_matches_centrosome_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    mask = np.zeros((24, 28), dtype=bool)
    mask[2:8, 3:10] = True
    mask[10:16, 12:20] = np.eye(6, 8, dtype=bool)
    mask[17:20, 21:25] = True
    mask[19, 20] = True

    numba = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).shrink_components_to_seed_points(mask)
    centrosome = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    ).shrink_components_to_seed_points(mask)

    np.testing.assert_array_equal(numba, centrosome)


def test_legacy_fast_shape_zernike_backend_matches_native_reference_values() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 2
    labels[11, 13] = 0
    measured_labels = np.array([1, 2], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    expected_indexes = (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
        (5, 3),
        (5, 5),
    )
    expected_values = np.array(
        [
            [
                8.7665673571929248e-01,
                5.1817387788033623e-18,
                8.1438057416546292e-02,
                6.2276161553829500e-02,
                9.5546542074512310e-18,
                1.1816182429749623e-17,
                1.4842616057596269e-02,
                1.3271968855734138e-02,
                1.0279885936186994e-01,
                2.3173440302102131e-18,
                6.9520320906306388e-18,
                3.4760160453153194e-18,
            ],
            [
                9.5492965855137191e-01,
                1.7669748230352868e-17,
                3.9788735772973635e-02,
                6.6261555863823256e-18,
                4.4174370575882171e-18,
                6.2471993977707561e-18,
                1.4920775914865186e-02,
                0.0,
                1.6164173907770615e-01,
                4.4174370575882171e-18,
                8.8348741151764342e-18,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    assert legacy_fast_indexes == expected_indexes
    np.testing.assert_allclose(
        legacy_fast_values,
        expected_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_shape_zernike_backend_uses_declared_measured_label_domain() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 3
    measured_labels = np.array([1, 3], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    expected_indexes = (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
        (5, 3),
        (5, 5),
    )
    expected_values = np.array(
        [
            [
                8.7665673571929248e-01,
                5.1817387788033623e-18,
                8.1438057416546292e-02,
                6.2276161553829500e-02,
                9.5546542074512310e-18,
                1.1816182429749623e-17,
                1.4842616057596269e-02,
                1.3271968855734138e-02,
                1.0279885936186994e-01,
                2.3173440302102131e-18,
                6.9520320906306388e-18,
                3.4760160453153194e-18,
            ],
            [
                9.9471839432434572e-01,
                1.7669748230352868e-17,
                2.0320210464905799e-16,
                6.6261555863823256e-18,
                4.4174370575882171e-18,
                6.2471993977707561e-18,
                2.4867959858108645e-02,
                0.0,
                1.6164173907770615e-01,
                4.4174370575882171e-18,
                8.8348741151764342e-18,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    assert legacy_fast_indexes == expected_indexes
    assert (
        legacy_fast_values.shape
        == expected_values.shape
        == (
            measured_labels.size,
            len(legacy_fast_indexes),
        )
    )
    np.testing.assert_allclose(
        legacy_fast_values,
        expected_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_grouped_minimum_position_matches_numpy_124_quicksort_ties() -> None:
    from openhcs.processing.backends.cellprofiler.label_geometry import (
        _grouped_minimum_positions,
    )

    values = np.zeros(20, dtype=np.float64)
    labels = np.arange(20, dtype=np.int32) % 2 + 1

    np.testing.assert_array_equal(
        _grouped_minimum_positions(
            values,
            labels,
            np.asarray((1, 2), dtype=np.int32),
        ),
        np.asarray((0, 17), dtype=int),
    )


def test_numpy_124_arccos_matches_cp_angle_order_bits() -> None:
    from openhcs.processing.backends.cellprofiler.label_geometry import (
        _numpy_124_arccos,
    )

    inputs = np.asarray(
        (
            float.fromhex("0x1.aa027f059dce0p-1"),
            float.fromhex("0x1.3f694e43bb8dcp-3"),
            float.fromhex("-0x1.1c01aa03be896p-1"),
        ),
        dtype=np.float64,
    )
    expected = np.asarray(
        (
            float.fromhex("0x1.2d0ead6066397p-1"),
            float.fromhex("0x1.6a08a5c0218bbp+0"),
            float.fromhex("0x1.145385fa3af71p+1"),
        ),
        dtype=np.float64,
    )

    np.testing.assert_array_equal(_numpy_124_arccos(inputs), expected)


def test_legacy_fast_shape_zernike_backend_zeros_pixels_outside_unit_circle() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[2, 2] = 0
    measured_labels = np.array([1], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    expected_indexes = (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
        (5, 3),
        (5, 5),
    )
    expected_values = np.array(
        [
            [
                8.4882636315677540e-01,
                0.0,
                1.5719006725125445e-01,
                0.0,
                8.0949153702017381e-18,
                3.9266107178561947e-18,
                1.5719006725125440e-01,
                0.0,
                4.5410463872584711e-02,
                0.0,
                0.0,
                1.4724790191960730e-18,
            ]
        ],
        dtype=np.float64,
    )

    assert legacy_fast_indexes == expected_indexes
    np.testing.assert_allclose(
        legacy_fast_values,
        expected_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_legacy_fast_shape_zernike_backend_matches_cp_boundary_tie_reference() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.array(
        [
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        np.array([1], dtype=np.int32),
        max_order=9,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    expected_first_ten = np.array(
        [
            9.836620553573459e-01,
            9.587285264010693e-03,
            1.527408829685409e-02,
            2.2072774040876608e-02,
            7.816253864125930e-03,
            3.954710099285566e-02,
            1.3732495466202541e-02,
            1.953523677243512e-02,
            8.122532660014267e-03,
            8.319151920377181e-03,
        ],
        dtype=np.float64,
    )

    assert legacy_fast_indexes[:10] == (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
    )
    np.testing.assert_allclose(
        legacy_fast_values[0, :10],
        expected_first_ten,
        atol=1e-12,
        rtol=1e-12,
    )


def test_legacy_fast_shape_zernike_backend_canonicalizes_cocircular_supports() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    object_mask = np.asarray(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    labels = np.zeros((328, 443), dtype=np.int32)
    labels[315:328, 432:443] = object_mask

    indexes, values = shape_zernike_moments(
        labels,
        np.asarray((1,), dtype=np.int32),
        max_order=9,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    expected_values = np.asarray(
        (
            8.5350954309143068e-01,
            8.2787164828840007e-02,
            7.9137666447560609e-02,
            8.6404309193858914e-02,
            2.0767945028413601e-02,
            2.2987453291421895e-02,
            7.7330738594692180e-03,
            2.7128754063428902e-02,
            1.2504326831993696e-02,
            1.1688212302389587e-02,
            2.8945825608166960e-02,
            2.0026230892118110e-02,
            2.0546799436526764e-02,
            8.1263836836855450e-03,
            4.9325301154274010e-03,
            9.3351032990203700e-03,
            4.3518682690635880e-03,
            1.5112238056028326e-02,
            1.7055783067288128e-02,
            2.4638103342696834e-02,
            1.7888863576524142e-02,
            1.0009308659453403e-02,
            1.1931120455247695e-02,
            5.4041116768740380e-03,
            8.4040112105264980e-03,
            9.1818310636040440e-03,
            4.0842576061681543e-04,
            7.2361990858590830e-03,
            2.6289597271375614e-02,
            1.2174707353807023e-02,
        ),
        dtype=np.float64,
    )

    assert indexes == (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
        (5, 3),
        (5, 5),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
        (7, 1),
        (7, 3),
        (7, 5),
        (7, 7),
        (8, 0),
        (8, 2),
        (8, 4),
        (8, 6),
        (8, 8),
        (9, 1),
        (9, 3),
        (9, 5),
        (9, 7),
        (9, 9),
    )
    np.testing.assert_allclose(
        values[0],
        expected_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_measure_object_size_shape_callable_defaults_are_declared_on_module() -> None:
    import inspect

    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape,
        measure_object_size_shape_feature_arrays,
    )

    assert (
        MeasureObjectSizeShapeModule.zernike_backend_provider
        is CellProfilerBackendProvider.LEGACY_FAST
    )
    assert (
        MeasureObjectSizeShapeModule.regionprops_backend_provider
        is AnalysisBackendProvider.NUMBA
    )
    assert (
        inspect.signature(measure_object_size_shape)
        .parameters["zernike_backend_provider"]
        .default
        is CellProfilerBackendProvider.LEGACY_FAST
    )
    assert (
        inspect.signature(measure_object_size_shape_feature_arrays)
        .parameters["zernike_backend_provider"]
        .default
        is CellProfilerBackendProvider.LEGACY_FAST
    )
    assert (
        inspect.signature(measure_object_size_shape)
        .parameters["regionprops_backend_provider"]
        .default
        is AnalysisBackendProvider.NUMBA
    )
    assert (
        inspect.signature(measure_object_size_shape_feature_arrays)
        .parameters["regionprops_backend_provider"]
        .default
        is AnalysisBackendProvider.NUMBA
    )


def test_measure_object_size_shape_orientation_is_cpu_dispatch_independent() -> (
    None
):
    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape_feature_arrays,
    )

    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 2:6] = np.array(
        (
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (0, 1, 1, 0),
        ),
        dtype=np.int32,
    )

    orientations = tuple(
        measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=False,
            calculate_zernikes=False,
            regionprops_backend_provider=provider,
        )[0][MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.value]
        for provider in (
            AnalysisBackendProvider.NUMBA,
            AnalysisBackendProvider.SKIMAGE,
        )
    )

    np.testing.assert_allclose(orientations[0], orientations[1], rtol=0.0, atol=1e-12)
    assert orientations[1][0] == pytest.approx(-45.0, abs=1e-12)


def test_measure_object_size_shape_orientation_uses_explicit_second_moments() -> (
    None
):
    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape_feature_arrays,
    )

    labels = np.asarray(
        (
            (0, 1, 1, 0),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (0, 1, 1, 1),
        ),
        dtype=np.int32,
    )

    orientations = tuple(
        measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=False,
            calculate_zernikes=False,
            regionprops_backend_provider=provider,
        )[0][MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.value]
        for provider in (
            AnalysisBackendProvider.NUMBA,
            AnalysisBackendProvider.SKIMAGE,
        )
    )

    np.testing.assert_allclose(orientations[0], orientations[1], rtol=0.0, atol=1e-12)
    assert orientations[1][0] == pytest.approx(45.0, abs=1e-12)


def test_measure_object_size_shape_preserves_vertical_orientation_representatives() -> (
    None
):
    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape_feature_arrays,
    )

    labels = np.zeros((8, 16), dtype=np.int32)
    labels[1:3, 1:8] = np.array(
        (
            (1, 1, 1, 1, 1, 1, 1),
            (0, 1, 1, 1, 1, 1, 0),
        ),
        dtype=np.int32,
    )
    labels[4:7, 10:14] = np.array(
        (
            (0, 2, 2, 0),
            (2, 2, 2, 2),
            (0, 2, 2, 0),
        ),
        dtype=np.int32,
    )

    orientations = tuple(
        measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=False,
            calculate_zernikes=False,
            regionprops_backend_provider=provider,
        )[0][MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.value]
        for provider in (
            AnalysisBackendProvider.NUMBA,
            AnalysisBackendProvider.SKIMAGE,
        )
    )

    np.testing.assert_array_equal(
        orientations[0],
        np.array((-90.0, 90.0)),
    )
    np.testing.assert_array_equal(orientations[0], orientations[1])
    assert orientations[0].mean() == 0.0


@pytest.mark.parametrize(
    ("mask", "expected_orientation"),
    (
        (
            (
                (0, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (0, 1, 1, 0),
            ),
            -44.99999999999998,
        ),
        (
            (
                (0, 0, 0, 0, 1),
                (0, 0, 0, 1, 0),
                (0, 0, 1, 0, 0),
                (0, 1, 0, 0, 0),
                (1, 0, 0, 0, 0),
            ),
            45.0,
        ),
        (
            (
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 0),
            ),
            45.0,
        ),
    ),
    ids=("tumor-object-26", "brightfield-object-24", "brightfield-object-67"),
)
def test_measure_object_size_shape_orientation_preserves_cp4281_tie_geometries(
    mask: tuple[tuple[int, ...], ...],
    expected_orientation: float,
) -> None:
    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape_feature_arrays,
    )

    labels = np.asarray(mask, dtype=np.int32)
    orientations = tuple(
        measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=False,
            calculate_zernikes=False,
            regionprops_backend_provider=provider,
        )[0][MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.value]
        for provider in (
            AnalysisBackendProvider.NUMBA,
            AnalysisBackendProvider.SKIMAGE,
        )
    )

    np.testing.assert_array_equal(orientations[0], orientations[1])
    assert orientations[0][0] == pytest.approx(expected_orientation, abs=1e-12)


@pytest.mark.parametrize(
    ("mask", "expected_orientation"),
    (
        (
            (
                (0, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
                (0, 1, 1, 1, 0),
            ),
            -44.999999999999964,
        ),
        (
            (
                (0, 1, 0, 0, 0),
                (1, 1, 1, 1, 0),
                (0, 1, 1, 1, 1),
                (0, 1, 1, 1, 1),
                (0, 0, 1, 1, 0),
            ),
            45.00000000000001,
        ),
        (
            (
                (0, 1, 1, 0),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 0),
            ),
            -44.99999999999998,
        ),
        (
            (
                (1, 1, 1),
                (0, 1, 1),
                (0, 0, 1),
            ),
            -45.0,
        ),
    ),
    ids=(
        "yeast-colonies",
        "advanced-mitochondria",
        "advanced-nucleoli",
        "beginner-nucleoli",
    ),
)
def test_measure_object_size_shape_orientation_matches_exact_case_matrix(
    mask: tuple[tuple[int, ...], ...],
    expected_orientation: float,
) -> None:
    from openhcs.processing.backends.analysis.region_properties import (
        AnalysisBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
        measure_object_size_shape_feature_arrays,
    )

    labels = np.asarray(mask, dtype=np.int32)
    expected = np.asarray((expected_orientation,), dtype=np.float64)
    for provider in (
        AnalysisBackendProvider.NUMBA,
        AnalysisBackendProvider.SKIMAGE,
    ):
        feature_values, _measured_labels = measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=False,
            calculate_zernikes=False,
            regionprops_backend_provider=provider,
        )

        np.testing.assert_array_equal(
            feature_values[
                MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.value
            ],
            expected,
        )


def test_legacy_fast_intensity_zernike_backend_matches_centrosome_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        intensity_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 2
    labels[11, 13] = 0
    image = np.linspace(0.0, 1.0, labels.size, dtype=np.float64).reshape(labels.shape)
    measured_labels = np.array([1, 2], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_magnitudes, legacy_fast_phases = (
        intensity_zernike_moments(
            image,
            labels,
            measured_labels,
            max_order=5,
            backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        )
    )
    expected_indexes = (
        (0, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, 1),
        (5, 3),
        (5, 5),
    )
    expected_magnitudes = np.array(
        [
            [
                3.0313588850174222e-01,
                4.6978525057005845e-02,
                2.8160164505626334e-02,
                2.1534243445478957e-02,
                1.0291546017873580e-02,
                2.1351940583848830e-02,
                5.1323732801961414e-03,
                4.5892649965774759e-03,
                3.5546437162839790e-02,
                3.5442769227822337e-03,
                1.0492107771464218e-02,
                6.7019059354025646e-03,
            ],
            [
                7.3519163763066209e-01,
                4.6267235714220228e-02,
                3.0632984901277447e-02,
                1.2455695750587118e-17,
                1.1566808928554946e-03,
                1.5036851607121574e-02,
                1.1487369337979092e-02,
                6.5420519111823968e-18,
                1.2444650116144018e-01,
                5.4942342410636167e-03,
                1.5759777165156256e-02,
                1.5181436718728502e-02,
            ],
        ],
        dtype=np.float64,
    )
    expected_phases = np.array(
        [
            [
                1.5707963267948966,
                1.4947527675157795,
                -1.5707963267948966,
                -1.5707963267948966,
                -1.5878522420194257,
                -1.5516140276398370,
                -1.5707963267948966,
                -1.5707963267948966,
                -1.5707963267948966,
                1.5460903029579192,
                -1.5198993446615918,
                -1.7406165821130795,
            ],
            [
                1.5707963267948966,
                1.5152978215491800,
                1.5707963267948966,
                -1.1902899496825317,
                1.5152978215491757,
                -1.5152978215491801,
                -1.5707963267948966,
                2.3561944901923450,
                -1.5707963267948966,
                1.5152978215491790,
                -1.5152978215491801,
                -1.6262948320406134,
            ],
        ],
        dtype=np.float64,
    )

    assert legacy_fast_indexes == expected_indexes
    np.testing.assert_allclose(
        legacy_fast_magnitudes,
        expected_magnitudes,
        atol=1e-12,
        rtol=1e-12,
    )
    meaningful_phase = expected_magnitudes > 1e-12
    np.testing.assert_allclose(
        legacy_fast_phases[meaningful_phase],
        expected_phases[meaningful_phase],
        atol=1e-12,
        rtol=1e-12,
    )


def test_native_intensity_zernike_matches_cellprofiler_4281_source_loop() -> None:
    import centrosome.zernike
    import scipy.ndimage

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        _construct_cellprofiler_4281_zernike_polynomials,
        intensity_zernike_moments,
    )
    from openhcs.processing.backends.cellprofiler.label_geometry import (
        minimum_enclosing_circle_from_labels,
    )

    labels = np.zeros((14, 16), dtype=np.int32)
    labels[1:8, 2:9] = 1
    labels[4:7, 7:10] = 0
    labels[8:13, 9:15] = 2
    labels[10, 11] = 0
    image = np.linspace(0.0, 1.0, labels.size, dtype=np.float64).reshape(labels.shape)
    image_mask = np.ones(labels.shape, dtype=bool)
    image_mask[1:3, 2:5] = False
    image_mask[11:, 13:] = False
    measured_labels = np.array([1, 2], dtype=np.int32)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(6)
    centers, radii = minimum_enclosing_circle_from_labels(
        labels,
        measured_labels,
    )
    y_coords, x_coords = np.nonzero(labels)
    object_labels = labels[y_coords, x_coords]
    normalized_yx = (
        np.column_stack((y_coords, x_coords)) - centers[object_labels - 1]
    ) / radii[object_labels - 1, np.newaxis]
    polynomials = _construct_cellprofiler_4281_zernike_polynomials(
        normalized_yx[:, 1],
        normalized_yx[:, 0],
        zernike_indexes,
    )
    selected = image_mask[y_coords, x_coords]
    y_coords = y_coords[selected]
    x_coords = x_coords[selected]
    object_labels = object_labels[selected]
    polynomials = polynomials[selected]
    areas = scipy.ndimage.sum(
        np.ones(object_labels.shape, dtype=int),
        labels=object_labels,
        index=measured_labels,
    )
    expected_magnitudes = []
    expected_phases = []
    for zernike_index in range(len(zernike_indexes)):
        real = scipy.ndimage.sum(
            image[y_coords, x_coords] * polynomials[:, zernike_index].real,
            labels=object_labels,
            index=measured_labels,
        )
        imaginary = scipy.ndimage.sum(
            image[y_coords, x_coords] * polynomials[:, zernike_index].imag,
            labels=object_labels,
            index=measured_labels,
        )
        expected_magnitudes.append(np.sqrt(real * real + imaginary * imaginary) / areas)
        expected_phases.append(np.arctan2(real, imaginary))

    actual_indexes, actual_magnitudes, actual_phases = intensity_zernike_moments(
        image,
        labels,
        measured_labels,
        image_mask=image_mask,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )

    assert actual_indexes == tuple((int(n), int(m)) for n, m in zernike_indexes)
    np.testing.assert_array_equal(
        actual_magnitudes,
        np.column_stack(expected_magnitudes),
    )
    np.testing.assert_array_equal(
        actual_phases,
        np.column_stack(expected_phases),
    )


def test_cellprofiler_4281_zernike_complex_square_is_version_stable() -> None:
    from openhcs.processing.backends.cellprofiler.zernike import (
        _construct_cellprofiler_4281_zernike_polynomials,
    )

    actual = _construct_cellprofiler_4281_zernike_polynomials(
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
        np.array([0.4, 0.5, -0.6], dtype=np.float64),
        np.array(((2, 2), (4, 2)), dtype=np.int32),
    )
    expected = np.array(
        [
            [
                complex(float.fromhex("0x1.3333333333334p-3"), float.fromhex("0x1.47ae147ae147cp-4")),
                complex(float.fromhex("-0x1.645a1cac08313p-2"), float.fromhex("-0x1.7c1bda5119ce1p-3")),
            ],
            [
                complex(float.fromhex("0x1.ae147ae147ae1p-3"), float.fromhex("-0x1.999999999999ap-3")),
                complex(float.fromhex("-0x1.8bac710cb295ep-2"), float.fromhex("0x1.78d4fdf3b645ap-2")),
            ],
            [
                complex(float.fromhex("0x1.147ae147ae148p-2"), float.fromhex("-0x1.70a3d70a3d70ap-2")),
                complex(float.fromhex("-0x1.4bc6a7ef9db24p-2"), float.fromhex("0x1.ba5e353f7cedap-2")),
            ],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_array_equal(actual, expected)


def test_native_intensity_zernike_preserves_measured_object_order() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        intensity_zernike_moments,
    )

    labels = np.zeros((12, 14), dtype=np.int32)
    labels[1:6, 2:8] = 2
    labels[7:11, 8:13] = 7
    image = np.linspace(0.0, 1.0, labels.size, dtype=np.float64).reshape(labels.shape)
    measured_labels = np.array([7, 2], dtype=np.int32)

    indexes, magnitudes, phases = intensity_zernike_moments(
        image,
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    single_object_rows = tuple(
        intensity_zernike_moments(
            image,
            labels,
            np.array([object_id], dtype=np.int32),
            max_order=5,
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
        for object_id in measured_labels
    )

    assert all(row_indexes == indexes for row_indexes, _, _ in single_object_rows)
    np.testing.assert_array_equal(
        magnitudes,
        np.vstack(
            tuple(row_magnitudes[0] for _, row_magnitudes, _ in single_object_rows)
        ),
    )
    np.testing.assert_array_equal(
        phases,
        np.vstack(tuple(row_phases[0] for _, _, row_phases in single_object_rows)),
    )


def test_zernike_label_geometry_cache_reuses_equal_label_values() -> None:
    from openhcs.processing.backends.cellprofiler import zernike

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[7:10, 7:11] = 2
    object_ids = np.array([1, 2], dtype=np.int32)
    zernike._ZERNIKE_LABEL_GEOMETRY_CACHE.clear()
    strategy = zernike.LegacyFastNumpyShapeZernikeBackendStrategy()

    first = strategy.zernike_label_geometry(labels, object_ids)
    second = strategy.zernike_label_geometry(labels.copy(), object_ids.copy())

    assert second is first
    assert len(zernike._ZERNIKE_LABEL_GEOMETRY_CACHE) == 1
    zernike._ZERNIKE_LABEL_GEOMETRY_CACHE.clear()


def test_zernike_numba_provider_is_not_registered_until_pure() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        ShapeZernikeBackendStrategy,
    )

    with pytest.raises(NotImplementedError, match="provider 'numba'"):
        ShapeZernikeBackendStrategy.for_memory_type(
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
