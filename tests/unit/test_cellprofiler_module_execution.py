import numpy as np

from benchmark.cellprofiler_compat.module_execution import (
    CellProfilerAlignedImageStack,
    CellProfilerFunctionContractMetadata,
    CellProfilerFunctionContractExecutor,
    CellProfilerImageExecutionMode,
    _coerce_invocation_kwargs,
    _compose_image_payload,
    _measurement_image_for_labels,
    _measurement_labels,
    _measurement_labels_for_image,
    _measurement_table_rows,
    _object_only_reference_image,
)
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterMethod,
    FilterMode,
    filter_objects,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_values import MeasurementTable
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def test_coerce_invocation_kwargs_uses_function_enum_annotations() -> None:
    coerced = _coerce_invocation_kwargs(
        identify_primary_objects,
        {
            "unclump_method": "Shape",
            "fill_holes": "After both thresholding and declumping",
            "limit_erase": "Continue",
        },
    )

    assert coerced["unclump_method"] is UnclumpMethod.SHAPE
    assert coerced["fill_holes"] is FillHolesOption.AFTER_BOTH
    assert coerced["limit_erase"] is ExcessObjectHandling.CONTINUE


def test_cellprofiler_contract_executor_applies_pure_2d_after_input_resolution():
    calls = []

    def add_one(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image + 1

    add_one.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(add_one, stack, {})

    assert calls == [(4, 5), (4, 5)]
    assert result.shape == stack.shape
    np.testing.assert_array_equal(result, np.ones_like(stack))


def test_cellprofiler_contract_executor_slices_aligned_runtime_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)
    labels = np.ones_like(stack, dtype=np.int32)

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        stack,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert result_image.shape == stack.shape
    assert result_labels.shape == labels.shape


def test_cellprofiler_contract_executor_preserves_multi_image_stack_payload():
    calls = []

    def keep_stack(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image

    keep_stack.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((3, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        keep_stack,
        stack,
        {},
        force_full_stack=True,
    )

    assert calls == [(3, 4, 5)]
    assert result.shape == stack.shape


def test_compose_image_payload_aligns_multislice_inputs_with_broadcast():
    raw_stack = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    illumination = np.full((4, 5), 3, dtype=np.float32)

    composition = _compose_image_payload(
        "CorrectIlluminationApply",
        (raw_stack, illumination),
    )

    assert composition.execution_mode is CellProfilerImageExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    assert isinstance(composition.payload, CellProfilerAlignedImageStack)
    assert len(composition.payload.slices) == 2
    for slice_index, composed_slice in enumerate(composition.payload.slices):
        assert composed_slice.shape == (2, 4, 5)
        np.testing.assert_array_equal(composed_slice[0], raw_stack[slice_index])
        np.testing.assert_array_equal(composed_slice[1], illumination)


def test_cellprofiler_contract_executor_applies_aligned_multi_image_stack():
    calls = []

    def subtract_illumination(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return (image[0] - image[1])[np.newaxis, ...]

    aligned_stack = CellProfilerAlignedImageStack(
        slices=(
            np.stack(
                (
                    np.full((4, 5), 11, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full((4, 5), 22, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        subtract_illumination,
        aligned_stack,
        {},
        execution_mode=CellProfilerImageExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert calls == [(2, 4, 5), (2, 4, 5)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result[0], np.full((4, 5), 8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full((4, 5), 19, dtype=np.float32))


def test_cellprofiler_contract_executor_infers_unknown_absorbed_contract():
    def two_dimensional_only(image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim != 2:
            raise RuntimeError("2D only")
        return image

    two_dimensional_only.__cellprofiler_declared_contract__ = "unknown"

    assert (
        CellProfilerFunctionContractMetadata.from_callable(
            two_dimensional_only
        ).resolve(two_dimensional_only)
        is ProcessingContract.PURE_2D
    )


def test_measurement_image_for_labels_reduces_stack_to_reference_slice() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image.shape == labels.shape
    np.testing.assert_array_equal(measurement_image, image[0])


def test_measurement_labels_collapse_singleton_label_stack() -> None:
    labels = np.ones((1, 4, 5), dtype=np.int32)

    measurement_labels = _measurement_labels(labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_measurement_labels_align_to_single_channel_image_stack() -> None:
    image = np.ones((1, 4, 5), dtype=np.float32)
    labels = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)

    measurement_labels = _measurement_labels_for_image(image, labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_object_only_reference_image_uses_one_stack_plane() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    reference_image = _object_only_reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0])


def test_measurement_table_rows_wrap_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    measurement_rows = _measurement_table_rows(row)

    assert measurement_rows == [row]


def test_filterobjects_relabels_additional_object_inputs_by_primary_retention() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[0:2, 0:2] = 1
    primary[2:5, 2:5] = 2
    cells = np.zeros_like(primary)
    cells[0:2, 0:2] = 10
    cells[2:5, 2:5] = 11

    result = filter_objects(
        image,
        mode=FilterMode.BORDER,
        object_labels=(primary, cells),
        additional_object_count=1,
        outline_object_indices=(0, 1),
        dtype_config=DtypeConfig(),
    )

    (
        _output_image,
        stats,
        filtered_primary,
        filtered_cells,
        primary_outline,
        cells_outline,
    ) = result

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary.max() == 1
    assert filtered_primary[3, 3] == 1
    assert filtered_cells.max() == 1
    assert filtered_cells[3, 3] == 1
    assert filtered_cells[0, 0] == 0
    assert primary_outline.max() == 1
    assert cells_outline.max() == 1


def test_filterobjects_uses_named_measurement_feature_rules() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2
    measurement_rows = [
        {"object_label": 1, "lower_quartile_intensity": 0.1},
        {"object_label": 2, "lower_quartile_intensity": 0.8},
    ]

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(name="NucleiMeasurements", rows=measurement_rows),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1
