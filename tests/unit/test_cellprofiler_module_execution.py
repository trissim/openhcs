import numpy as np

from benchmark.cellprofiler_compat.module_execution import (
    CellProfilerFunctionContractMetadata,
    CellProfilerFunctionContractExecutor,
    _coerce_invocation_kwargs,
    _measurement_image_for_labels,
    _measurement_labels,
    _measurement_labels_for_image,
    _measurement_table_rows,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)
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


def test_measurement_table_rows_wrap_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    measurement_rows = _measurement_table_rows(row)

    assert measurement_rows == [row]
