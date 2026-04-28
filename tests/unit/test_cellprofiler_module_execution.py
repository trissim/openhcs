import numpy as np

from benchmark.cellprofiler_compat.module_execution import (
    _coerce_invocation_kwargs,
    _measurement_image_for_labels,
    _measurement_labels,
    _measurement_table_rows,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)


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


def test_measurement_table_rows_wrap_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    measurement_rows = _measurement_table_rows(row)

    assert measurement_rows == [row]
