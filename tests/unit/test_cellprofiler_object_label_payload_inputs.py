import inspect

import numpy as np
import pytest

from openhcs.processing.backends.cellprofiler.object_images import (
    ImageMode,
    convert_objects_to_image,
)
from openhcs.processing.backends.cellprofiler.area_occupied import (
    AreaOccupiedRow,
    OperandChoice,
    measure_image_area_occupied,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    measure_image_intensity_objects,
)
from openhcs.processing.backends.cellprofiler.skeleton import measure_object_skeleton
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)


def test_area_occupied_accepts_object_label_payload_input() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    payload = ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))

    output_image, measurements = inspect.unwrap(measure_image_area_occupied)(
        image,
        operand_choices=(OperandChoice.OBJECTS,),
        area_occupied_rows=(
            AreaOccupiedRow(OperandChoice.OBJECTS, "Cells"),
        ),
        object_labels=(payload,),
    )

    assert output_image.dtype == image.dtype
    np.testing.assert_array_equal(output_image, image)
    (measurement,) = measurements.row_mappings()
    assert measurement["area_occupied"] == 4.0
    assert measurement["total_area"] == 16.0


def test_measure_image_intensity_accepts_projected_object_label_payload() -> None:
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1:, 1:] = 1

    output, measurements = measure_image_intensity_objects(
        image,
        labels=ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
    )

    np.testing.assert_array_equal(output, image)
    (measurement,) = measurements.row_mappings()
    assert measurement["total_intensity"] == 24.0


def test_measure_image_intensity_rejects_unprojected_object_label_stack() -> None:
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.ones((1, 3, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="already be projected to one 2-D"):
        measure_image_intensity_objects(
            image,
            labels=ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=labels)
            ),
        )


def test_convert_objects_to_image_accepts_object_label_payload_input() -> None:
    image = np.zeros((3, 3), dtype=np.float32)
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1
    payload = ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))

    converted = inspect.unwrap(convert_objects_to_image)(
        image,
        payload,
        image_mode=ImageMode.BINARY,
    )

    assert converted.dtype == np.float32
    assert converted.sum() == 1.0


def test_measure_object_skeleton_rejects_unprojected_object_label_stack() -> None:
    image = np.zeros((5, 5), dtype=bool)
    labels = np.zeros((2, 5, 5), dtype=np.int32)

    with pytest.raises(
        ValueError, match="runtime-projected 2-D image and label planes"
    ):
        inspect.unwrap(measure_object_skeleton)(
            image,
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        )
