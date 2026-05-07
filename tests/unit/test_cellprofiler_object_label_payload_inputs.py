import inspect

import numpy as np

from benchmark.cellprofiler_library.functions.convertobjectstoimage import (
    ImageMode,
    convert_objects_to_image,
)
from benchmark.cellprofiler_library.functions.measureimageareaoccupied import (
    OperandChoice,
    measure_image_area_occupied,
)
from openhcs.core.runtime_values import ObjectLabelPayload


def test_area_occupied_accepts_object_label_payload_input() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    payload = ObjectLabelPayload(labels=labels)

    output_image, measurements = inspect.unwrap(measure_image_area_occupied)(
        image,
        operand_choices=(OperandChoice.OBJECTS,),
        input_names=("Cells",),
        retained_image_names=("CellMask",),
        object_labels=(payload,),
    )

    assert output_image.dtype == image.dtype
    assert output_image.sum() == 4
    assert measurements[0].area_occupied == 4.0
    assert measurements[0].total_area == 16.0


def test_convert_objects_to_image_accepts_object_label_payload_input() -> None:
    image = np.zeros((3, 3), dtype=np.float32)
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1
    payload = ObjectLabelPayload(labels=labels)

    converted = inspect.unwrap(convert_objects_to_image)(
        image,
        payload,
        image_mode=ImageMode.BINARY,
    )

    assert converted.dtype == np.float32
    assert converted.sum() == 1.0
