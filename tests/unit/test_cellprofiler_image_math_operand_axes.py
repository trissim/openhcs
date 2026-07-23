import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadBundleContext
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation,
    image_math,
)


def test_image_math_splits_only_declared_source_binding_operand_axis() -> None:
    first = ImagePayloadMetadata(source_image_names=("First",)).payload_with(
        np.full((2, 3), 0.2, dtype=np.float32)
    )
    second = ImagePayloadMetadata(source_image_names=("Second",)).payload_with(
        np.full((2, 3), 0.3, dtype=np.float32)
    )
    image = ImagePayloadBundleContext.from_payloads((first, second)).compose()

    result = image_math(
        image,
        operation=ImageMathOperation.ADD,
        truncate_high=False,
    )

    np.testing.assert_allclose(image_payload_data(result), 0.5)
    assert image_payload_metadata(result).plane_axis is None


def test_image_math_preserves_declared_runtime_slice_as_one_operand() -> None:
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.array(
            (
                ((0.0, 0.25), (0.5, 1.0)),
                ((0.1, 0.2), (0.3, 0.4)),
            ),
            dtype=np.float32,
        )
    )

    result = image_math(image, operation=ImageMathOperation.INVERT)

    np.testing.assert_allclose(image_payload_data(result), 1.0 - image_payload_data(image))
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_image_math_does_not_infer_operand_axis_from_array_rank() -> None:
    image = np.full((2, 3, 4), 0.25, dtype=np.float32)

    result = image_math(image, operation=ImageMathOperation.INVERT)

    np.testing.assert_allclose(result, 0.75)
