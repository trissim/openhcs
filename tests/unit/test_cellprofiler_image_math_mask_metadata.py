import numpy as np

from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    MaskedImagePayload,
)
from openhcs.processing.backends.cellprofiler.image_math import image_math


def test_image_math_ignore_masks_preserves_source_provenance_metadata() -> None:
    pixels = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    payload = MaskedImagePayload(
        data=pixels,
        mask=np.array([[True, False], [True, True]]),
        metadata=ImagePayloadMetadata(source_path="/input/source.tif"),
    )

    result = image_math(
        payload,
        operation="Invert",
        ignore_masks=True,
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, ImageMetadataPayload)
    assert result.metadata.source_path == "/input/source.tif"
    np.testing.assert_allclose(result.data, 1.0 - pixels)
