from __future__ import annotations

import inspect

import numpy as np

from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.processing.backends.cellprofiler.display_modules import (
    display_data_on_image,
)


def test_display_data_on_image_preserves_nominal_runtime_image_context() -> None:
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 2] = False
    image = ImagePayloadMetadata(
        source_channel_axis=-1,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5)),
    ).payload_with(
        np.ones((4, 5, 3), dtype=np.float32),
        mask,
    )

    output = inspect.unwrap(display_data_on_image)(image)

    assert isinstance(output, MaskedImagePayload)
    assert image_payload_data(output).shape == (4, 5, 3)
    np.testing.assert_array_equal(image_payload_mask(output), mask)
    assert image_payload_metadata(output).source_channel_axis == -1
