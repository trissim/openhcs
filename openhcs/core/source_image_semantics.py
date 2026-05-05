"""Source image payload transforms implied by typed pipeline image semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.pipeline_image_schema import (
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
    image_type_loads_as_monochrome,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_with_context,
)
from openhcs.core.source_matching import source_metadata_value


def apply_source_image_loading_semantics(
    payload: Any,
    *,
    source_metadata: Mapping[str, str] | None,
    source_path: str | None,
) -> Any:
    """Apply typed source image semantics to pixels loaded from storage."""

    if source_metadata is None:
        return payload
    image_type = source_metadata_value(
        source_metadata,
        SOURCE_IMAGE_TYPE_METADATA_FIELD,
    )
    if image_type is None or not image_type_loads_as_monochrome(image_type):
        return payload

    data = image_payload_data(payload)
    if is_color_image_slice(data):
        return _transformed_source_image_payload(
            payload,
            _cellprofiler_rgb_to_gray(np.asarray(data)[..., :3]),
            source_path,
        )
    if is_color_image_stack(data):
        converted = np.stack(
            [
                _cellprofiler_rgb_to_gray(np.asarray(plane)[..., :3])
                for plane in np.asarray(data)
            ],
            axis=0,
        )
        return _transformed_source_image_payload(payload, converted, source_path)
    return payload


def _cellprofiler_rgb_to_gray(rgb_data: Any) -> np.ndarray:
    from skimage.color import rgb2gray

    return rgb2gray(rgb_data)


def _transformed_source_image_payload(
    payload: Any,
    data: Any,
    source_path: str | None,
) -> Any:
    return image_payload_with_context(
        data,
        mask=image_payload_mask(payload),
        metadata=ImagePayloadMetadata.for_array_payload(data, source_path=source_path),
    )
