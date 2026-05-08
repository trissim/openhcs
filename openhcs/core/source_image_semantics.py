"""Source image payload transforms implied by typed pipeline image semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.pipeline_image_schema import (
    ImageTypeSourceRole,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
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
    if image_type is None:
        return payload
    return SourceImagePayloadSemantics(
        role=ImageTypeSourceRole.for_image_type(image_type),
        source_path=source_path,
    ).apply(payload)


@dataclass(frozen=True, slots=True)
class SourceImagePayloadSemantics:
    """Typed source-image role behavior applied to one loaded payload."""

    role: ImageTypeSourceRole
    source_path: str | None

    def apply(self, payload: Any) -> Any:
        original_data = image_payload_data(payload)
        data = self._source_data(original_data)
        if data is original_data and not self.role.materialize_source_mask:
            return payload
        return image_payload_with_context(
            data,
            mask=self._source_mask(payload, data),
            metadata=ImagePayloadMetadata.for_array_payload(
                data,
                source_path=self.source_path,
            ),
        )

    def _source_data(self, data: Any) -> Any:
        if not self.role.load_as_monochrome:
            return data
        if is_color_image_slice(data):
            return self._cellprofiler_rgb_to_gray(np.asarray(data)[..., :3])
        if is_color_image_stack(data):
            return np.stack(
                [
                    self._cellprofiler_rgb_to_gray(np.asarray(plane)[..., :3])
                    for plane in np.asarray(data)
                ],
                axis=0,
            )
        return data

    def _source_mask(self, payload: Any, data: Any) -> Any | None:
        mask = image_payload_mask(payload)
        if mask is not None or not self.role.materialize_source_mask:
            return mask
        return np.ones(self._source_mask_shape(data), dtype=bool)

    @staticmethod
    def _source_mask_shape(data: Any) -> tuple[int, ...]:
        array = np.asarray(data)
        if is_color_image_slice(array):
            return tuple(int(value) for value in array.shape[:2])
        if is_color_image_stack(array):
            return tuple(int(value) for value in array.shape[:-1])
        return tuple(int(value) for value in array.shape)

    @staticmethod
    def _cellprofiler_rgb_to_gray(rgb_data: Any) -> np.ndarray:
        from skimage.color import rgb2gray

        return rgb2gray(rgb_data)
