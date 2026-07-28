"""Generic source payload transforms declared by source bindings."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    normalize_image_payload_intensity,
)
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.source_bindings import NamedSourceBinding


def _rgb_to_monochrome(data: np.ndarray) -> np.ndarray:
    rgb = np.asarray(data)[..., :3]
    if np.all(rgb == rgb[..., :1]):
        return np.ascontiguousarray(rgb[..., 0])

    from skimage.color import rgb2gray

    return rgb2gray(rgb)


def _monochrome_source_data(
    data: RuntimeArrayData,
    channel_axis: int | None,
    metadata: ImagePayloadMetadata,
) -> RuntimeArrayData:
    if channel_axis is None:
        return data
    normalized = normalize_image_payload_intensity(
        metadata.payload_with(data),
        dtype=np.float32,
    )
    channel_last = np.moveaxis(
        np.asarray(image_payload_data(normalized)),
        channel_axis,
        -1,
    )
    return _rgb_to_monochrome(channel_last)


def _loaded_source_payload_metadata(
    payload: RuntimeArrayData,
    binding: NamedSourceBinding,
    source_context: ImagePayloadSourceMetadataContext | None,
) -> ImagePayloadMetadata:
    existing = image_payload_metadata(payload)
    if source_context is None:
        metadata = (
            existing
            if existing.has_values
            else ImagePayloadMetadata.for_array_payload(payload)
        )
    else:
        metadata = source_context.metadata(payload, source_binding=binding)
    metadata = metadata.replace_fields(
        source_provenance=metadata.source_provenance.with_source_image_names(
            (binding.alias,)
        )
    )
    source_channel_axis = binding.source_channel_axis_for_shape(
        np.shape(image_payload_data(payload)),
        observed_axis=metadata.source_channel_axis,
    )
    metadata = replace(metadata, source_channel_axis=source_channel_axis)
    metadata.normalized_source_channel_axis(payload)
    return metadata


def apply_source_binding_payload(
    payload: RuntimeArrayData,
    binding: NamedSourceBinding,
    source_context: ImagePayloadSourceMetadataContext | None,
) -> RuntimeArrayData:
    """Apply one resolved source binding to a loaded runtime payload."""
    metadata = _loaded_source_payload_metadata(payload, binding, source_context)
    data = image_payload_data(payload)
    source_channel_axis = metadata.source_channel_axis
    data, source_channel_axis = binding.artifact_kind.normalize_source_payload(
        data,
        source_channel_axis,
    )
    if binding.load_as_monochrome and source_channel_axis is not None:
        data = _monochrome_source_data(data, source_channel_axis, metadata)
        source_channel_axis = None
        metadata = replace(
            metadata.without_unit_interval_intensity_scale(),
            intensity_scale=ImagePayloadMetadata.for_array(data).intensity_scale,
        )
    if binding.load_as_mask:
        data = np.asarray(data, dtype=bool)

    mask = image_payload_mask(payload)
    metadata = replace(metadata, source_channel_axis=source_channel_axis)
    if mask is not None:
        return MaskedImagePayload(data, mask, metadata)
    if metadata.has_values:
        return ImageMetadataPayload(data, metadata)
    return data
