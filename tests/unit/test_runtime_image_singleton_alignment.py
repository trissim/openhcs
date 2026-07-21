from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.source_spatial_domain import SourceSpatialDomain


_SPATIAL_DOMAIN = SourceSpatialDomain(source_shape_yx=(4, 5))
_SINGLETON_MASK = np.array(
    (
        (True, False, True, True, True),
        (True, True, True, True, True),
        (False, True, True, True, True),
        (True, True, True, False, True),
    ),
    dtype=bool,
)


def _scalar_image(value: float, *, masked: bool = False):
    return ImagePayloadMetadata(source_spatial_domain=_SPATIAL_DOMAIN).payload_with(
        np.full((4, 5), value, dtype=np.float32),
        _SINGLETON_MASK if masked else None,
    )


def _singleton_owner(owner_kind: str):
    if owner_kind == "aligned_stack":
        return AlignedImageStack((_scalar_image(11, masked=True),))
    if owner_kind == "runtime_payload":
        return ImagePayloadMetadata(
            source_spatial_domain=_SPATIAL_DOMAIN,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.full((1, 4, 5), 11, dtype=np.float32),
            _SINGLETON_MASK[np.newaxis, ...],
        )
    raise AssertionError(f"Unknown singleton owner kind: {owner_kind!r}.")


@pytest.mark.parametrize("owner_kind", ("aligned_stack", "runtime_payload"))
@pytest.mark.parametrize("singleton_first", (False, True))
def test_scalar_image_aligns_with_nominal_singleton_owner(
    owner_kind: str,
    singleton_first: bool,
) -> None:
    singleton = _singleton_owner(owner_kind)
    scalar = _scalar_image(7)
    payloads = (singleton, scalar) if singleton_first else (scalar, singleton)

    composition = compose_aligned_image_payload("Measurement consumer", payloads)

    assert composition.execution_mode is (
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 1
    bundle = composition.payload.slices[0]
    assert image_payload_metadata(bundle).plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert image_payload_data(bundle).shape == (2, 4, 5)
    expected_values = (11, 7) if singleton_first else (7, 11)
    for index, expected_value in enumerate(expected_values):
        np.testing.assert_array_equal(
            image_payload_data(bundle)[index],
            np.full((4, 5), expected_value, dtype=np.float32),
        )
    np.testing.assert_array_equal(image_payload_mask(bundle), _SINGLETON_MASK)


@pytest.mark.parametrize("owner_kind", ("aligned_stack", "runtime_payload"))
def test_scalar_image_still_requires_owner_for_multi_slice_alignment(
    owner_kind: str,
) -> None:
    if owner_kind == "aligned_stack":
        aligned = AlignedImageStack((_scalar_image(11), _scalar_image(12)))
    else:
        aligned = ImagePayloadMetadata(
            source_spatial_domain=_SPATIAL_DOMAIN,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.stack(
                (
                    np.full((4, 5), 11, dtype=np.float32),
                    np.full((4, 5), 12, dtype=np.float32),
                )
            ),
            None,
        )

    with pytest.raises(ValueError, match="explicit .* owner"):
        compose_aligned_image_payload(
            "Measurement consumer",
            (_scalar_image(7), aligned),
        )


def test_same_shaped_scalar_images_do_not_invent_runtime_alignment() -> None:
    composition = compose_aligned_image_payload(
        "Measurement consumer",
        (_scalar_image(7), _scalar_image(11)),
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert not isinstance(composition.payload, AlignedImageStack)
    assert image_payload_metadata(composition.payload).plane_axis is (
        RuntimePlaneAxis.SOURCE_BINDING
    )


def test_singleton_runtime_projection_consumes_rgb_image_and_mask_axis_together() -> None:
    payload = ImagePayloadMetadata(
        source_spatial_domain=_SPATIAL_DOMAIN,
        source_channel_axis=3,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.zeros((1, 4, 5, 3), dtype=np.float32),
        _SINGLETON_MASK[np.newaxis, ...],
    )

    projected = RuntimeSliceProjection.value_for_singleton_slice(
        payload,
        source_description="RGB image output",
    )

    assert isinstance(projected, MaskedImagePayload)
    assert image_payload_data(projected).shape == (4, 5, 3)
    np.testing.assert_array_equal(image_payload_mask(projected), _SINGLETON_MASK)
    projected_metadata = image_payload_metadata(projected)
    assert projected_metadata.plane_axis is None
    assert projected_metadata.normalized_source_channel_axis(projected) == 2


def test_singleton_runtime_projection_rejects_multi_slice_payload() -> None:
    payload = ImagePayloadMetadata(
        source_spatial_domain=_SPATIAL_DOMAIN,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.zeros((2, 4, 5), dtype=np.float32),
        None,
    )

    with pytest.raises(ValueError, match="exactly one runtime slice"):
        RuntimeSliceProjection.value_for_singleton_slice(
            payload,
            source_description="Image output",
        )
