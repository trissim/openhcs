"""Focused source-image provenance projection regressions."""

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceImageIdentity,
    SourceImageProvenance,
    SourceImageProvenanceContributor,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_matching import SourceImageSetIdentityPolicy


def test_source_provenance_projects_scalar_image_set_identity() -> None:
    provenance = SourceImageProvenance(
        source_path="/input/A01_s001_w1.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    )
    policy = SourceImageSetIdentityPolicy(frozenset((AllComponents.CHANNEL,)))

    identities = provenance.image_set_identities(policy)

    assert tuple(identity.components for identity in identities) == (
        (("site", "1"), ("well", "A01")),
    )


def test_source_provenance_preserves_plane_positions_before_axis_deduplication() -> (
    None
):
    provenance = SourceImageProvenance(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1.tif",
                "/input/A01_s001_w2.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "1", "channel": "2"},
            ),
        ),
    )
    policy = SourceImageSetIdentityPolicy(frozenset((AllComponents.CHANNEL,)))

    plane_identities = provenance.image_set_plane_identities(policy)
    axis = provenance.image_set_axis(policy)

    assert len(plane_identities) == 2
    assert plane_identities[0] == plane_identities[1]
    assert axis == (plane_identities[0],)


def test_source_provenance_axis_retains_distinct_image_sets() -> None:
    provenance = SourceImageProvenance(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w3.tif",
                "/input/A01_s002_w3.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "3"},
                {"well": "A01", "site": "2", "channel": "3"},
            ),
        ),
    )
    policy = SourceImageSetIdentityPolicy(frozenset((AllComponents.CHANNEL,)))

    axis = provenance.image_set_axis(policy)

    assert tuple(next(iter(identities)).components for identities in axis) == (
        (("site", "1"), ("well", "A01")),
        (("site", "2"), ("well", "A01")),
    )


def test_declared_source_projection_resolves_singleton_plane_contributor() -> None:
    source = ImagePayloadMetadata(
        source_path="/input/A01_DNA.tif",
        source_component_metadata={"well": "A01", "channel": "1"},
        source_image_names=("DNA",),
    ).payload_with(np.zeros((4, 5), dtype=np.float32))
    output = ImagePayloadMetadata(
        source_image_names=("OrigOverlay",),
    ).payload_with(np.ones((4, 5), dtype=np.float32))
    derived = image_payload_metadata(source).derive_payload(source, output)
    payload = ImagePayloadMetadata.compose((derived,)).payload_with(
        np.expand_dims(derived.data, axis=0)
    )

    projected_payload = image_payload_metadata(payload).project_declared_source_image(
        payload,
        "DNA",
    )
    projected = image_payload_metadata(projected_payload)

    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected_payload.data.shape == (1, 4, 5)
    assert projected.source_provenance.represented_source_image_names == (
        "OrigOverlay",
        "DNA",
    )
    assert projected.source_image_provenance_planes.contributor_count == 1


def test_complete_source_identity_accepts_nested_contributor() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (
                RuntimeSourceImageProvenancePlane(
                    contributors=(
                        SourceImageProvenanceContributor(
                            SourceImageIdentity(path="/input/A01_DNA.tif"),
                            source_image_name="DNA",
                        ),
                    )
                ),
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    payload = metadata.payload_with(np.zeros((1, 4, 5), dtype=np.float32))

    assert metadata.has_complete_source_identity(
        payload,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
    )
    assert metadata.source_image_paths == ("/input/A01_DNA.tif",)


def test_complete_source_identity_requires_multi_plane_axis_declaration() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_z001.tif", "/input/A01_z002.tif"),
            component_metadata=(
                {"well": "A01", "z_index": "1"},
                {"well": "A01", "z_index": "2"},
            ),
        )
    )
    payload = metadata.payload_with(np.zeros((2, 4, 5), dtype=np.float32))

    assert not metadata.has_complete_source_identity(
        payload,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )


def test_derived_singleton_runtime_plane_retains_declared_source_name() -> None:
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_DNA.tif",),
            component_metadata=({"well": "A01", "channel": "1"},),
        ),
        source_image_names=("DNA",),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((1, 4, 5), dtype=np.float32))
    output = ImagePayloadMetadata(
        source_image_names=("OrigOverlay",),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32))

    derived = image_payload_metadata(source).derive_payload(
        source,
        output,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
    )
    projected_payload = image_payload_metadata(derived).project_declared_source_image(
        derived,
        "DNA",
    )
    projected = image_payload_metadata(projected_payload)

    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected_payload.data.shape == (1, 4, 5)
    assert projected.source_provenance.represented_source_image_names == (
        "OrigOverlay",
        "DNA",
    )
    assert projected.source_image_provenance_planes.contributor_count == 1


def _repeated_source_alias_payload(
    *, data_plane_count: int = 7
) -> tuple[object, np.ndarray, np.ndarray]:
    pixels = np.stack(
        tuple(
            np.full((4, 5), plane_index + 1, dtype=np.float32)
            for plane_index in range(data_plane_count)
        )
    )
    mask = np.stack(
        tuple(
            np.full((4, 5), plane_index % 2 == 0, dtype=bool)
            for plane_index in range(data_plane_count)
        )
    )
    aliases = (
        "OrigDNA",
        "OrigER",
        "OrigRNA",
        "OrigActin",
        "OrigMito",
        "OrigGolgi",
        "OrigER",
    )
    metadata = ImagePayloadMetadata(
        source_image_names=aliases,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/input/plane_{index}.tif" for index in range(7)),
            component_metadata=tuple(
                {"well": "A01", "site": str(index + 1)} for index in range(7)
            ),
        ),
        source_plane_intensity_scales=(
            10.0,
            255.0,
            30.0,
            40.0,
            50.0,
            60.0,
            65535.0,
        ),
        source_plane_dtypes=(
            "uint8",
            "uint8",
            "uint8",
            "uint8",
            "uint8",
            "uint8",
            "uint16",
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    return metadata.payload_with(pixels, mask), pixels, mask


def test_declared_source_projection_selects_ordered_repeated_alias_planes() -> None:
    payload, pixels, mask = _repeated_source_alias_payload()

    projected_payload = image_payload_metadata(payload).project_declared_source_image(
        payload, "OrigER"
    )
    projected = image_payload_metadata(projected_payload)

    np.testing.assert_array_equal(
        image_payload_data(projected_payload),
        np.stack((pixels[1], pixels[6])),
    )
    np.testing.assert_array_equal(
        image_payload_mask(projected_payload),
        np.stack((mask[1], mask[6])),
    )
    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected.source_plane_intensity_scales == (255.0, 65535.0)
    assert projected.source_plane_dtypes == ("uint8", "uint16")
    assert projected.source_image_names == ("OrigER", "OrigER")
    assert projected.source_image_provenance_planes.paths == (
        "/input/plane_1.tif",
        "/input/plane_6.tif",
    )
    assert tuple(
        dict(metadata or {})
        for metadata in projected.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "2"},
        {"well": "A01", "site": "7"},
    )


def test_declared_source_projection_drops_axis_for_single_selected_plane() -> None:
    payload, pixels, mask = _repeated_source_alias_payload()

    projected_payload = image_payload_metadata(payload).project_declared_source_image(
        payload, "OrigRNA"
    )
    projected = image_payload_metadata(projected_payload)

    np.testing.assert_array_equal(image_payload_data(projected_payload), pixels[2])
    np.testing.assert_array_equal(image_payload_mask(projected_payload), mask[2])
    assert projected.plane_axis is None
    assert projected.intensity_scale == 30.0
    assert projected.source_dtype == "uint8"
    assert projected.source_path == "/input/plane_2.tif"


def test_declared_source_projection_validates_runtime_plane_cardinality() -> None:
    payload, _, _ = _repeated_source_alias_payload(data_plane_count=6)

    with pytest.raises(
        ValueError,
        match="does not match its declared 'runtime_slice' axis of size 7",
    ):
        image_payload_metadata(payload).project_declared_source_image(
            payload,
            "OrigER",
        )
