from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.artifacts import ArtifactOutputPlan, ImageArtifactType
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceImageIdentity,
    SourceImageProvenanceContributor,
    SourceImageProvenancePlanes,
)


def _source_plane(site: int) -> RuntimeSourceImageProvenancePlane:
    identity = SourceImageIdentity(
        f"/input/A01_s{site}_w1.tif",
        {"well": "A01", "site": str(site), "channel": "1"},
    )
    return RuntimeSourceImageProvenancePlane(
        identity,
        (SourceImageProvenanceContributor(identity, source_image_name="OrigGreen"),),
        "OrigGreen",
    )


def test_image_artifact_normalization_applies_declared_scalar_identity() -> None:
    data = np.arange(20, dtype=np.uint16).reshape(4, 5)
    mask = data % 2 == 0
    payload = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
        source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
        source_image_names=("OrigGreen",),
        intensity_scale=65535.0,
        source_dtype="uint16",
    ).payload_with(data, mask)

    output_plan = ArtifactOutputPlan(
        name="TumorImage",
        path="/memory/TumorImage.pkl",
        artifact_type=ImageArtifactType,
    )
    runtime_value = RuntimeValue.normalize(
        output_plan,
        payload,
        axis_id="A01",
    )
    normalized = runtime_value.data
    metadata = image_payload_metadata(normalized)

    assert runtime_value.key.name == output_plan.name
    assert image_payload_data(normalized) is data
    assert image_payload_mask(normalized) is mask
    assert metadata.source_path == "/input/A01_s1_w1.tif"
    assert dict(metadata.source_component_metadata or {}) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    assert metadata.intensity_scale == 65535.0
    assert metadata.source_dtype == "uint16"
    assert metadata.source_image_names == ("TumorImage",)
    assert metadata.source_provenance.represented_source_image_names == (
        "TumorImage",
        "OrigGreen",
    )


def test_image_artifact_normalization_names_each_aligned_image_payload() -> None:
    arrays = (
        np.zeros((4, 5), dtype=np.float32),
        np.ones((4, 5), dtype=np.float32),
    )
    payloads = tuple(
        ImagePayloadMetadata(
            source_path=f"/input/A01_s{site}_w1.tif",
            source_component_metadata={
                "well": "A01",
                "site": str(site),
                "channel": "1",
            },
            source_image_names=("OrigGreen",),
            source_dtype="float32",
        ).payload_with(array)
        for site, array in enumerate(arrays, start=1)
    )

    source_plan = ArtifactOutputPlan(
        name="OrigGreen",
        path="/memory/OrigGreen.pkl",
        artifact_type=ImageArtifactType,
    )
    source_values = tuple(
        RuntimeValue.normalize(source_plan, payload, axis_id="A01")
        for payload in payloads
    )
    output_plan = ArtifactOutputPlan(
        name="UntangledWorms",
        path="/memory/UntangledWorms.pkl",
        artifact_type=ImageArtifactType,
    )
    runtime_value = RuntimeValue.normalize(
        output_plan,
        RuntimeSliceAlignedValues(source_values),
        axis_id="A01",
    )
    normalized = runtime_value.data

    assert runtime_value.key.name == output_plan.name
    assert isinstance(normalized, RuntimeSliceAlignedValues)
    assert normalized.slice_count == 2
    for index, (array, source_path) in enumerate(
        zip(
            arrays,
            ("/input/A01_s1_w1.tif", "/input/A01_s2_w1.tif"),
            strict=True,
        )
    ):
        value = normalized.value_for_slice(index)
        metadata = image_payload_metadata(value)
        assert image_payload_data(value) is array
        assert metadata.source_path == source_path
        assert metadata.source_dtype == "float32"
        assert metadata.source_image_names == ("UntangledWorms",)
        assert metadata.source_provenance.represented_source_image_names == (
            "UntangledWorms",
            "OrigGreen",
        )


def test_derived_singleton_runtime_plane_projects_by_output_name() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (_source_plane(1),)
        ),
        source_image_names=("OrigGreen",),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    derived = metadata.with_source_provenance(
        metadata.source_provenance.with_derived_source_image_names(
            ("OrigGreenOverlay",)
        )
    )
    payload = derived.payload_with(np.zeros((1, 4, 5), dtype=np.float32))
    projected_payload = derived.project_declared_source_image(
        payload,
        "OrigGreenOverlay",
    )
    projected = image_payload_metadata(projected_payload)

    assert derived.source_image_provenance_planes.runtime_source_image_names == (
        "OrigGreenOverlay",
    )
    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert image_payload_data(projected_payload).shape == (1, 4, 5)
    assert projected.source_image_names == ("OrigGreenOverlay",)
    assert projected.source_provenance.represented_source_image_names == (
        "OrigGreenOverlay",
        "OrigGreen",
    )
    assert projected.source_image_provenance_planes.contributor_count == 1


def test_declared_singleton_name_owns_projection_over_nested_source_name() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (_source_plane(1),)
        ),
        source_image_names=("OrigGreenOverlay",),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    payload = metadata.payload_with(np.zeros((1, 4, 5), dtype=np.float32))
    projected_payload = metadata.project_declared_source_image(
        payload,
        "OrigGreenOverlay",
    )
    projected = image_payload_metadata(projected_payload)

    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert image_payload_data(projected_payload).shape == (1, 4, 5)
    assert projected.source_image_names == ("OrigGreenOverlay",)
    assert projected.source_provenance.represented_source_image_names == (
        "OrigGreenOverlay",
        "OrigGreen",
    )


def test_one_derived_name_applies_to_every_declared_runtime_plane() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (_source_plane(1), _source_plane(2))
        ),
        source_image_names=("OrigGreen", "OrigGreen"),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    derived = metadata.source_provenance.with_derived_source_image_names(
        ("OrigGreenOverlay",)
    )

    assert derived.source_image_provenance_planes.runtime_source_image_names == (
        "OrigGreenOverlay",
        "OrigGreenOverlay",
    )
    assert derived.source_plane_selection("OrigGreenOverlay") == (0, 1)


def test_named_source_projection_selects_one_contributor_per_runtime_plane() -> None:
    planes = tuple(
        RuntimeSourceImageProvenancePlane(
            SourceImageIdentity(component_metadata={"well": "A01", "site": site}),
            (
                SourceImageProvenanceContributor(
                    SourceImageIdentity(
                        f"/input/A01_s{site}_w1.tif",
                        {"well": "A01", "site": site, "channel": "1"},
                    ),
                    source_image_name="DNA",
                ),
                SourceImageProvenanceContributor(
                    SourceImageIdentity(
                        f"/input/A01_s{site}_w2.tif",
                        {"well": "A01", "site": site, "channel": "2"},
                    ),
                    source_image_name="RNA",
                ),
            ),
        )
        for site in (1, 2)
    )
    provenance = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(planes)
    ).source_provenance

    projected = provenance.for_source_image("RNA")

    assert projected.source_plane_count == 2
    assert projected.source_image_names == ("RNA",)
    assert projected.source_image_provenance_planes.paths == (
        "/input/A01_s1_w2.tif",
        "/input/A01_s2_w2.tif",
    )
    assert projected.source_image_provenance_planes.contributor_count == 0
    assert tuple(
        dict(metadata or {})
        for metadata in projected.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": 1, "channel": "2"},
        {"well": "A01", "site": 2, "channel": "2"},
    )


def test_source_name_projects_one_exact_named_runtime_slice() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (_source_plane(1), _source_plane(2))
        ),
        source_image_names=("First", "Second"),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    payload = metadata.payload_with(np.zeros((2, 4, 5), dtype=np.float32))

    projected_payload = metadata.project_declared_source_image(payload, "Second")
    projected = image_payload_metadata(projected_payload)

    assert image_payload_data(projected_payload).shape == (4, 5)
    assert projected.plane_axis is None
    assert projected.source_path == "/input/A01_s2_w1.tif"
    assert projected.source_image_names == ("OrigGreen",)


def test_contributor_missing_identity_does_not_import_runtime_plane_topology() -> None:
    contributor = SourceImageProvenanceContributor(source_image_name="OrigGreen")
    fallback = _source_plane(1)

    merged = contributor.with_missing_from(fallback)

    assert merged.source_identity == fallback.source_identity
    assert merged.source_image_name == "OrigGreen"
    assert merged.contributors == ()


def test_derived_plane_names_require_exact_declared_cardinality() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (_source_plane(1), _source_plane(2), _source_plane(3))
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match="one shared name or exactly one name"):
        metadata.source_provenance.with_derived_source_image_names(("A", "B"))


def test_scalar_contributors_survive_runtime_plane_projection() -> None:
    identity = SourceImageIdentity(
        "/input/A01_s1_w1.tif",
        {"well": "A01", "site": "1", "channel": "1"},
    )
    metadata = ImagePayloadMetadata(
        source_path=identity.path,
        source_component_metadata=identity.component_metadata,
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (
                SourceImageProvenanceContributor(
                    identity,
                    source_image_name="DNA",
                ),
            )
        ),
        source_image_names=("OrigOverlay",),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    projected = metadata.for_leading_source_plane(0)

    assert projected.plane_axis is None
    assert projected.source_image_names == ("OrigOverlay",)
    assert projected.source_provenance.represented_source_image_names == (
        "OrigOverlay",
        "DNA",
    )
    assert projected.source_image_provenance_planes.contributors == (
        SourceImageProvenanceContributor(
            identity,
            source_image_name="DNA",
        ),
    )
    payload = projected.payload_with(np.zeros((4, 5), dtype=np.float32))
    projected_payload = projected.project_declared_source_image(payload, "DNA")
    assert image_payload_metadata(projected_payload) == projected
