import numpy as np
import pytest

from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelVariantData,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceImageIdentity,
    SourceImageProvenance,
    SourceImageProvenancePlanes,
)


def test_payload_domain_replacement_drops_inherited_plane_axis() -> None:
    labels = np.asarray(
        (
            ((0, 1), (0, 0)),
            ((0, 1), (0, 2)),
        ),
        dtype=np.int32,
    )
    source = ObjectLabelSet(
        name="erodedDownsizedNuclei",
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1, 2)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    transformed = object_label_value_with_dense_labels(
        source,
        labels.copy(),
        domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
    )

    assert transformed.domain == ObjectLabelDomain.declared(
        scope=ObjectLabelDomainScope.PAYLOAD,
        declared_object_ids=(1, 2),
    )
    assert transformed.plane_axis is None
    np.testing.assert_array_equal(transformed.labels, labels)


def test_explicit_payload_domain_plane_axis_remains_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="payload-scoped labels cannot declare a plane axis",
    ):
        ObjectLabelSet(
            name="invalid",
            variant_data=ObjectLabelVariantData(
                labels=np.zeros((2, 2, 2), dtype=np.int32)
            ),
            domain=ObjectLabelDomain.declared(
                scope=ObjectLabelDomainScope.PAYLOAD,
                declared_object_count=0,
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_payload_measurement_reference_keeps_volume_planes_as_contributors() -> None:
    source_planes = SourceImageProvenancePlanes(
        tuple(
            RuntimeSourceImageProvenancePlane(
                SourceImageIdentity(f"/plate/A01_z{index:03d}.tif"),
                source_image_name="MembFinal",
            )
            for index in range(3)
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((3, 4, 5), dtype=np.int32)
        ),
        source_provenance=SourceImageProvenance(
            source_image_provenance_planes=source_planes,
            source_image_names=("MembFinal",) * 3,
        ),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_count=0,
        ),
    )

    reference_image = labels.measurement_reference_image()
    reference_metadata = reference_image.metadata

    assert reference_metadata.plane_axis is None
    assert reference_metadata.source_provenance.source_plane_count == 0
    assert (
        reference_metadata.source_image_provenance_planes.contributor_count == 3
    )
    assert reference_metadata.source_image_names == ()
    assert reference_metadata.source_provenance.represented_source_image_names == (
        "MembFinal",
    )
    composed = ImagePayloadMetadata.compose(
        (reference_image,),
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    )
    assert composed.source_provenance.represented_source_image_names == (
        "MembFinal",
    )
