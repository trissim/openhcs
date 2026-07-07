from dataclasses import dataclass

import pytest
import numpy as np
import pandas as pd

from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactOutputPlan,
    ArtifactScope,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
    MetadataArtifactType,
)
from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectionSourceIdentityRequirement,
    RuntimeProjectionSourceIdentityRequest,
    RuntimeSliceProjection,
    RuntimeSliceProjectionStrategy,
    RuntimeProjectionAxis,
)
from openhcs.core.source_image_semantics import (
    SourceImagePayloadSemantics,
    source_image_payload_role,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
    SourceVoxelSpacing,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_values import (
    FieldSpec,
    DerivedImagePayloadContext,
    DenseObjectLabelPlaneDomainStackRequest,
    DenseObjectLabelSliceStackRequest,
    ImagePayloadChannelProjection,
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MeasurementTable,
    MeasurementScope,
    MeasurementSubject,
    MaskedImagePayload,
    NamedImage,
    ObjectLabelPayload,
    ObjectLabelDerivedPlaneProjectionRequest,
    ObjectLabelDomainScope,
    ObjectLabelSet,
    ObjectLabelDenseDataStrategy,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelRepresentation,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelSetPlaneStackContract,
    ObjectLabelVariantCompatibilityStrategy,
    ObjectLabelReplacementRequest,
    RelationshipEndpoint,
    ObjectRelationship,
    RuntimeArrayPayload,
    RuntimeImageSourceIdentityCompleteness,
    RuntimeStoragePolicy,
    RuntimeValue,
    RuntimeValueSchema,
    SparseIJVLabelRows,
    SpatialGrid,
    SingletonObjectLabelStackCollapseStrategy,
    SourceImageObjectLabelBuildRequest,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    ImagePayloadSourceMetadataContext,
    ImagePayloadMetadataCompositionMode,
    ImagePayloadMetadataCompositionRequest,
    SourceImageProvenancePlanes,
    image_payload_data,
    image_mask_for_data_domain,
    image_payload_metadata,
    RuntimeImagePayloadContext,
    ColumnarRows,
    normalize_image_payload_intensity,
    normalize_artifact_value,
    object_label_dense_array,
    object_label_value_with_dense_labels,
    ObjectLabelValueBuilderStrategy,
    ObjectLabelSetReplacementStrategy,
    SourceImageIdentity,
)
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectFeatureArrayDomain,
    ObjectFeatureArrayDomainStrategy,
    ObjectFeatureMissingValue,
    ObjectFeatureMissingValueStrategy,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    ObjectFeatureValueTable,
    ObjectLabelVariant,
    MeasurementRowAxisField,
    MeasurementObjectRowIdentity,
    RuntimePlaneAxis,
    SpatialGridOrdering,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DAuxiliaryOutputAggregator,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    ObjectSizeShapeMeasurementRowsRequest,
    ShapeObjectFeatureValueTable,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    ShapeZernikeFeatureAuthority,
)


@dataclass(frozen=True, slots=True)
class _RuntimeValueTestColumnarRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]


def test_runtime_slice_aligned_values_repeat_across_divisible_outer_domain():
    values = RuntimeSliceAlignedValues(("a", "b"))

    assert [values.value_for_aligned_slice(index, 4) for index in range(4)] == [
        "a",
        "b",
        "a",
        "b",
    ]


class ArrayLike(RuntimeArrayPayload):
    shape = (3, 3)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


class NominalObjectLabelDomainCarrier(ObjectLabelDomainMetadata):
    def __init__(self, domain: ObjectLabelDomain) -> None:
        self._domain = domain

    def object_label_domain(self) -> ObjectLabelDomain:
        return self._domain


class SpecificNominalObjectLabelDomainCarrier(NominalObjectLabelDomainCarrier):
    pass


class SpecificNominalObjectLabelDomainStrategy(ObjectLabelDomainMetadataStrategy):
    value_type = SpecificNominalObjectLabelDomainCarrier

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        return ObjectLabelDomain(declared_object_ids=(8,))


class StructuralObjectLabelDomainLookalike:
    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(declared_object_count=99)


def test_measurement_table_slice_offset_reinfers_projected_row_schema():
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=(
            {"ObjectNumber": 1, "AreaShape_Area": 10.0},
            {
                "object_name": "Nuclei",
                "feature_name": "AreaShape_Area",
                "value": 10.0,
            },
        ),
        fields=(FieldSpec("ObjectNumber"), FieldSpec("AreaShape_Area")),
        validated_runtime_schema=True,
    )

    assert table.validated_runtime_schema is True

    shifted = RuntimeSliceProjection.measurement_table_with_slice_offset(table, 1)

    assert tuple(field.name for field in shifted.fields) == (
        "slice_index",
        "ObjectNumber",
        "AreaShape_Area",
    )
    assert shifted.validated_runtime_schema is False
    assert shifted.rows[0]["ObjectNumber"] == 1
    assert shifted.rows[1]["feature_name"] == "AreaShape_Area"
    assert {row["slice_index"] for row in shifted.rows} == {1}


def test_object_label_dense_data_uses_nominal_payload_registry() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(labels=labels)
    label_set = ObjectLabelSet(name="Cells", labels=labels)

    assert ObjectLabelDenseDataStrategy.for_payload(payload).data(payload) is labels
    assert ObjectLabelDenseDataStrategy.for_payload(label_set).data(label_set) is labels
    assert ObjectLabelDenseDataStrategy.for_payload(labels).data(labels) is labels
    assert object_label_dense_array(payload, dtype=np.int32).dtype == np.int32


def test_object_label_set_from_payload_uses_domain_and_context_authorities() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    unedited = np.array([[0, 1], [1, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited,
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_path="/payload/image.tif",
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(5, 6),
            source_shape_yx=(20, 30),
        ),
    )

    label_set = ObjectLabelSet(
        name="Cells",
        labels=payload,
        domain=ObjectLabelDomain(declared_object_count=9),
        source_path="/explicit/image.tif",
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(40, 50)),
    )

    assert label_set.labels is labels
    assert label_set.unedited_labels is unedited
    assert label_set.domain.declared_object_count == 9
    assert label_set.domain.declared_object_ids == (1, 2)
    assert label_set.domain.scope is ObjectLabelDomainScope.PLANE
    assert label_set.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert label_set.source_path == "/explicit/image.tif"
    assert label_set.spatial_origin_yx == (5, 6)
    assert label_set.source_spatial_shape_yx == (40, 50)


def test_sparse_ijv_object_label_dense_data_uses_source_shape() -> None:
    sparse_rows = SparseIJVLabelRows(np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32))
    label_set = ObjectLabelSet(
        name="Cells",
        labels=sparse_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(5, 6)),
    )

    dense = object_label_dense_array(label_set, dtype=np.int32)

    assert dense.shape == (5, 6)
    assert dense[0, 1] == 2
    assert dense[2, 3] == 4


def test_sparse_ijv_object_label_dense_data_preserves_runtime_slices() -> None:
    sparse_rows = SparseIJVLabelRows(
        np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)
    )
    payload = ObjectLabelPayload(
        labels=sparse_rows,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(6, 7)),
    )

    dense = object_label_dense_array(payload, dtype=np.int32)

    assert dense.shape == (3, 6, 7)
    assert dense[0, 1, 2] == 3
    assert dense[2, 3, 4] == 5


def test_object_label_domain_preservation_uses_nominal_metadata_contract() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    source = NominalObjectLabelDomainCarrier(
        ObjectLabelDomain(declared_object_ids=(4, 7))
    )

    rebuilt = object_label_value_with_dense_labels(source, labels)

    assert rebuilt.domain.declared_object_ids == (4, 7)


def test_object_label_domain_preservation_uses_mro_specific_provider() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    source = SpecificNominalObjectLabelDomainCarrier(
        ObjectLabelDomain(declared_object_ids=(4, 7))
    )

    rebuilt = object_label_value_with_dense_labels(source, labels)

    assert rebuilt.domain.declared_object_ids == (8,)


def test_object_label_domain_preservation_rejects_structural_lookalikes() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)

    rebuilt = object_label_value_with_dense_labels(
        StructuralObjectLabelDomainLookalike(),
        labels,
    )

    assert rebuilt.domain.declared_object_count is None
    assert rebuilt.domain.declared_object_ids == ()


def test_object_label_pure_2d_aggregator_preserves_dense_payload_domains() -> None:
    first = ObjectLabelPayload(
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(1,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    second = ObjectLabelPayload(
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(2,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    np.testing.assert_array_equal(
        aggregated.labels,
        np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
    )
    assert aggregated.domain.declared_object_id_domains == ((1,), (2,))


def test_object_label_pure_2d_aggregator_accepts_empty_declared_planes() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (payload,),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert aggregated.domain.declared_object_id_domains == ((1,), ())


def test_object_label_pure_2d_aggregator_preserves_slice_source_paths() -> None:
    first = ObjectLabelPayload(
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        source_path="/input/A01_s001_w1_z001_t001.TIF",
    )
    second = ObjectLabelPayload(
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        source_path="/input/A01_s002_w1_z001_t001.TIF",
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.source_path is None
    assert aggregated.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )


def test_object_label_pure_2d_aggregator_declares_source_binding_plane_axis() -> None:
    first = ObjectLabelPayload(
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        source_path="/input/rawDNA.tif",
    )
    second = ObjectLabelPayload(
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        source_path="/input/rawGFP.tif",
    )
    first_set = ObjectLabelSet(
        name="Nuclei",
        labels=first.labels,
        source_path=first.source_path,
        source_image_name="rawDNA",
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        labels=second.labels,
        source_path=second.source_path,
        source_image_name="rawGFP",
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert aggregated.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert aggregated.source_image_name == "rawDNA"
    assert aggregated.source_image_provenance_planes.paths == (
        "/input/rawDNA.tif",
        "/input/rawGFP.tif",
    )


def test_source_image_object_label_build_treats_repeated_source_names_as_runtime_slices() -> (
    None
):
    image = RuntimeImagePayloadContext(
        np.zeros((2, 5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s1_w1.tif",
                    "/input/A01_s2_w1.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            ),
            source_image_names=("OrigHoechst", "OrigHoechst"),
        ),
        mask=None,
    ).payload()
    labels = np.zeros((2, 5, 6), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(image=image, labels=labels).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PLANE
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.source_image_names == ("OrigHoechst", "OrigHoechst")


def test_source_image_object_label_payload_accepts_declared_representation() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    labels = SparseIJVLabelRows(np.zeros((0, 3), dtype=np.int32))

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
    ).payload(representation=ObjectLabelRepresentation.SPARSE_IJV)

    assert payload.representation is ObjectLabelRepresentation.SPARSE_IJV


def test_source_image_object_label_build_keeps_distinct_source_names_as_bindings() -> (
    None
):
    image = RuntimeImagePayloadContext(
        np.zeros((2, 5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s1_w1.tif",
                    "/input/A01_s1_w2.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            ),
            source_image_names=("OrigHoechst", "OrigER"),
        ),
        mask=None,
    ).payload()
    labels = np.zeros((2, 5, 6), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(image=image, labels=labels).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PLANE
    assert payload.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert payload.source_image_names == ("OrigHoechst", "OrigER")


def test_object_label_pure_2d_aggregator_prefers_explicit_source_plane_aliases() -> (
    None
):
    first_set = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawGFP",),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert aggregated.source_image_names == ("rawDNA", "rawGFP")
    assert aggregated.source_image_name == "CellProfilerInternalImage"


def test_object_label_pure_2d_aggregator_does_not_force_duplicate_source_aliases() -> (
    None
):
    first_set = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert aggregated.source_image_names == ("rawDNA", "rawDNA")


def test_object_label_projected_plane_reduces_domain_to_payload_scope() -> None:
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("rawDNA", "rawGFP"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = labels.with_projected_plane(labels.labels[1], 1)

    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (2,)
    assert projected.domain.declared_object_id_domains == ()
    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected.source_image_names == ("rawGFP",)


def test_object_label_projected_plane_promotes_channel_provenance_to_scalar() -> None:
    labels = ObjectLabelPayload(
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s1_w1.tif",
                "/input/A01_s2_w1.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        ),
        source_image_names=("OrigHoechst", "OrigHoechst"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = labels.with_projected_plane(labels.labels[1], 1)
    source_image = RuntimeImagePayloadContext(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s1_w1.tif",
                    "/input/A01_s2_w1.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            ),
            source_image_names=("OrigHoechst", "OrigHoechst"),
        ),
        mask=None,
    ).payload()
    enriched = projected.with_source_image_context(source_image)

    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (2,)
    assert projected.source_path == "/input/A01_s2_w1.tif"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.source_image_provenance_planes.component_metadata == ()
    assert projected.source_image_names == ("OrigHoechst",)
    assert enriched.source_path == "/input/A01_s2_w1.tif"
    assert dict(enriched.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }
    assert enriched.source_image_provenance_planes.paths == ()
    assert enriched.source_image_provenance_planes.component_metadata == ()
    assert enriched.source_image_names == ("OrigHoechst",)


def test_source_backed_object_label_stack_declares_plane_count_from_provenance() -> (
    None
):
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s1.tif", "/input/A01_s2.tif")
        ),
    )

    plane_count = ObjectLabelSetPlaneStackContract().value_plane_count(labels)

    assert plane_count == 2


def test_object_label_measurement_replacement_projects_matching_plane_domain() -> None:
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
                [[0, 3], [0, 0]],
            ],
            dtype=np.int32,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("rawDNA", "rawGFP", "rawFarRed"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = ObjectLabelMeasurementPayloadStrategy.for_source(labels).materialize(
        labels,
        ObjectLabelReplacementRequest(labels.labels[2]),
    )

    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (3,)
    assert projected.domain.declared_object_id_domains == ()
    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected.source_image_names == ("rawFarRed",)


def test_object_label_derived_plane_uses_replacement_domain_and_source_provenance() -> (
    None
):
    source = ObjectLabelPayload(
        labels=np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 9], [0, 0]],
            ],
            dtype=np.int32,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/rawDNA.tif", "/input/rawActin.tif"),
            component_metadata=(
                {"channel": "DNA", "site": "1"},
                {"channel": "Actin", "site": "1"},
            ),
        ),
        source_image_names=("rawDNA", "rawActin"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (9,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    derived = np.asarray([[0, 4], [0, 0]], dtype=np.int32)

    projected = ObjectLabelMeasurementPayloadStrategy.for_source(
        source,
    ).materialize(
        source,
        ObjectLabelDerivedPlaneProjectionRequest(derived, 1),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (4,)
    assert projected.domain.declared_object_id_domains == ()
    assert projected.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projected.source_path == "/input/rawActin.tif"
    assert dict(projected.source_component_metadata) == {
        "channel": "Actin",
        "site": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.source_image_names == ("rawActin",)
    np.testing.assert_array_equal(projected.labels, derived)


def test_object_label_pure_2d_aggregator_preserves_sparse_ijv_sets() -> None:
    first = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    second = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(aggregated.labels, SparseIJVLabelRows)
    assert aggregated.labels.has_slice_index
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_object_label_pure_2d_aggregator_preserves_sparse_ijv_payloads() -> None:
    first = ObjectLabelPayload(
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    second = ObjectLabelPayload(
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(aggregated.labels, SparseIJVLabelRows)
    assert aggregated.labels.has_slice_index
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_empty_sparse_ijv_stack_preserves_runtime_slice_count() -> None:
    shape = (4, 5)
    empty = np.zeros(shape, dtype=np.int32)
    source_domain = SourceSpatialDomain(source_shape_yx=shape)
    first = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(empty),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_count=0),
        source_spatial_domain=source_domain,
    )
    second = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(empty),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_count=0),
        source_spatial_domain=source_domain,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert isinstance(aggregated.labels, SparseIJVLabelRows)
    assert aggregated.labels.label_data_runtime_slice_count() == 2
    assert RuntimeSliceProjection.slice_count_from_values((aggregated,)) == 2
    np.testing.assert_array_equal(
        object_label_dense_array(aggregated),
        np.zeros((2, *shape), dtype=np.int32),
    )

    projected_items = (
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=aggregated,
            source_description="empty sparse labels",
        )
    )

    assert len(projected_items) == 2
    assert [item.runtime_plane_metadata.plane_indices for item in projected_items] == [
        (0,),
        (1,),
    ]
    assert [object_label_dense_array(item.value).shape for item in projected_items] == [
        shape,
        shape,
    ]


def test_shape_object_feature_table_uses_registered_nominal_contract() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value: np.asarray(
                [10.0]
            ),
            MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS.value: np.asarray(
                [2.0]
            ),
        },
        measured_object_ids=(2,),
        object_domain=(1, 2),
    )

    assert issubclass(ShapeObjectFeatureValueTable, ObjectFeatureValueTable)
    assert (
        ShapeObjectFeatureValueTable
        in ObjectFeatureValueTable.registered_strategy_types()
    )

    rows = table.rows()
    assert rows[0]["object_label"] == 1
    assert np.isnan(rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value])
    assert rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS.value] == 0.0
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z.value] == 0.0
    )
    assert rows[1]["object_label"] == 2
    assert rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value] == 10.0
    assert (
        rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS.value]
        == 2.0
    )


def test_shape_object_feature_table_rejects_undeclared_dense_feature_domain() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            "UndeclaredDenseFeature": np.asarray([0.1, 0.2, 0.3]),
        },
        measured_object_ids=(1, 3),
        object_domain=(1, 2, 3),
    )

    with pytest.raises(ValueError, match="feature-array domain"):
        table.rows()


def test_shape_descriptor_row_ordinal_domain_is_registered_nominally() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value: np.asarray(
                [0.0, 20.0],
            ),
            "Zernike_0_0": np.asarray([0.1, 0.2]),
        },
        measured_object_ids=(1, 3),
        object_domain=(1, 2, 3),
    )

    assert (
        ObjectFeatureArrayDomainStrategy.for_enum_member(
            ObjectFeatureArrayDomain.ROW_ORDINAL
        ).domain
        is ObjectFeatureArrayDomain.ROW_ORDINAL
    )
    assert (
        ObjectFeatureMissingValueStrategy.for_enum_member(
            ObjectFeatureMissingValue.ZERO
        ).missing_value
        is ObjectFeatureMissingValue.ZERO
    )

    rows = table.rows()
    assert (
        rows[1][
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value
        ]
        == 0.0
    )
    assert (
        rows[2][
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value
        ]
        == 20.0
    )
    assert rows[1]["Zernike_0_0"] == 0.2
    assert np.isnan(rows[2]["Zernike_0_0"])


def test_shape_center_features_align_to_measured_sparse_object_ids() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value: np.asarray(
                [14.0]
            ),
            MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y.value: np.asarray(
                [21.0]
            ),
        },
        measured_object_ids=(892,),
        object_domain=(1, 892),
    )

    rows = table.rows()

    assert np.isnan(
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value]
    )
    assert (
        rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value] == 14.0
    )
    assert (
        rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y.value] == 21.0
    )


def test_sparse_shape_measurement_preserves_high_object_id_feature_domain() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows(
            np.asarray(
                (
                    (1, 2, 892),
                    (1, 3, 892),
                    (2, 2, 892),
                    (2, 3, 892),
                ),
                dtype=np.int32,
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_ids=(892,),
        ),
    )

    rows = ObjectSizeShapeMeasurementRowsRequest(
        labels=labels,
        calculate_advanced=False,
        calculate_zernikes=True,
        shape_backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        zernike_backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        regionprops_backend_provider=AnalysisBackendProvider.NUMBA,
    ).rows()

    assert len(rows) == 1
    assert rows[0][MeasurementRowAxisField.OBJECT_LABEL.value] == 892
    assert rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value] == 4.0
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value] == 2.5
    )
    assert (
        ShapeZernikeFeatureAuthority.shape_zernike_feature_name(degree=0, repetition=0)
        in rows[0]
    )


def test_object_label_payload_builder_uses_nominal_payload_registry() -> None:
    source_labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    transformed_labels = np.array([[0, 2], [1, 0]], dtype=np.float32)
    payload = ObjectLabelPayload(
        labels=source_labels,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(3, 5),
            source_shape_yx=(20, 30),
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
        ),
    )

    rebuilt = object_label_value_with_dense_labels(
        payload,
        transformed_labels,
        domain_declaration=ExplicitObjectLabelDomainDeclaration(
            ObjectLabelDomain(
                declared_object_count=1,
                declared_object_ids=(2,),
            )
        ),
    )

    assert isinstance(
        ObjectLabelValueBuilderStrategy.for_source(payload),
        ObjectLabelValueBuilderStrategy,
    )
    assert rebuilt.labels is transformed_labels
    assert rebuilt.domain.declared_object_count == 1
    assert rebuilt.domain.declared_object_ids == (2,)
    assert rebuilt.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert rebuilt.spatial_origin_yx == (3, 5)
    assert rebuilt.source_spatial_shape_yx == (20, 30)


def test_object_label_payload_from_source_image_uses_image_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 3),
                source_shape_yx=(10, 12),
            ),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.TIF",
                    "/input/A01_s002_w1_z001_t001.TIF",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            ),
        ),
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        declared_object_count=2,
    ).payload()

    assert payload.labels is labels
    assert payload.domain.declared_object_count == 2
    assert payload.spatial_origin_yx == (2, 3)
    assert payload.source_spatial_shape_yx == (10, 12)
    assert payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )
    assert tuple(
        dict(item) for item in payload.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )


def test_composed_image_metadata_preserves_single_channel_source_context() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.TIF",),
                component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
            )
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s002_w1_z001_t001.TIF",),
                component_metadata=({"well": "A01", "site": "2", "channel": "1"},),
            )
        ),
        mask=None,
    ).payload()

    metadata = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )
    assert metadata.for_source_plane(1).source_path == (
        "/input/A01_s002_w1_z001_t001.TIF"
    )
    assert dict(metadata.for_source_plane(1).source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }


def test_composed_image_metadata_preserves_transformed_scalar_image_type() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "grayscale image",
                "channel": "1",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.JPG",),
                component_metadata=(
                    {
                        SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image",
                        "site": "1",
                    },
                ),
            ),
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "grayscale image",
                "channel": "1",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s002_w1_z001_t001.JPG",),
                component_metadata=(
                    {
                        SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image",
                        "site": "2",
                    },
                ),
            ),
        ),
        mask=None,
    ).payload()

    metadata = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert dict(metadata.source_component_metadata) == {
        SOURCE_IMAGE_TYPE_METADATA_FIELD: "grayscale image",
        "channel": "1",
    }
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image", "site": "1", "channel": "1"},
        {SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image", "site": "2", "channel": "1"},
    )


def test_source_image_payload_role_uses_transformed_scalar_role_before_provenance() -> (
    None
):
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "grayscale image",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image"},
                    {SOURCE_IMAGE_TYPE_METADATA_FIELD: "color image"},
                ),
            ),
        ),
        mask=None,
    ).payload()

    role = source_image_payload_role(payload)

    assert role is not None
    assert role.image_type() == "grayscale image"


def test_bundle_image_metadata_keeps_agreed_present_component_values() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "timepoint": "7",
            },
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
        ),
        mask=None,
    ).payload()

    bundle = ImagePayloadMetadataCompositionRequest(
        (first, second),
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    ).metadata()
    stack = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert dict(bundle.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
        "timepoint": "7",
    }
    assert dict(stack.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }


def test_bundle_image_metadata_preserves_payload_source_provenance() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.tif",),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                    },
                ),
            )
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w2_z001_t001.tif",),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                    },
                ),
            )
        ),
        mask=None,
    ).payload()

    bundle = ImagePayloadMetadataCompositionRequest(
        (first, second),
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    ).metadata()
    stack = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert bundle.source_image_provenance_planes.count == 2
    assert tuple(
        plane.source_identity.path
        for plane in bundle.source_image_provenance_planes.planes
    ) == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s001_w2_z001_t001.tif",
    )
    assert dict(bundle.source_component_metadata) == {
        "extension": ".tif",
        "well": "A01",
        "site": "1",
        "z_index": "1",
    }
    assert stack.source_image_provenance_planes.count == 2


def test_object_label_payload_from_composed_source_image_keeps_site_axis_metadata() -> (
    None
):
    source_slices = (
        RuntimeImagePayloadContext(
            np.zeros((4, 5), dtype=np.float32),
            metadata=ImagePayloadMetadata(
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/input/A01_s001_w1_z001_t001.TIF",),
                    component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
                )
            ),
            mask=None,
        ).payload(),
        RuntimeImagePayloadContext(
            np.ones((4, 5), dtype=np.float32),
            metadata=ImagePayloadMetadata(
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/input/A01_s002_w1_z001_t001.TIF",),
                    component_metadata=({"well": "A01", "site": "2", "channel": "1"},),
                )
            ),
            mask=None,
        ).payload(),
    )
    image = RuntimeImagePayloadContext(
        np.stack(tuple(image_payload_data(payload) for payload in source_slices)),
        metadata=ImagePayloadMetadataCompositionRequest(source_slices).metadata(),
        mask=None,
    ).payload()
    labels = np.ones((2, 4, 5), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
    ).payload()

    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )
    assert tuple(
        dict(item) for item in payload.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )


def test_object_label_payload_from_source_image_keeps_ambiguous_3d_domain_payload_scoped() -> (
    None
):
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )
    labels = np.array(
        [
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 3, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=np.int32,
    )

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        declared_object_count=4,
    ).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.domain.declared_object_count == 4
    assert payload.domain.declared_object_id_domains == ()


def test_object_label_payload_from_source_image_declares_source_binding_plane_domain() -> (
    None
):
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("rawDNA.tif", "rawGFP.tif")
            )
        ),
    )
    labels = np.array(
        [
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 3, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=np.int32,
    )

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
    ).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PLANE
    assert payload.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert payload.domain.declared_object_count is None
    assert payload.domain.declared_object_id_domains == ((1, 2), (3, 4))


def test_object_label_payload_domain_declaration_overrides_source_binding_planes() -> (
    None
):
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("z0.tif", "z1.tif")
            )
        ),
    )
    labels = np.array(
        [
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 3, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=np.int32,
    )

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
        declared_object_count=4,
    ).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.domain.declared_object_count == 4
    assert payload.domain.declared_object_id_domains == ()


def test_source_image_plane_axis_request_uses_channel_provenance() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("rawDNA.tif", "rawGFP.tif")
            )
        ),
    )

    assert (
        SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(image)
        ).axis()
        is RuntimePlaneAxis.SOURCE_BINDING
    )


def test_source_image_plane_axis_request_uses_channel_component_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            )
        ),
    )

    assert (
        SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(image)
        ).axis()
        is RuntimePlaneAxis.SOURCE_BINDING
    )


def test_source_image_plane_axis_request_uses_site_component_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            )
        ),
    )
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    assert (
        SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(image)
        ).axis()
        is RuntimePlaneAxis.RUNTIME_SLICE
    )
    assert (
        SourceImageObjectLabelBuildRequest(image=image, labels=labels)
        .payload()
        .plane_axis
        is RuntimePlaneAxis.RUNTIME_SLICE
    )


def test_object_label_set_from_source_image_uses_image_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 3),
                source_shape_yx=(10, 12),
            ),
            source_path="/input/A01_s001_w1_z001_t001.TIF",
            source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
        ),
    )
    sparse_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))

    label_set = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=sparse_rows,
        declared_object_count=2,
    ).label_set(
        name="OverlappingWorms",
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    assert label_set.labels is sparse_rows
    assert label_set.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert label_set.domain.declared_object_count == 2
    assert label_set.spatial_origin_yx == (2, 3)
    assert label_set.source_spatial_shape_yx == (10, 12)
    assert label_set.source_path == "/input/A01_s001_w1_z001_t001.TIF"
    assert dict(label_set.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }


def test_object_label_set_source_image_context_fills_missing_stack_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A02_s001_w1_z001_t001.tif",
                    "/input/A02_s002_w1_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A02", "site": 1, "channel": 1},
                    {"well": "A02", "site": 2, "channel": 1},
                ),
            )
        ),
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((2, 2, 2), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )

    contextualized = labels.with_source_image_context(image)

    assert contextualized.name == "Nuclei"
    assert contextualized.domain.declared_object_count == 3
    assert contextualized.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in contextualized.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_object_label_payload_source_image_context_fills_partial_stack_metadata() -> (
    None
):
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A02_s001_w1_z001_t001.tif",
                    "/input/A02_s002_w1_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A02", "site": 1, "channel": 1},
                    {"well": "A02", "site": 2, "channel": 1},
                ),
            )
        ),
    )
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 4, 5), dtype=np.int32),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(None, None), component_metadata=(None, None)
        ),
    )

    contextualized = labels.with_source_image_context(image)

    assert contextualized.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in contextualized.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_source_image_loading_semantics_attaches_component_metadata() -> None:
    image = np.zeros((4, 5), dtype=np.uint16)

    payload = SourceImagePayloadSemantics.from_source_metadata(
        {"well": "01", "site": "POS002", "channel": "D"},
        "01_POS002_D.TIF",
    ).apply(image)

    metadata = image_payload_metadata(payload)
    assert metadata.source_path == "01_POS002_D.TIF"
    assert dict(metadata.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }


def test_object_label_source_image_semantics_treats_rgb_image_as_label_plane() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0:2, 0:2] = (255, 0, 0)
    image[2:4, 3:5] = (0, 255, 0)

    payload = SourceImagePayloadSemantics.from_source_metadata(
        {SOURCE_IMAGE_TYPE_METADATA_FIELD: "Objects"},
        "objects.png",
    ).apply(image)

    labels = image_payload_data(payload)
    assert labels.shape == (4, 5)
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert image_payload_metadata(payload).source_path == "objects.png"


def test_object_label_set_replacement_preserves_sparse_ijv_representation() -> None:
    source_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))
    replacement_rows = SparseIJVLabelRows(np.array([[0, 1, 1]], dtype=np.int32))
    source = ObjectLabelSet(
        name="OverlappingWorms",
        labels=source_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    replacement = ObjectLabelSet(
        name="OverlappingWorms",
        labels=replacement_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    rebuilt = source.with_labels(
        ObjectLabelSetReplacementStrategy.for_source(source).replacement_labels(
            replacement
        )
    )

    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert rebuilt.labels is replacement_rows


def test_object_label_payload_rebuild_preserves_sparse_ijv_representation() -> None:
    source_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))
    replacement_rows = SparseIJVLabelRows(np.array([[0, 1, 1]], dtype=np.int32))
    source = ObjectLabelPayload(
        labels=source_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    rebuilt = source.with_labels(replacement_rows)

    assert isinstance(rebuilt, ObjectLabelPayload)
    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert rebuilt.labels is replacement_rows


def test_object_label_payload_source_context_preserves_sparse_ijv_representation() -> (
    None
):
    image = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(),
        mask=None,
    ).payload()
    sparse_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))
    source = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=sparse_rows,
    ).payload(representation=ObjectLabelRepresentation.SPARSE_IJV)

    contextualized = source.with_source_image_context(image)

    assert contextualized.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert contextualized.labels is sparse_rows


def test_sparse_ijv_object_label_replacement_converts_dense_labels() -> None:
    source = ObjectLabelSet(
        name="OverlappingWorms",
        labels=SparseIJVLabelRows(np.array([[0, 0, 1]], dtype=np.int32)),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    dense_replacement = np.array([[0, 2], [3, 0]], dtype=np.int32)

    rebuilt = source.with_labels(
        ObjectLabelSetReplacementStrategy.for_source(source).replacement_labels(
            dense_replacement
        )
    )

    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(rebuilt.labels, SparseIJVLabelRows)
    np.testing.assert_array_equal(
        rebuilt.labels.as_array(),
        np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32),
    )


def test_object_label_payload_with_measurement_labels_preserves_domain_and_variants() -> (
    None
):
    labels = np.zeros((1, 2, 2), dtype=np.int32)
    unedited = np.ones_like(labels)
    small_removed = np.full_like(labels, 2)
    payload = ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited,
        small_removed_labels=small_removed,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(4, 5),
            source_shape_yx=(10, 11),
        ),
        domain=ObjectLabelDomain(
            declared_object_count=2,
            declared_object_ids=(1, 2),
        ),
    )
    selected = labels[0]

    rebuilt = ObjectLabelMeasurementPayloadStrategy.for_source(payload).materialize(
        payload,
        ObjectLabelReplacementRequest(selected),
    )

    assert isinstance(
        ObjectLabelMeasurementPayloadStrategy.for_source(payload),
        ObjectLabelMeasurementPayloadStrategy,
    )
    assert isinstance(rebuilt, ObjectLabelPayload)
    assert rebuilt.labels is selected
    assert rebuilt.unedited_labels is None
    assert rebuilt.small_removed_labels is None
    assert rebuilt.domain.declared_object_count == 2
    assert rebuilt.domain.declared_object_ids == (1, 2)
    assert rebuilt.spatial_origin_yx == (4, 5)
    assert rebuilt.source_spatial_shape_yx == (10, 11)


def test_object_label_variant_compatibility_uses_nominal_registry() -> None:
    variant = np.ones((1, 2, 2), dtype=np.int32)
    matching_labels = np.zeros((1, 2, 2), dtype=np.int32)
    selected_labels = matching_labels[0]

    assert isinstance(
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant),
        ObjectLabelVariantCompatibilityStrategy,
    )
    assert (
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant).matching_labels(
            variant,
            matching_labels,
        )
        is variant
    )
    assert (
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant).matching_labels(
            variant,
            selected_labels,
        )
        is None
    )


def test_singleton_object_label_stack_collapse_uses_nominal_registry() -> None:
    labels = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    payload = ObjectLabelPayload(
        labels=labels,
        unedited_labels=labels.copy(),
        small_removed_labels=labels.copy(),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(5, 6),
        ),
    )

    collapsed_array = SingletonObjectLabelStackCollapseStrategy.for_labels(
        labels
    ).collapse(labels)
    collapsed_payload = SingletonObjectLabelStackCollapseStrategy.for_labels(
        payload
    ).collapse(payload)

    assert isinstance(
        SingletonObjectLabelStackCollapseStrategy.for_labels(labels),
        SingletonObjectLabelStackCollapseStrategy,
    )
    np.testing.assert_array_equal(collapsed_array, labels[0])
    assert isinstance(collapsed_payload, ObjectLabelPayload)
    np.testing.assert_array_equal(collapsed_payload.labels, labels[0])
    np.testing.assert_array_equal(collapsed_payload.unedited_labels, labels[0])
    np.testing.assert_array_equal(collapsed_payload.small_removed_labels, labels[0])
    assert collapsed_payload.spatial_origin_yx == (1, 2)
    assert collapsed_payload.source_spatial_shape_yx == (5, 6)


def test_runtime_projection_requirement_projects_singleton_object_label_stack() -> None:
    labels = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01", "site": 1, "channel": 1},),
        ),
    )

    (item,) = (
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description="singleton labels",
        )
    )

    assert isinstance(item.value, ObjectLabelPayload)
    np.testing.assert_array_equal(item.value.labels, labels[0])
    assert item.source_component_metadata == {"well": "A01", "site": 1, "channel": 1}
    assert item.runtime_plane_metadata is None


def test_runtime_projection_requirement_tracks_multi_plane_runtime_coordinates() -> (
    None
):
    labels = np.arange(50, dtype=np.int32).reshape(2, 5, 5)
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"well": "A01", "site": 1, "channel": 1, "z_index": 1},
                {"well": "A01", "site": 1, "channel": 1, "z_index": 2},
            ),
        ),
    )

    first, second = (
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description="labels",
        )
    )

    assert first.runtime_plane_metadata is not None
    assert second.runtime_plane_metadata is not None
    assert first.runtime_plane_metadata.roi_metadata == {
        "plane_indices": (0,),
        "plane_shape": (2,),
    }
    assert second.runtime_plane_metadata.roi_metadata == {
        "plane_indices": (1,),
        "plane_shape": (2,),
    }
    assert first.runtime_plane_metadata.source_plane_indices == (0,)
    assert second.runtime_plane_metadata.source_plane_indices == (1,)
    assert first.source_component_metadata == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
    }
    assert second.source_component_metadata == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 2,
    }


def test_runtime_projection_expands_indexed_scalar_source_metadata() -> None:
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    payload = RuntimeImagePayloadContext(
        np.arange(12, dtype=np.uint16).reshape(3, 2, 2),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "1",
                SOURCE_PLANE_INDEX_FIELD: "0",
                SOURCE_PLANE_COUNT_FIELD: "3",
            },
        ),
    ).payload()

    projected = (
        RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description=source_path,
        )
    )

    assert len(projected) == 3
    assert [item.runtime_plane_metadata.source_plane_indices for item in projected] == [
        (0,),
        (1,),
        (2,),
    ]
    assert [item.source_component_metadata["z_index"] for item in projected] == [
        "1",
        "2",
        "3",
    ]
    assert [
        item.source_component_metadata[SOURCE_PLANE_INDEX_FIELD] for item in projected
    ] == ["0", "1", "2"]


def test_runtime_projection_rejects_unindexed_scalar_volume_source_metadata() -> None:
    source_path = "/input/A01_s001_w1_z001_t001.tif"
    payload = RuntimeImagePayloadContext(
        np.arange(8, dtype=np.uint16).reshape(2, 2, 2),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
            },
        ),
    ).payload()

    with pytest.raises(
        ValueError,
        match="requires complete per-slice component metadata",
    ):
        (
            RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
        ).project_payload_items(
            RuntimeProjectionSourceIdentityRequest(
                value=payload,
                source_description=source_path,
            )
        )


def test_dense_object_label_slice_stack_projects_payload_labels() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(labels=labels)

    stack = DenseObjectLabelSliceStackRequest(
        payload,
        slice_count=3,
        dtype=np.int32,
    ).stack()

    assert stack is not None
    assert stack.labels.shape == (3, 2, 2)
    assert stack.labels.dtype == np.int32
    np.testing.assert_array_equal(stack.slice(2), labels)


def test_dense_object_label_slice_stack_preserves_projected_payload_domain() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 2]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2, 3)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    stack = DenseObjectLabelSliceStackRequest(
        payload,
        slice_count=2,
        dtype=np.int32,
    ).stack()

    assert stack is not None
    sliced = stack.slice(1)
    assert isinstance(sliced, ObjectLabelPayload)
    np.testing.assert_array_equal(sliced.labels, labels[1])
    assert sliced.domain.declared_object_ids == (1, 2, 3)


def test_dense_object_label_slice_stack_groups_interleaved_label_planes() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 2], [0, 0]],
            [[0, 0], [3, 0]],
            [[0, 0], [0, 4]],
        ],
        dtype=np.int32,
    )

    stack = DenseObjectLabelSliceStackRequest(labels, slice_count=2).stack()

    assert stack is not None
    np.testing.assert_array_equal(
        stack.slice(0),
        np.array([[1, 0], [3, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        stack.slice(1),
        np.array([[0, 2], [0, 4]], dtype=np.int32),
    )


def test_object_label_value_runtime_slice_projection_owns_domain_and_labels() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 2], [0, 0]],
            [[0, 0], [3, 0]],
            [[0, 0], [0, 4]],
        ],
        dtype=np.int32,
    )
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,), (4,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = payload.with_runtime_slice_projection(
        slice_index=0,
        slice_count=2,
        plane_indices=(0, 2),
    )

    assert isinstance(projected, ObjectLabelPayload)
    np.testing.assert_array_equal(
        projected.labels,
        np.array([[1, 0], [3, 0]], dtype=np.int32),
    )
    assert projected.domain.declared_object_id_domains == ((1,), (3,))


def test_object_label_runtime_slice_projection_keeps_volume_plane_domain() -> None:
    labels = np.zeros((2, 3, 4, 4), dtype=np.int32)
    labels[0, :, 0:2, 0:2] = 1
    labels[1, :, 1:4, 1:4] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(projected, ObjectLabelPayload)
    np.testing.assert_array_equal(projected.labels, labels[1])
    assert projected.domain.declared_object_ids == (2,)
    assert projected.domain.declared_object_id_domains == ()


def test_object_label_runtime_slice_projection_preserves_grouped_source_planes() -> (
    None
):
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 2], [0, 0]],
        ],
        dtype=np.int32,
    )
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "z_index": "1",
            "timepoint": "1",
        },
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s001_w2_z001_t001.TIF",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                    "timepoint": "1",
                },
            ),
        ),
    )

    projected = payload.with_runtime_slice_projection(
        slice_index=0,
        slice_count=1,
        plane_indices=(0, 1),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.domain.declared_object_id_domains == ((1,), (2,))
    assert projected.source_path is None
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "z_index": "1",
        "timepoint": "1",
    }
    assert projected.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s001_w2_z001_t001.TIF",
    )
    assert tuple(
        dict(item)
        for item in projected.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "2",
            "z_index": "1",
            "timepoint": "1",
        },
    )


def test_normalize_artifact_value_builds_key_schema_and_storage_policy():
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("DAPI",),
    )

    value = normalize_artifact_value(
        output_plan,
        [{"object_id": 1, "area": 12.0}],
        axis_id="A01",
    )

    assert value.name == "measurements"
    assert value.artifact_type is MeasurementsArtifactType
    assert value.key.scope.axis_id == "A01"
    assert value.key.scope.group_key == "DAPI"
    assert value.schema.artifact_type is MeasurementsArtifactType
    assert value.storage == RuntimeStoragePolicy(
        backend="memory",
        path="/memory/measurements.pkl",
        materialize=False,
    )


def test_normalize_artifact_value_aggregates_slice_aligned_object_label_domains():
    output_plan = ArtifactOutputPlan(
        name="GridObjects",
        path="/memory/GridObjects.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    first = ObjectLabelPayload(
        labels=np.array([[0, 1], [0, 3]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )
    second = ObjectLabelPayload(
        labels=np.array([[0, 2], [4, 0]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )

    value = normalize_artifact_value(
        output_plan,
        RuntimeSliceAlignedValues((first, second)),
        axis_id="A01",
    )
    payload = value.data

    assert isinstance(payload, ObjectLabelPayload)
    assert value.schema.slice_aligned is False
    assert payload.domain.declared_object_count == 4
    assert payload.domain.declared_object_id_domains == ((1, 2, 3, 4), (1, 2, 3, 4))
    assert payload.domain.scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        payload.labels,
        np.array(
            [
                [[0, 1], [0, 3]],
                [[0, 2], [4, 0]],
            ],
            dtype=np.int32,
        ),
    )


def test_normalize_artifact_value_preserves_slice_aligned_object_label_sources():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    first = ObjectLabelPayload(
        labels=np.array([[0, 1], [0, 0]], dtype=np.int32),
        source_path="/input/A02_s001_w1_z001_t001.tif",
        source_component_metadata={"well": "A02", "site": 1, "channel": 1},
    )
    second = ObjectLabelPayload(
        labels=np.array([[0, 2], [0, 0]], dtype=np.int32),
        source_path="/input/A02_s002_w1_z001_t001.tif",
        source_component_metadata={"well": "A02", "site": 2, "channel": 1},
    )

    value = normalize_artifact_value(
        output_plan,
        RuntimeSliceAlignedValues((first, second)),
        axis_id="A02",
    )
    payload = value.data

    assert isinstance(payload, ObjectLabelPayload)
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item) for item in payload.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )
    assert value.schema.source_provenance.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in (
            value.schema.source_provenance.source_image_provenance_planes.component_metadata
        )
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_normalize_artifact_value_rejects_metadata_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        artifact_type=MetadataArtifactType,
    )

    with pytest.raises(TypeError, match="expected metadata mapping"):
        normalize_artifact_value(output_plan, ["not", "metadata"], axis_id="A01")


def test_normalize_artifact_value_rejects_object_label_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    with pytest.raises(TypeError, match="expected object_labels payload"):
        normalize_artifact_value(output_plan, {"not": "labels"}, axis_id="A01")


def test_object_label_payload_validator_accepts_nominal_slice_aggregate():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    payload = ObjectLabelPayload(
        labels=np.array(
            [
                [[0, 1], [0, 2]],
                [[3, 0], [4, 0]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    value = RuntimeValue(
        key=ArtifactKey(
            name="nuclei",
            artifact_type=ObjectLabelsArtifactType,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=payload,
        schema=RuntimeValueSchema(
            artifact_type=ObjectLabelsArtifactType,
            slice_aligned=True,
            object_name="nuclei",
        ),
    )

    normalized = normalize_artifact_value(output_plan, value, axis_id="A01")

    assert normalized.data is payload


def test_spatial_grid_normalizes_to_mapping_runtime_value():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        artifact_type=SpatialGridArtifactType,
    )
    grid = SpatialGrid(
        name="Grid",
        rows=30,
        columns=30,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
    )

    value = normalize_artifact_value(output_plan, grid, axis_id="A01")

    assert value.artifact_type is SpatialGridArtifactType
    assert value.schema.artifact_type is SpatialGridArtifactType
    assert value.data["rows"] == 30
    assert value.data["x_location_of_lowest_x_spot"] == 27.0
    assert value.data["ordering"] == SpatialGridOrdering.BY_ROWS.value
    assert SpatialGrid.from_runtime_value(value) == grid


def test_slice_aligned_spatial_grid_normalizes_to_validated_mapping_sequence():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        artifact_type=SpatialGridArtifactType,
    )
    grids = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=2.0,
                y_origin=4.0,
            ),
        )
    )

    value = normalize_artifact_value(output_plan, grids, axis_id="A01")

    assert value.artifact_type is SpatialGridArtifactType
    assert value.schema.slice_aligned is True
    assert [grid["x_origin"] for grid in value.data] == [1.0, 2.0]


def test_spatial_grid_preserves_column_ordering():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        artifact_type=SpatialGridArtifactType,
    )
    grid = SpatialGrid(
        name="Grid",
        rows=2,
        columns=3,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
        ordering=SpatialGridOrdering.BY_COLUMNS,
    )

    value = normalize_artifact_value(output_plan, grid, axis_id="A01")

    assert value.data["ordering"] == SpatialGridOrdering.BY_COLUMNS.value
    assert (
        SpatialGrid.from_runtime_value(value).ordering is SpatialGridOrdering.BY_COLUMNS
    )


def test_normalize_artifact_value_accepts_object_label_arrays():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    value = normalize_artifact_value(output_plan, ArrayLike(), axis_id="A01")

    assert value.artifact_type is ObjectLabelsArtifactType


def test_normalize_artifact_value_accepts_registered_external_arrays():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.zeros((3, 3), dtype=np.uint16)

    value = normalize_artifact_value(output_plan, labels, axis_id="A01")

    assert value.data is labels
    assert value.artifact_type is ObjectLabelsArtifactType


def test_normalize_named_image_preserves_raw_payload_and_schema():
    output_plan = ArtifactOutputPlan(
        name="DNA",
        path="/memory/DNA.pkl",
        artifact_type=ImageArtifactType,
    )
    image = ArrayLike()

    value = normalize_artifact_value(
        output_plan,
        NamedImage(
            name="DNA",
            data=image,
            dimensions=("z", "y", "x"),
            source_image_name="raw_DNA",
        ),
        axis_id="A01",
    )

    assert value.data is image
    assert value.schema.artifact_type is ImageArtifactType
    assert value.schema.dimensions == ("z", "y", "x")
    assert value.schema.source_image_name == "raw_DNA"


def test_object_label_runtime_value_preserves_source_provenance_in_schema():
    output_plan = ArtifactOutputPlan(
        name="Mitochondria",
        path="/memory/Mitochondria.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Mitochondria",
            labels=labels,
            source_path="/input/A02_s001_w5_z001_t001.tif",
            source_component_metadata={
                "well": "A02",
                "site": "1",
                "channel": "5",
            },
        ),
        axis_id="A02",
    )
    restored = ObjectLabelSet.from_runtime_value(value)

    assert value.schema.source_path == "/input/A02_s001_w5_z001_t001.tif"
    assert value.schema.source_component_metadata == {
        "well": "A02",
        "site": "1",
        "channel": "5",
    }
    assert restored.source_path == "/input/A02_s001_w5_z001_t001.tif"
    assert restored.source_component_metadata == {
        "well": "A02",
        "site": "1",
        "channel": "5",
    }


def test_object_label_set_preserves_payload_parent_image_spacing() -> None:
    payload = ObjectLabelPayload(
        labels=np.array([[0, 1], [0, 0]], dtype=np.int32),
        parent_image_source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    )

    labels = ObjectLabelSet(name="Cells", labels=payload)
    runtime_payload = labels.runtime_payload()

    assert labels.parent_image_source_voxel_spacing == SourceVoxelSpacing(
        (2.0, 1.0, 1.0)
    )
    assert isinstance(runtime_payload, ObjectLabelPayload)
    assert runtime_payload.parent_image_source_voxel_spacing == SourceVoxelSpacing(
        (2.0, 1.0, 1.0)
    )


def test_masked_image_payload_behaves_like_array_with_mask() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([[True, False, True], [True, True, False]])

    payload = MaskedImagePayload(data=image, mask=mask)

    assert payload.shape == image.shape
    assert payload.ndim == 2
    assert payload.dtype == image.dtype
    np.testing.assert_array_equal(np.asarray(payload), image)
    np.testing.assert_array_equal(payload.mask, mask)


def test_derived_image_payload_context_projects_bundle_mask_to_single_output() -> None:
    image = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, :, 0, 0] = False
    source = MaskedImagePayload(data=np.stack((image, image)), mask=mask)

    result = DerivedImagePayloadContext(source, image).payload()

    assert isinstance(result, MaskedImagePayload)
    np.testing.assert_array_equal(result.mask, np.all(mask, axis=0))


def test_masked_image_payload_flatten_uses_backing_array() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = MaskedImagePayload(data=image, mask=np.ones((2, 3), dtype=bool))

    np.testing.assert_array_equal(payload.flatten(), image.flatten())


def test_derived_image_payload_context_merges_source_provenance_into_output_metadata() -> (
    None
):
    source = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.tif",
                    "/input/A01_s002_w1_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            )
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    result = DerivedImagePayloadContext(source, output).payload()

    result_metadata = image_payload_metadata(result)
    assert result_metadata.source_dtype == "float32"
    assert result_metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in result_metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )


def test_derived_image_payload_context_replaces_stale_scalar_source_with_planes() -> (
    None
):
    source = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w5_z001_t001.tif",
                    "/input/A01_s002_w5_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "5"},
                    {"well": "A01", "site": "2", "channel": "5"},
                ),
            )
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w5_z001_t001.tif",
            source_component_metadata={"well": "A01", "site": "1", "channel": "5"},
        ),
    )

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_path is None
    assert metadata.source_component_metadata is None
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w5_z001_t001.tif",
        "/input/A01_s002_w5_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "5"},
        {"well": "A01", "site": "2", "channel": "5"},
    )


def test_derived_image_payload_context_replaces_stale_matching_plane_source_identity() -> (
    None
):
    source = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.tif",
                    "/input/A01_s002_w3_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "3"},
                    {"well": "A01", "site": "2", "channel": "3"},
                ),
            ),
            source_image_names=("OrigSyto",),
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.tif",
                    "/input/A01_s002_w1_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            ),
            source_image_names=("OrigHoechst",),
        ),
    )

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w3_z001_t001.tif",
        "/input/A01_s002_w3_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "3"},
        {"well": "A01", "site": "2", "channel": "3"},
    )
    assert metadata.source_image_names == ("OrigSyto",)


def test_derived_image_payload_context_replaces_stale_scalar_source_identity() -> None:
    source = ImageMetadataPayload(
        data=np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w3_z001_t001.tif",
            source_component_metadata={"well": "A01", "site": "1", "channel": "3"},
            source_image_names=("OrigSyto",),
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.tif",
            source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
            source_image_names=("OrigHoechst",),
        ),
    )

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_path == "/input/A01_s001_w3_z001_t001.tif"
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "3",
    }
    assert metadata.source_image_provenance_planes.paths == ()
    assert metadata.source_image_names == ("OrigSyto",)


def test_derived_image_payload_context_replaces_singleton_stack_scalar_source_identity() -> (
    None
):
    source = ImageMetadataPayload(
        data=np.zeros((4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/Sequence1_s001_w1_z001_t007.tif",
            source_component_metadata={
                "well": "Sequence1",
                "site": "1",
                "channel": "1",
                "timepoint": "7",
            },
            source_image_names=("OrigColor",),
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((1, 4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/Sequence1_s001_w1_z001.tif",
            source_component_metadata={
                "well": "Sequence1",
                "site": "1",
                "channel": "1",
            },
            source_image_names=("AdjacentImage",),
        ),
    )

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_path == "/input/Sequence1_s001_w1_z001_t007.tif"
    assert metadata.source_component_metadata == {
        "well": "Sequence1",
        "site": "1",
        "channel": "1",
        "timepoint": "7",
    }
    assert metadata.source_image_provenance_planes.paths == ()
    assert metadata.source_image_names == ("OrigColor",)


def test_derived_image_payload_context_expands_indexed_scalar_source_metadata_for_stack() -> (
    None
):
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    source = ImageMetadataPayload(
        data=np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "1",
                SOURCE_PLANE_INDEX_FIELD: "0",
                SOURCE_PLANE_COUNT_FIELD: "3",
            },
        ),
    )
    output = np.ones((3, 4, 5), dtype=np.float32)

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_image_provenance_planes.paths == (
        source_path,
        source_path,
        source_path,
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "3",
            SOURCE_PLANE_INDEX_FIELD: "2",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


def test_derived_image_payload_context_expands_indexed_scalar_output_metadata_for_stack() -> (
    None
):
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    output = ImageMetadataPayload(
        data=np.ones((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
                "z_index": "1",
                SOURCE_PLANE_INDEX_FIELD: "0",
                SOURCE_PLANE_COUNT_FIELD: "3",
            },
        ),
    )

    result = DerivedImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        output,
    ).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_image_provenance_planes.paths == (
        source_path,
        source_path,
        source_path,
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "3",
            SOURCE_PLANE_INDEX_FIELD: "2",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


def test_derived_image_payload_context_keeps_scalar_output_source_over_stack_source() -> (
    None
):
    source = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.tif",
                    "/input/A01_s001_w2_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            ),
            source_image_names=("Grayscale", "SmallBlockIllum"),
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.tif",
            source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
            source_image_names=("Grayscale",),
        ),
    )

    result = DerivedImagePayloadContext(source, output).payload()
    metadata = image_payload_metadata(result)

    assert metadata.source_path == "/input/A01_s001_w1_z001_t001.tif"
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    assert metadata.source_image_provenance_planes.paths == ()
    assert metadata.source_image_provenance_planes.component_metadata == ()
    assert metadata.source_image_names == ("Grayscale",)


def test_object_label_source_context_keeps_source_aligned_stack_planes() -> None:
    paths = tuple(f"/input/A01_s001_w1_z{index:03d}_t001.tif" for index in range(1, 61))
    component_metadata = tuple(
        {"well": "A01", "site": "1", "channel": "1", "z_index": str(index)}
        for index in range(1, 61)
    )
    image = RuntimeImagePayloadContext(
        np.zeros((60, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=paths,
                component_metadata=component_metadata,
            ),
        ),
    ).payload()
    labels = ObjectLabelSet(
        name="downsizedNuclei",
        labels=np.ones((60, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    contextualized = labels.with_source_image_context(image)

    assert contextualized.source_image_provenance_planes.paths == paths
    assert (
        tuple(
            dict(item)
            for item in contextualized.source_image_provenance_planes.component_metadata
        )
        == component_metadata
    )


def test_object_label_source_context_keeps_source_aligned_stack_planes_with_scalar_label_source() -> (
    None
):
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    paths = (source_path, source_path, source_path)
    component_metadata = (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "3",
            SOURCE_PLANE_INDEX_FIELD: "2",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )
    image = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=paths,
                component_metadata=component_metadata,
            ),
        ),
    ).payload()
    labels = ObjectLabelPayload(
        labels=np.ones((3, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )

    contextualized = labels.with_source_image_context(image)

    assert contextualized.source_path == source_path
    assert dict(contextualized.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }
    assert contextualized.source_image_provenance_planes.paths == paths
    assert (
        tuple(
            dict(item)
            for item in contextualized.source_image_provenance_planes.component_metadata
        )
        == component_metadata
    )


def test_image_mask_for_data_domain_broadcasts_spatial_mask_to_leading_axes() -> None:
    image = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False

    projected = image_mask_for_data_domain(
        source_payload=image,
        data=image,
        explicit_mask=mask,
    )

    assert projected.shape == image.shape
    np.testing.assert_array_equal(projected[0], mask)
    np.testing.assert_array_equal(projected[1], mask)


def test_masked_image_payload_accepts_singleton_stack_spatial_mask() -> None:
    data = np.zeros((1, 4, 5), dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False

    payload = MaskedImagePayload(data=data, mask=mask)

    np.testing.assert_array_equal(payload.mask, mask)


def test_image_payload_channel_projection_preserves_channel_mask_and_metadata() -> None:
    data = np.zeros((2, 3, 4), dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)
    mask[1, 0, 0] = False
    payload = MaskedImagePayload(
        data=data,
        mask=mask,
        metadata=ImagePayloadMetadata(
            source_plane_intensity_scales=(255.0, 65535.0),
            source_plane_dtypes=("uint8", "uint16"),
        ),
    )

    result = ImagePayloadChannelProjection.from_channel(payload, data, 1).payload()

    assert isinstance(result, MaskedImagePayload)
    assert result.data.shape == (1, 3, 4)
    np.testing.assert_array_equal(result.mask, mask[1:2])
    assert result.metadata.intensity_scale == 65535.0
    assert result.metadata.source_dtype == "uint16"


def test_selected_channel_singleton_stack_keeps_scalar_source_identity() -> None:
    data = np.stack(
        (
            np.full((3, 4), 1.0, dtype=np.float32),
            np.full((3, 4), 2.0, dtype=np.float32),
        )
    )
    source = RuntimeImagePayloadContext(
        data,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1.tif", "/input/A01_s001_w2.tif"),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            )
        ),
    ).payload()
    selected = ImagePayloadChannelProjection.from_channel(source, data, 0).payload()

    contextualized = DerivedImagePayloadContext(source, selected).payload()
    metadata = image_payload_metadata(contextualized)

    assert RuntimeImageSourceIdentityCompleteness(selected).complete()
    assert RuntimeImageSourceIdentityCompleteness(contextualized).complete()
    assert metadata.source_path == "/input/A01_s001_w1.tif"
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    assert metadata.source_image_provenance_planes.count == 0


def test_masked_image_payload_accepts_grayscale_volume_stack_mask_domains() -> None:
    data = np.zeros((1, 3, 4, 5), dtype=np.float32)

    for mask_shape in ((3, 4, 5), (1, 4, 5), (4, 5)):
        payload = MaskedImagePayload(
            data=data,
            mask=np.ones(mask_shape, dtype=bool),
        )

        assert payload.mask.shape == mask_shape


def test_masked_image_payload_accepts_color_stack_mask_domains() -> None:
    data = np.zeros((2, 4, 5, 3), dtype=np.float32)

    for mask_shape in ((2, 4, 5), (4, 5)):
        payload = MaskedImagePayload(
            data=data,
            mask=np.ones(mask_shape, dtype=bool),
        )

        assert payload.mask.shape == mask_shape


def test_image_metadata_payload_carries_source_intensity_scale() -> None:
    image = np.zeros((2, 3), dtype=np.uint16)
    metadata = ImagePayloadMetadata.for_array(
        image,
        source_path="/plate/A01_s001_w1.png",
    )

    payload = ImageMetadataPayload(data=image, metadata=metadata)

    assert payload.shape == image.shape
    assert payload.metadata.intensity_scale == 65535.0
    assert payload.metadata.source_dtype == "uint16"
    assert payload.metadata.source_path == "/plate/A01_s001_w1.png"
    np.testing.assert_array_equal(np.asarray(payload), image)


def test_source_image_metadata_context_stamps_source_spatial_domain() -> None:
    image = np.zeros((520, 696), dtype=np.uint16)

    metadata = (
        ImagePayloadSourceMetadataContext(SourceImageIdentity("/plate/A01_s001_w1.png"))
        .metadata_request(image)
        .metadata()
    )

    assert metadata.spatial_origin_yx == (0, 0)
    assert metadata.source_spatial_shape_yx == (520, 696)


def test_image_metadata_payload_exposes_array_methods() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    copied = payload.copy()

    np.testing.assert_array_equal(copied, image)
    assert copied is not image
    np.testing.assert_array_equal(payload.astype(np.float64), image.astype(np.float64))
    np.testing.assert_array_equal(payload.reshape(3, 2), image.reshape(3, 2))


def test_image_metadata_payload_supports_nominal_array_comparison() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    np.testing.assert_array_equal(payload > 2, image > 2)


def test_image_metadata_payload_ufunc_preserves_context_for_numeric_results() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    metadata = ImagePayloadMetadata(source_dtype="float32")
    payload = ImageMetadataPayload(data=image, metadata=metadata)

    result = np.add(payload, 1.0)

    assert isinstance(result, ImageMetadataPayload)
    assert result.metadata == metadata
    np.testing.assert_array_equal(result.data, image + 1.0)


def test_image_payload_metadata_composition_tracks_per_source_planes() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
        mask=None,
    ).payload()

    metadata = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert metadata.source_plane_intensity_scales == (65535.0, 255.0)
    assert metadata.for_source_plane(0).intensity_scale == 65535.0
    assert metadata.for_source_plane(1).source_dtype == "uint8"


def test_image_payload_metadata_composition_tracks_unit_interval_proof() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            intensity_scale=65535.0,
            source_dtype="uint16",
            unit_interval_intensity_scale=65535,
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            intensity_scale=255.0,
            source_dtype="uint8",
            unit_interval_intensity_scale=255,
        ),
        mask=None,
    ).payload()

    metadata = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert metadata.source_plane_unit_interval_intensity_scales == (65535, 255)
    assert metadata.for_source_plane(0).unit_interval_intensity_scale == 65535
    assert metadata.for_source_plane(1).unit_interval_intensity_scale == 255


def test_image_payload_metadata_common_unit_interval_uses_scalar_fallback() -> None:
    metadata = ImagePayloadMetadata(
        unit_interval_intensity_scale=65535,
        source_plane_unit_interval_intensity_scales=(None, None, None),
    )

    assert metadata.common_unit_interval_intensity_scale() == 65535


def test_image_payload_metadata_tracks_spatial_crop_edges() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 12),
        output_shape_yx=(4, 5),
        offset_yx=(3, 2),
        physical_border_edges_yx=(False, False, False, False),
    )

    assert metadata.spatial_origin_yx == (3, 2)
    assert metadata.source_spatial_shape_yx == (10, 12)
    assert metadata.physical_border_edges_for_shape((4, 5)) == (
        False,
        False,
        False,
        False,
    )
    assert metadata.for_source_plane(0).spatial_origin_yx == (3, 2)


def test_image_payload_metadata_reads_object_label_spatial_context() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 5), dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(3, 2),
            source_shape_yx=(10, 12),
        ),
        source_path="/input/A01_s001_w1_z001_t001.TIF",
    )

    metadata = image_payload_metadata(payload)

    assert metadata.spatial_origin_yx == (3, 2)
    assert metadata.source_spatial_shape_yx == (10, 12)
    assert metadata.source_path == "/input/A01_s001_w1_z001_t001.TIF"


def test_object_label_runtime_slice_projection_projects_source_path() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((3, 4, 5), dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(10, 12)),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s002_w1_z001_t001.TIF",
                "/input/A01_s003_w1_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
                {"well": "A01", "site": "3", "channel": "1"},
            ),
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=3),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.labels.shape == (4, 5)
    assert projected.source_path == "/input/A01_s002_w1_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert image_payload_metadata(projected).source_path == (
        "/input/A01_s002_w1_z001_t001.TIF"
    )


def test_object_label_set_runtime_slice_projection_projects_source_path() -> None:
    label_set = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 4, 5), dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(10, 12)),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s002_w1_z001_t001.TIF",
                "/input/A01_s003_w1_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
                {"well": "A01", "site": "3", "channel": "1"},
            ),
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        label_set,
        RuntimeProjectionAxis(slice_index=2, extent=3),
    )

    assert isinstance(projected, ObjectLabelSet)
    assert projected.labels.shape == (4, 5)
    assert projected.source_path == "/input/A01_s003_w1_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "3",
        "channel": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert image_payload_metadata(projected.runtime_payload()).source_path == (
        "/input/A01_s003_w1_z001_t001.TIF"
    )


def test_measurement_table_runtime_slice_projection_projects_source_identity() -> None:
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=(
            {"slice_index": 0, "ObjectNumber": 1, "AreaShape_Area": 10.0},
            {"slice_index": 1, "ObjectNumber": 2, "AreaShape_Area": 20.0},
        ),
        fields=(
            FieldSpec("slice_index", dtype="int"),
            FieldSpec("ObjectNumber", dtype="int"),
            FieldSpec("AreaShape_Area", dtype="float"),
        ),
        object_name="Nuclei",
        object_id_field="ObjectNumber",
        source_image_name="OrigDNA",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s002_w1_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        table,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(projected, MeasurementTable)
    assert tuple(projected.rows) == (
        {"slice_index": 1, "ObjectNumber": 2, "AreaShape_Area": 20.0},
    )
    assert projected.source_path == "/input/A01_s002_w1_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.runtime_schema(
        MeasurementsArtifactType
    ).source_provenance.source_path == ("/input/A01_s002_w1_z001_t001.TIF")


def test_relationship_runtime_slice_projection_projects_source_identity() -> None:
    relationship = ObjectRelationship(
        name="ParentChild",
        source=RelationshipEndpoint(
            "Cells",
            role="parent",
            id_field="parent_id",
        ),
        target=RelationshipEndpoint(
            "Nuclei",
            role="child",
            id_field="child_id",
        ),
        source_ids=(10, 11),
        target_ids=(1, 2),
        relationship_type="parent_child",
        slice_indices=(0, 1),
        slice_count=2,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w4_z001_t001.TIF",
                "/input/A01_s002_w4_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "4"},
                {"well": "A01", "site": "2", "channel": "4"},
            ),
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        relationship,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(projected, ObjectRelationship)
    assert projected.source_ids == (11,)
    assert projected.target_ids == (2,)
    assert projected.source_path == "/input/A01_s002_w4_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "4",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.runtime_schema(
        RelationshipsArtifactType
    ).source_provenance.source_path == ("/input/A01_s002_w4_z001_t001.TIF")


def test_image_payload_metadata_composition_preserves_shared_spatial_context() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 12),
        output_shape_yx=(4, 5),
        offset_yx=(0, 2),
        physical_border_edges_yx=(True, False, False, False),
    )
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32), metadata=metadata, mask=None
    ).payload()
    second = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32), metadata=metadata, mask=None
    ).payload()

    composed = ImagePayloadMetadataCompositionRequest((first, second)).metadata()

    assert composed.spatial_origin_yx == (0, 2)
    assert composed.source_spatial_shape_yx == (10, 12)
    assert composed.physical_border_edges_for_shape((4, 5)) == (
        True,
        False,
        False,
        False,
    )


def test_pure_2d_auxiliary_aggregator_preserves_image_payload_metadata() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
        mask=None,
    ).payload()

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 2, 3)
    assert (
        image_payload_metadata(stacked).for_source_plane(0).intensity_scale == 65535.0
    )
    assert image_payload_metadata(stacked).for_source_plane(1).source_dtype == "uint8"


def test_pure_2d_auxiliary_aggregator_preserves_stacked_object_labels() -> None:
    first = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32),
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 2,
    )
    second = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32) * 3,
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 4,
    )

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(stacked, ObjectLabelPayload)
    assert stacked.labels.shape == (2, 2, 3, 4)
    assert stacked.unedited_labels is not None
    assert stacked.unedited_labels.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(stacked.labels[0], first.labels)
    np.testing.assert_array_equal(stacked.labels[1], second.labels)


def test_pure_2d_auxiliary_aggregator_preserves_columnar_rows() -> None:
    first = _RuntimeValueTestColumnarRows(
        {"object_label": (1, 2), "feature_name": ("a", "b")}
    )
    second = _RuntimeValueTestColumnarRows(
        {"object_label": (3,), "feature_name": ("c",)}
    )

    single = Pure2DAuxiliaryOutputAggregator.aggregate([first], "numpy")
    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(single, ColumnarRows)
    assert single.row_count() == 2
    assert tuple(single.column_values("slice_index")) == (0, 0)
    assert isinstance(stacked, ColumnarRows)
    assert stacked.row_count() == 3
    assert tuple(stacked.column_values("object_label")) == (1, 2, 3)
    assert tuple(stacked.column_values("slice_index")) == (0, 0, 1)


def test_pure_2d_auxiliary_aggregator_preserves_columnar_row_axis() -> None:
    rows = _RuntimeValueTestColumnarRows(
        {
            "object_label": (1, 2),
            "feature_name": ("a", "b"),
            "slice_index": (3, 4),
        }
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate([rows], "numpy")

    assert isinstance(aggregated, ColumnarRows)
    assert tuple(aggregated.column_values("slice_index")) == (3, 4)


def test_pure_2d_auxiliary_aggregator_preserves_sequence_row_axis() -> None:
    @dataclass(frozen=True)
    class SliceMeasurementRow:
        slice_index: int
        object_label: int
        value: float

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [
            (SliceMeasurementRow(slice_index=0, object_label=1, value=10.0),),
            (SliceMeasurementRow(slice_index=1, object_label=1, value=20.0),),
        ],
        "numpy",
    )

    assert [(row.slice_index, row.object_label, row.value) for row in aggregated] == [
        (0, 1, 10.0),
        (1, 1, 20.0),
    ]


def test_pure_2d_auxiliary_aggregator_uses_runtime_object_label_domains() -> None:
    first = ObjectLabelPayload(
        labels=np.asarray([[0, 4], [0, 0]], dtype=np.int32),
        source_image_names=("rawDNA",),
        domain=ObjectLabelDomain(
            declared_object_ids=(4,),
        ),
    )
    second = ObjectLabelPayload(
        labels=np.asarray([[0, 7], [0, 0]], dtype=np.int32),
        source_image_names=("rawActin",),
        domain=ObjectLabelDomain(
            declared_object_ids=(7,),
        ),
    )

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(stacked, ObjectLabelPayload)
    assert stacked.domain.scope is ObjectLabelDomainScope.PLANE
    assert stacked.domain.declared_object_id_domains == ((4,), (7,))
    assert stacked.source_image_names == ("rawDNA", "rawActin")


def test_dense_object_label_plane_domain_stack_uses_square_diagonal_domains() -> None:
    labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
    labels[0, 0] = np.asarray([[1, 0], [0, 2]], dtype=np.int32)
    labels[1, 1] = np.asarray([[3, 0], [0, 4]], dtype=np.int32)
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_names=("A", "B"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (10,), (20,), (3, 4, 5)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    stack = DenseObjectLabelPlaneDomainStackRequest(payload).stack()

    assert stack is not None
    assert stack.object_id_domains == ((1, 2), (3, 4, 5))
    assert stack.measurement_row_identity is MeasurementObjectRowIdentity.ROW_SEQUENCE


def test_dense_object_label_plane_domain_stack_projects_diagonal_plane_domains() -> (
    None
):
    labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
    labels[0, 0] = np.asarray([[1, 0], [0, 2]], dtype=np.int32)
    labels[1, 1] = np.asarray([[3, 0], [0, 4]], dtype=np.int32)
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_names=("A", "B"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (10,), (20,), (3, 4, 5)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    stack = DenseObjectLabelPlaneDomainStackRequest(payload).stack()

    assert stack is not None
    first_plane = stack.plane(0)
    second_plane = stack.plane(1)
    assert isinstance(first_plane, ObjectLabelPayload)
    assert isinstance(second_plane, ObjectLabelPayload)
    assert first_plane.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert second_plane.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert first_plane.domain.declared_object_ids == (1, 2)
    assert second_plane.domain.declared_object_ids == (3, 4, 5)


def test_pure_2d_slice_index_projector_projects_object_label_payload_domains() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[0, 7], [0, 0]], dtype=np.int32),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("rawDNA", "rawActin", "rawMito"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (7,), (9,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = RuntimeSliceProjectionStrategy.strategy_for_value(
        payload,
    ).identity_projected_value(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=3),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (7,)
    assert projected.source_image_names == ("rawActin",)


def test_pure_2d_slice_index_projector_preserves_explicit_payload_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[0, 7], [0, 0]], dtype=np.int32),
        source_image_names=("rawActin",),
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )

    projected = RuntimeSliceProjectionStrategy.strategy_for_value(
        payload,
    ).identity_projected_value(
        payload,
        RuntimeProjectionAxis(slice_index=0, extent=1),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_count == 9
    assert projected.domain.declared_object_ids == ()
    assert projected.source_image_names == ("rawActin",)


def test_normalize_image_payload_intensity_uses_semantic_scale() -> None:
    image = np.array([[0, 4095]], dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        image,
        metadata=ImagePayloadMetadata(intensity_scale=4095.0, source_dtype="uint16"),
        mask=None,
    ).payload()

    normalized = normalize_image_payload_intensity(payload)

    assert image_payload_metadata(normalized).intensity_scale == 4095.0
    assert image_payload_metadata(normalized).unit_interval_intensity_scale == 4095
    assert image_payload_data(normalized).dtype == np.float32
    np.testing.assert_allclose(image_payload_data(normalized), [[0.0, 1.0]])


def test_normalize_image_payload_intensity_falls_back_to_dtype_scale() -> None:
    image = np.array([[0, 255]], dtype=np.uint8)

    normalized = normalize_image_payload_intensity(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[0.0, 1.0]])


def test_masked_image_payload_rejects_unaligned_mask_shape() -> None:
    image = np.zeros((2, 3), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="mask shape"):
        MaskedImagePayload(data=image, mask=mask)


def test_normalize_object_label_set_adds_object_schema():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = ArrayLike()

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            source_image_name="DNA",
            dimensions=("y", "x"),
        ),
        axis_id="A01",
    )

    assert value.data is labels
    assert value.schema.object_name == "Nuclei"
    assert value.schema.source_image_name == "DNA"
    assert value.schema.dimensions == ("y", "x")
    assert value.schema.label_representation is ObjectLabelRepresentation.DENSE_LABELS


def test_normalize_object_label_set_preserves_dense_label_variants():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    unedited_labels = np.array([[3, 1], [2, 0]], dtype=np.int32)
    small_removed_labels = np.array([[0, 1], [2, 4]], dtype=np.int32)

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        ),
        axis_id="A01",
    )

    assert isinstance(value.data, ObjectLabelPayload)
    assert value.schema.label_variants == (
        ObjectLabelVariant.FINAL,
        ObjectLabelVariant.UNEDITED,
        ObjectLabelVariant.SMALL_REMOVED,
    )
    restored = ObjectLabelSet.from_runtime_value(value)
    np.testing.assert_array_equal(restored.labels, labels)
    np.testing.assert_array_equal(restored.unedited_labels, unedited_labels)
    np.testing.assert_array_equal(restored.small_removed_labels, small_removed_labels)


def test_normalize_object_label_set_accepts_sparse_ijv_representation():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = [{"i": 0, "j": 1, "label": 7}]

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            representation=ObjectLabelRepresentation.SPARSE_IJV,
        ),
        axis_id="A01",
    )

    assert value.data is labels
    assert value.schema.label_representation is ObjectLabelRepresentation.SPARSE_IJV


def test_normalize_measurement_table_infers_fields_and_object_schema():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = [{"object_id": 1, "area": 12.0}]

    value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="NucleiMeasurements",
            rows=rows,
            object_name="Nuclei",
            object_id_field="object_id",
        ),
        axis_id="A01",
    )

    assert value.data is rows
    assert value.schema.object_name == "Nuclei"
    assert value.schema.object_id_field == "object_id"
    assert value.schema.measurement_subject == MeasurementSubject(
        MeasurementScope.OBJECT,
        "Nuclei",
        "object_id",
    )
    assert value.schema.fields == (FieldSpec("object_id"), FieldSpec("area"))


def test_measurement_table_normalizes_mixed_long_and_wide_rows():
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=[
            {"object_label": 1, "area": 12.0},
            {
                "object_label": 1,
                "feature_name": "Perimeter",
                "result_value": 8.0,
            },
        ],
        object_name="Nuclei",
    )

    assert table.fields == ()
    assert table.rows == [
        {
            "object_label": 1,
            "feature_name": "area",
            "result_value": 12.0,
        },
        {
            "object_label": 1,
            "feature_name": "Perimeter",
            "result_value": 8.0,
        },
    ]


def test_measurement_table_treats_value_named_columns_as_wide_without_feature_axis():
    rows = [{"image_number": 1, "mean_value": 0.5, "min_value": 0.1}]

    table = MeasurementTable(name="ImageMeasurements", rows=rows)

    assert table.rows is rows


def test_normalize_measurement_table_accepts_registered_columnar_rows():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = pd.DataFrame({"object_id": [1], "area": [12.0]})

    value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="NucleiMeasurements",
            rows=rows,
            object_name="Nuclei",
            object_id_field="object_id",
        ),
        axis_id="A01",
    )

    assert value.data is rows
    assert value.schema.fields == (FieldSpec("object_id"), FieldSpec("area"))


def test_normalize_measurement_table_accepts_generic_subject():
    output_plan = ArtifactOutputPlan(
        name="ImageMeasurements",
        path="/memory/ImageMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = [{"mean_intensity": 12.0}]

    value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="ImageMeasurements",
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.IMAGE, "DNA"),
        ),
        axis_id="A01",
    )

    assert value.schema.measurement_subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        "DNA",
    )
    assert value.schema.source_image_name == "DNA"
    assert value.schema.object_name is None


def test_object_measurement_subject_allows_implicit_object_ids():
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Nuclei")

    assert subject.id_field is None


def test_normalize_object_relationship_materializes_table_columns():
    output_plan = ArtifactOutputPlan(
        name="ParentChild",
        path="/memory/ParentChild.pkl",
        artifact_type=RelationshipsArtifactType,
    )

    value = normalize_artifact_value(
        output_plan,
        ObjectRelationship(
            name="ParentChild",
            source=RelationshipEndpoint(
                "Cells",
                role="parent",
                id_field="parent_id",
            ),
            target=RelationshipEndpoint(
                "Nuclei",
                role="child",
                id_field="child_id",
            ),
            source_ids=[10, 11],
            target_ids=[1, 2],
            relationship_type="parent_child",
        ),
        axis_id="A01",
    )

    assert value.data == {
        "relationship_type": "parent_child",
        "source_role": "parent",
        "target_role": "child",
        "source_object": "Cells",
        "target_object": "Nuclei",
        "parent_id": [10, 11],
        "child_id": [1, 2],
    }
    assert value.schema.relationship is not None
    assert value.schema.relationship.source.name == "Cells"
    assert value.schema.relationship.target.name == "Nuclei"


def test_normalize_object_relationship_preserves_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="ParentChild",
        path="/memory/ParentChild.pkl",
        artifact_type=RelationshipsArtifactType,
    )

    value = normalize_artifact_value(
        output_plan,
        ObjectRelationship(
            name="ParentChild",
            source=RelationshipEndpoint(
                "Cells",
                role="parent",
                id_field="parent_id",
            ),
            target=RelationshipEndpoint(
                "Nuclei",
                role="child",
                id_field="child_id",
            ),
            source_ids=(10, 11),
            target_ids=(1, 2),
            relationship_type="parent_child",
            slice_indices=(0, 1),
            slice_count=2,
        ),
        axis_id="A01",
    )

    relationship = ObjectRelationship.from_runtime_value(value)

    assert value.data["slice_index"] == (0, 1)
    assert value.data["slice_count"] == 2
    assert relationship.slice_indices == (0, 1)
    assert relationship.slice_count == 2


def test_native_runtime_value_name_must_match_output_plan():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    with pytest.raises(ValueError, match="does not match planned artifact"):
        normalize_artifact_value(
            output_plan,
            ObjectLabelSet(name="Cells", labels=ArrayLike()),
            axis_id="A01",
        )


def test_object_relationship_rejects_mismatched_id_lengths():
    with pytest.raises(ValueError, match="equal length"):
        ObjectRelationship(
            name="ParentChild",
            source=RelationshipEndpoint(
                "Cells",
                role="parent",
                id_field="parent_id",
            ),
            target=RelationshipEndpoint(
                "Nuclei",
                role="child",
                id_field="child_id",
            ),
            source_ids=[1],
            target_ids=[1, 2],
        )
