from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactOutputPlan,
    ImageArtifactType,
    MeasurementsArtifactType,
    MetadataArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_artifact_values import (
    ArtifactKey,
    RuntimeValue,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    ImageUnitIntervalIntensityMetadata,
    MaskedImagePayload,
    image_mask_for_data_domain,
    image_payload_data,
    image_payload_metadata,
    normalize_image_payload_intensity,
)
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_array_values import RuntimeArrayPayload, runtime_array_operand
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelSetReplacementStrategy,
    ObjectLabelStorageStrategy,
    ObjectLabelVariantData,
    object_label_dense_array,
    object_label_variant_matching_labels,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_object_label_aggregation import (
    ObjectLabelPure2DSliceAggregator,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipDeclaration,
    ObjectRelationship,
)
from openhcs.core.runtime_object_label_domains import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    ObjectLabelDomainScope,
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
    ObjectFeatureArrayDomain,
    ObjectFeatureArrayDomainStrategy,
    ObjectFeatureMissingValue,
    ObjectFeatureValueTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
    ObjectLabelVariant,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGridOrdering,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectionSourceIdentityRequest,
    RuntimeProjectionSourceIdentityRequirement,
    RuntimeSliceProjection,
    RuntimeSliceProjectionStrategy,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.source_bindings import NamedSourceBinding, SourceProjectionRole
from openhcs.core.source_image_provenance import (
    SourceImageIdentity,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_image_semantics import apply_source_binding_payload
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
    SourceVoxelSpacing,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.steps.function_runtime import (
    DefaultImageOutputSourceContextStrategy,
)
from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    ObjectSizeShapeMeasurementRowsRequest,
    ShapeObjectFeatureValueTable,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    ShapeZernikeFeatureAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DAuxiliaryOutputAggregator,
)


def test_object_label_set_is_a_runtime_array_payload() -> None:
    labels = np.asarray(((0, 1), (2, 2)), dtype=np.int32)
    label_set = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=labels),
    )

    assert isinstance(label_set, RuntimeArrayPayload)
    assert runtime_array_operand(label_set) is labels
    np.testing.assert_array_equal(np.asarray(label_set), labels)


def material_object_label_domain(labels: object) -> ObjectLabelDomain:
    """Declare the object IDs produced by test label data."""
    return PresentObjectLabelIdsDomainDeclaration().declared_domain(None, labels)


@dataclass(frozen=True, slots=True)
class _RuntimeValueTestColumnarRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        self.validate_fields()


def test_runtime_slice_aligned_values_select_from_exact_outer_domain():
    values = RuntimeSliceAlignedValues(("a", "b"))

    assert [values.value_for_aligned_slice(index, 2) for index in range(2)] == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("outer_slice_count", (None, 1, 4))
def test_runtime_slice_aligned_values_reject_absent_or_inexact_outer_domain(
    outer_slice_count: int | None,
):
    values = RuntimeSliceAlignedValues(("a", "b"))

    with pytest.raises(ValueError):
        values.value_for_aligned_slice(0, outer_slice_count)


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


def test_object_label_storage_uses_nominal_storage_authority() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))
    label_set = ObjectLabelSet(
        name="Cells", variant_data=ObjectLabelVariantData(labels=labels)
    )

    assert type(ObjectLabelStorageStrategy.for_value(payload)) is type(
        ObjectLabelStorageStrategy.for_value(label_set)
    )
    assert object_label_dense_array(payload) is labels
    assert object_label_dense_array(label_set) is labels
    assert object_label_dense_array(labels) is labels
    assert object_label_dense_array(payload, dtype=np.int32).dtype == np.int32


def test_object_label_set_rejects_nested_payload_and_implicit_authority_merge() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    unedited = np.array([[0, 1], [1, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels, unedited_labels=unedited),
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
        ),
        source_path="/payload/image.tif",
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(5, 6),
            source_shape_yx=(20, 30),
        ),
    )

    with pytest.raises(TypeError, match="requires label data"):
        ObjectLabelSet(
            name="Cells",
            variant_data=ObjectLabelVariantData(labels=payload),
            domain=ObjectLabelDomain(declared_object_count=9),
            source_path="/explicit/image.tif",
            source_spatial_domain=SourceSpatialDomain(source_shape_yx=(40, 50)),
        )


def test_object_label_payload_rejects_arbitrary_columnar_rows() -> None:
    rows = _RuntimeValueTestColumnarRows(
        columns={"label": (1,)},
        fields=(FieldSpec("label", int),),
    )

    with pytest.raises(TypeError, match="no registered nominal strategy"):
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=rows))


def test_object_label_metadata_preserves_nominal_provenance_and_parent_spacing() -> (
    None
):
    parent_spacing = SourceVoxelSpacing((2.0, 1.0, 1.0))
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        source_path="/plate/A01_s001_w1.tif",
        source_component_metadata={"well": "A01", "site": 1},
        source_image_names=("DNA",),
        parent_image_source_voxel_spacing=parent_spacing,
    )

    metadata = payload.metadata

    assert metadata.source_provenance == payload.source_provenance
    assert metadata.source_voxel_spacing == parent_spacing


def test_sparse_ijv_object_label_dense_data_uses_source_shape() -> None:
    sparse_rows = SparseIJVLabelRows(np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32))
    label_set = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=sparse_rows),
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
        variant_data=ObjectLabelVariantData(labels=sparse_rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
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

    with pytest.raises(TypeError, match="no registered nominal strategy"):
        object_label_value_with_dense_labels(
            StructuralObjectLabelDomainLookalike(),
            labels,
        )


def test_object_label_pure_2d_aggregator_preserves_dense_payload_domains() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(1,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(2,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
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


def test_object_label_pure_2d_aggregator_preserves_single_payload_identity() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(1,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (payload,),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert aggregated.labels.ndim == 3
    assert aggregated.domain.declared_object_id_domains == ((1,),)


def test_object_label_pure_2d_aggregator_rejects_unprojected_plane_stack() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match=r"declares 2 plane\(s\)"):
        ObjectLabelPure2DSliceAggregator.aggregate(
            (payload,),
            "numpy",
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_object_label_pure_2d_aggregator_preserves_slice_source_paths() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/A01_s002_w1_z001_t001.TIF",
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.source_path is None
    assert aggregated.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )


def test_object_label_pure_2d_aggregator_uses_image_metadata_composition() -> None:
    spatial_domain = SourceSpatialDomain(
        origin_yx=(2, 3),
        source_shape_yx=(8, 9),
    )
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/A01_s001_w1.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
        source_image_names=("rawDNA",),
        source_spatial_domain=spatial_domain,
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/A01_s002_w1.tif",
        source_component_metadata={
            "well": "A01",
            "site": "2",
            "channel": "1",
        },
        source_image_names=("rawDNA",),
        source_spatial_domain=spatial_domain,
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )
    expected_metadata = ImagePayloadMetadata.compose(
        (first, second),
        mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert aggregated.source_provenance == expected_metadata.source_provenance
    assert aggregated.source_spatial_domain == expected_metadata.source_spatial_domain
    assert (
        aggregated.parent_image_source_voxel_spacing
        == expected_metadata.source_voxel_spacing
    )
    assert aggregated.source_component_metadata is not None
    assert dict(aggregated.source_component_metadata) == {
        "well": "A01",
        "channel": "1",
        "extension": ".tif",
    }


def test_object_label_pure_2d_aggregator_declares_source_binding_plane_axis() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/rawDNA.tif",
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/rawGFP.tif",
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )
    first_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=first.labels),
        source_path=first.source_path,
        source_image_name="rawDNA",
        domain=first.domain,
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=second.labels),
        source_path=second.source_path,
        source_image_name="rawGFP",
        domain=second.domain,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert aggregated.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert aggregated.source_image_name == "rawDNA"
    assert aggregated.source_image_provenance_planes.paths == (
        "/input/rawDNA.tif",
        "/input/rawGFP.tif",
    )


def test_source_image_object_label_build_accepts_matching_runtime_slice_axis() -> None:
    image = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((2, 5, 6), dtype=np.float32), None)
    labels = np.zeros((2, 5, 6), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    ).payload()

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


def test_source_image_object_label_build_accepts_matching_source_binding_axis() -> None:
    image = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((2, 5, 6), dtype=np.float32), None)
    labels = np.zeros((2, 5, 6), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.SOURCE_BINDING,
            axis_size=2,
        ),
    ).payload()

    assert payload.domain.scope is ObjectLabelDomainScope.PLANE
    assert payload.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert payload.source_image_names == ("OrigHoechst", "OrigER")


def test_object_label_pure_2d_aggregator_prefers_explicit_source_plane_aliases() -> (
    None
):
    first_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawGFP",),
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert aggregated.source_image_names == ("rawDNA", "rawGFP")
    assert aggregated.source_image_name == "CellProfilerInternalImage"


def test_object_label_pure_2d_aggregator_respects_declared_source_binding_axis() -> (
    None
):
    first_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        source_image_name="CellProfilerInternalImage",
        source_image_names=("rawDNA",),
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first_set, second_set),
        "numpy",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert aggregated.source_image_names == ("rawDNA", "rawDNA")


def test_object_label_projected_plane_reduces_domain_to_payload_scope() -> None:
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 1], [0, 0]],
                    [[0, 2], [0, 0]],
                ],
                dtype=np.int32,
            )
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
    assert projected.plane_axis is None
    assert projected.source_image_names == ("rawGFP",)


def test_object_label_domain_payload_scope_collapses_plane_id_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (1, 3), (2, 3)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    payload_domain = domain.with_scope(ObjectLabelDomainScope.PAYLOAD)

    assert payload_domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert payload_domain.declared_object_ids == (1, 2, 3)
    assert payload_domain.declared_object_id_domains == ()


def test_object_label_projected_plane_promotes_channel_provenance_to_scalar() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 1], [0, 0]],
                    [[0, 2], [0, 0]],
                ],
                dtype=np.int32,
            )
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
    source_image = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((2, 2, 2), dtype=np.float32), None)
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


def test_source_backed_object_label_stack_does_not_infer_plane_count() -> None:
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 1], [0, 0]],
                    [[0, 2], [0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s1.tif", "/input/A01_s2.tif")
        ),
    )

    plane_count = labels.declared_plane_count()

    assert plane_count is None


def test_object_label_measurement_replacement_rejects_implicit_plane_selection() -> (
    None
):
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 1], [0, 0]],
                    [[0, 2], [0, 0]],
                    [[0, 3], [0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("rawDNA", "rawGFP", "rawFarRed"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    with pytest.raises(ValueError, match=r"declares 3 plane\(s\)"):
        labels.with_measurement_labels(labels.labels[2])


def test_object_label_source_plane_projection_preserves_projected_variants() -> None:
    labels = np.asarray(
        (
            ((0, 1), (0, 0)),
            ((0, 2), (0, 0)),
        ),
        dtype=np.int32,
    )
    small_removed = labels.copy()
    small_removed[1, 1, 1] = 3
    source = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=labels, small_removed_labels=small_removed
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("rawDNA", "rawGFP"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2, 3)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = source.with_source_plane_measurement_labels(labels[1], 1)

    np.testing.assert_array_equal(projected.labels, labels[1])
    np.testing.assert_array_equal(projected.small_removed_labels, small_removed[1])
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_ids == (2, 3)
    assert projected.source_image_names == ("rawGFP",)


def test_object_label_replacement_identity_preserves_planar_variants() -> None:
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)
    small_removed = labels.copy()
    small_removed[1, 1] = 2
    source = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=labels, small_removed_labels=small_removed
        ),
    )

    projected = source.with_measurement_labels(labels)

    np.testing.assert_array_equal(projected.labels, labels)
    np.testing.assert_array_equal(projected.small_removed_labels, small_removed)


def test_object_label_plane_projection_preserves_ordered_domains_and_variants() -> None:
    labels = np.asarray(
        (
            ((0, 1), (0, 0)),
            ((0, 2), (0, 0)),
            ((0, 3), (0, 0)),
        ),
        dtype=np.int32,
    )
    small_removed = labels.copy()
    small_removed[2, 1, 1] = 4
    source = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=labels, small_removed_labels=small_removed
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/one.tif", "/input/two.tif", "/input/three.tif"),
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = source.with_plane_projection((2, 0))

    np.testing.assert_array_equal(projected.labels, labels[(2, 0), ...])
    np.testing.assert_array_equal(
        projected.small_removed_labels,
        small_removed[(2, 0), ...],
    )
    assert projected.domain.declared_object_id_domains == ((3, 4), (1,))
    assert projected.source_image_provenance_planes.paths == (
        "/input/three.tif",
        "/input/one.tif",
    )


def test_sparse_object_label_plane_projection_preserves_representation() -> None:
    dense = np.asarray(
        (
            ((0, 1), (0, 0)),
            ((0, 2), (0, 0)),
            ((0, 3), (0, 0)),
        ),
        dtype=np.int32,
    )
    source = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_dense_stack(dense)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(2, 2)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = source.with_plane_projection((2, 0))

    assert projected.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(projected.labels, SparseIJVLabelRows)
    np.testing.assert_array_equal(
        object_label_dense_array(projected),
        dense[(2, 0), ...],
    )
    assert projected.domain.declared_object_id_domains == ((3,), (1,))


def test_object_label_pure_2d_aggregator_preserves_sparse_ijv_sets() -> None:
    first_labels = SparseIJVLabelRows.from_dense_labels(
        np.asarray([[0, 1], [0, 0]], dtype=np.int32)
    )
    second_labels = SparseIJVLabelRows.from_dense_labels(
        np.asarray([[0, 2], [0, 0]], dtype=np.int32)
    )
    first = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=first_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=material_object_label_domain(first_labels),
    )
    second = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=second_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=material_object_label_domain(second_labels),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(aggregated.labels, SparseIJVLabelRows)
    assert aggregated.labels.has_slice_index
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_object_label_pure_2d_aggregator_preserves_sparse_ijv_payloads() -> None:
    first_labels = SparseIJVLabelRows.from_dense_labels(
        np.asarray([[0, 1], [0, 0]], dtype=np.int32)
    )
    second_labels = SparseIJVLabelRows.from_dense_labels(
        np.asarray([[0, 2], [0, 0]], dtype=np.int32)
    )
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=first_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=material_object_label_domain(first_labels),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=second_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=material_object_label_domain(second_labels),
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
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
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_dense_labels(empty)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_count=0),
        source_spatial_domain=source_domain,
    )
    second = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_dense_labels(empty)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_count=0),
        source_spatial_domain=source_domain,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
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

    rows = table.rows()
    assert rows[0]["object_label"] == 1
    assert rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value] == 10.0
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS.value]
        == 2.0
    )
    assert MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z.value not in rows[0]
    assert rows[1]["object_label"] == 2
    assert np.isnan(rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value])
    assert (
        rows[1][MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS.value]
        == 0.0
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
    assert ObjectFeatureMissingValue.ZERO.scalar == 0.0

    rows = table.rows()
    assert (
        rows[0][
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value
        ]
        == 0.0
    )
    assert (
        rows[1][
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value
        ]
        == 20.0
    )
    assert (
        rows[2][
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER.value
        ]
        == 0.0
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
        object_domain=(892,),
    )

    rows = table.rows()

    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value] == 14.0
    )
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y.value] == 21.0
    )


def test_sparse_shape_measurement_preserves_high_object_id_feature_domain() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
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
    ).measurement_rows()

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
        variant_data=ObjectLabelVariantData(labels=source_labels),
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

    assert rebuilt.labels is transformed_labels
    assert rebuilt.domain.declared_object_count == 1
    assert rebuilt.domain.declared_object_ids == (2,)
    assert rebuilt.plane_axis is None
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
    first = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.TIF",),
            component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
        )
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s002_w1_z001_t001.TIF",),
            component_metadata=({"well": "A01", "site": "2", "channel": "1"},),
        )
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)

    metadata = ImagePayloadMetadata.compose((first, second))

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


def test_composed_image_metadata_preserves_declared_source_channel_axis() -> None:
    first = ImagePayloadMetadata(
        source_component_metadata={"channel": "1"},
        source_channel_axis=-1,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.JPG",),
            component_metadata=({"site": "1"},),
        ),
    ).payload_with(np.zeros((4, 5, 3), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        source_component_metadata={"channel": "1"},
        source_channel_axis=-1,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s002_w1_z001_t001.JPG",),
            component_metadata=({"site": "2"},),
        ),
    ).payload_with(np.ones((4, 5, 3), dtype=np.float32), None)

    metadata = ImagePayloadMetadata.compose((first, second))

    assert metadata.source_channel_axis == 3
    assert dict(metadata.source_component_metadata) == {"channel": "1"}
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"site": "1", "channel": "1"},
        {"site": "2", "channel": "1"},
    )


def test_collapse_leading_plane_axis_keeps_common_identity_and_contributors() -> None:
    metadata = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("DNA", "Actin"),
        source_plane_dtypes=("uint16", "uint16"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s001_w2_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "1", "channel": "2"},
            ),
        ),
    )

    collapsed = metadata.collapse_leading_plane_axis()

    assert collapsed.plane_axis is None
    assert dict(collapsed.source_component_metadata or {}) == {
        "well": "A01",
        "site": "1",
    }
    assert collapsed.source_path is None
    assert collapsed.source_image_names == ()
    assert collapsed.source_image_provenance_planes.contributor_count == 2
    assert collapsed.source_provenance.represented_source_image_names == (
        "DNA",
        "Actin",
    )
    assert collapsed.source_plane_dtypes == ()


def test_source_channel_axis_is_resolved_only_from_declaration() -> None:
    payload = ImagePayloadMetadata(
        source_channel_axis=-1,
    ).payload_with(np.zeros((4, 5, 3), dtype=np.float32), None)

    metadata = image_payload_metadata(payload)

    assert metadata.normalized_source_channel_axis(payload) == 2


def test_source_channel_axis_is_absent_unless_declared() -> None:
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.TIF",),
            component_metadata=(None,),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

    metadata = image_payload_metadata(payload)
    assert metadata.source_channel_axis is None
    assert metadata.normalized_source_channel_axis(payload) is None


def test_source_context_preserves_target_payload_channel_axis_domain() -> None:
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_color.tif",
        source_component_metadata={"well": "A01", "site": "1"},
        source_image_names=("OrigColor",),
        source_channel_axis=3,
    ).payload_with(np.zeros((2, 4, 5, 3), dtype=np.float32), None)
    target = ImagePayloadMetadata(
        source_channel_axis=4,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((1, 2, 4, 5, 3), dtype=np.float32), None)

    contextualized = image_payload_metadata(target).with_source_context_from(
        image_payload_metadata(source)
    ).attach_source_context_to(target)

    metadata = image_payload_metadata(contextualized)
    assert metadata.normalized_source_channel_axis(contextualized) == 4
    assert metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert metadata.source_path == "/input/A01_s001_color.tif"
    assert dict(metadata.source_component_metadata or {}) == {
        "well": "A01",
        "site": "1",
    }
    assert metadata.source_image_names == ("OrigColor",)


def test_composed_image_metadata_distinguishes_bundle_union_from_stack_consensus() -> None:
    first = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "timepoint": "7",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)

    bundle = ImagePayloadMetadata.compose(
        (first, second),
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    )
    stack = ImagePayloadMetadata.compose((first, second))

    assert bundle.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert dict(bundle.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
        "timepoint": "7",
    }
    assert stack.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert dict(stack.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    expected_plane_metadata = (
        {"well": "A01", "site": "1", "channel": "1", "timepoint": "7"},
        {"well": "A01", "site": "1", "channel": "1"},
    )
    assert (
        tuple(
            dict(item)
            for item in bundle.source_image_provenance_planes.component_metadata
        )
        == expected_plane_metadata
    )
    assert (
        tuple(
            dict(item)
            for item in stack.source_image_provenance_planes.component_metadata
        )
        == expected_plane_metadata
    )


def test_bundle_image_metadata_preserves_payload_source_provenance() -> None:
    first = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
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
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)

    bundle = ImagePayloadMetadata.compose(
        (first, second),
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    )
    stack = ImagePayloadMetadata.compose((first, second))

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


def test_bundle_image_metadata_preserves_declared_source_alias_axis() -> None:
    payloads = tuple(
        ImagePayloadMetadata(source_image_names=(name,)).payload_with(
            np.full((4, 5), index, dtype=np.float32), None
        )
        for index, name in enumerate(("OrigRed", "IllumRed", "OrigGreen", "IllumGreen"))
    )

    metadata = ImagePayloadMetadata.compose(
        payloads,
        mode=ImagePayloadMetadataCompositionMode.BUNDLE,
    )

    assert metadata.source_image_names == (
        "OrigRed",
        "IllumRed",
        "OrigGreen",
        "IllumGreen",
    )
    assert metadata.source_image_provenance_planes.count == 4


def test_removed_source_binding_planes_become_nested_stack_contributors() -> None:
    site_payloads = []
    for site_index in range(3):
        source_binding_metadata = ImagePayloadMetadata(
            source_component_metadata={"site": str(site_index + 1)},
            source_image_names=("Blue", "Green", "Red"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=tuple(
                    f"/input/site{site_index + 1}_{channel}.tif"
                    for channel in ("blue", "green", "red")
                ),
                component_metadata=tuple(
                    {
                        "site": str(site_index + 1),
                        "channel": str(channel_index + 1),
                    }
                    for channel_index in range(3)
                ),
            ),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_plane_intensity_scales=(255.0, 255.0, 255.0),
            source_plane_dtypes=("uint8", "uint8", "uint8"),
            unit_interval_intensity=ImageUnitIntervalIntensityMetadata(
                source_plane_scales=(255, 255, 255)
            ),
        )
        collapsed = source_binding_metadata.without_leading_plane_axis()
        assert collapsed.source_provenance.source_plane_count == 0
        assert collapsed.source_image_provenance_planes.contributor_count == 3
        assert collapsed.source_plane_intensity_scales == ()
        assert collapsed.source_plane_dtypes == ()
        assert collapsed.source_plane_unit_interval_intensity_scales == ()
        site_payloads.append(
            collapsed.payload_with(np.zeros((4, 5, 3), dtype=np.float32))
        )

    stacked = ImagePayloadMetadata.compose(tuple(site_payloads))

    assert stacked.source_provenance.source_plane_count == 3
    assert stacked.source_image_provenance_planes.contributor_count == 9
    assert tuple(
        len(plane.contributors)
        for plane in stacked.source_image_provenance_planes.planes
    ) == (3, 3, 3)
    round_tripped_planes = SourceImageProvenancePlanes.from_records(
        stacked.source_image_provenance_planes.records
    )
    assert round_tripped_planes.identity == (
        stacked.source_image_provenance_planes.identity
    )


def test_object_label_payload_from_composed_source_image_keeps_site_axis_metadata() -> (
    None
):
    source_slices = (
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.TIF",),
                component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
            )
        ).payload_with(np.zeros((4, 5), dtype=np.float32), None),
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s002_w1_z001_t001.TIF",),
                component_metadata=({"well": "A01", "site": "2", "channel": "1"},),
            )
        ).payload_with(np.ones((4, 5), dtype=np.float32), None),
    )
    image = ImagePayloadMetadata.compose(source_slices).payload_with(
        np.stack(tuple(image_payload_data(payload) for payload in source_slices)),
        None,
    )
    labels = np.ones((2, 4, 5), dtype=np.int32)

    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
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
    assert payload.plane_axis is None
    assert payload.domain.declared_object_count == 4
    assert payload.domain.declared_object_id_domains == ()


def test_object_label_payload_from_source_image_declares_source_binding_plane_domain() -> (
    None
):
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("rawDNA.tif", "rawGFP.tif"),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
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
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.SOURCE_BINDING,
            axis_size=2,
        ),
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
    assert payload.plane_axis is None
    assert payload.domain.declared_object_count == 4
    assert payload.domain.declared_object_id_domains == ()


def test_source_image_metadata_does_not_guess_plane_axis_from_paths() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("rawDNA.tif", "rawGFP.tif")
            )
        ),
    )

    assert image_payload_metadata(image).plane_axis is None


def test_source_image_metadata_does_not_guess_axis_from_channel_components() -> None:
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

    assert image_payload_metadata(image).plane_axis is None


def test_source_image_metadata_preserves_explicit_runtime_slice_axis() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            ),
        ),
    )
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    assert image_payload_metadata(image).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    ).payload()

    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


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
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2, 2), dtype=np.int32)),
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
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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


def test_plane_scoped_object_labels_promote_scalar_source_identity() -> None:
    image = ImagePayloadMetadata(
        source_path="/input/A02_s001_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A02",
            "site": "1",
            "channel": "1",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((1, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    contextualized = labels.with_source_image_context(image)

    contextualized.validate_source_alignment("Nuclei")
    assert contextualized.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
    )
    assert tuple(
        dict(metadata)
        for metadata in contextualized.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A02",
            "site": "1",
            "channel": "1",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )


def test_plane_scoped_object_labels_expand_indexed_scalar_source_identity() -> None:
    source_path = "/input/A02_s001_w1_z001_t001.tif"
    image = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A02",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "2",
        },
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    contextualized = labels.with_source_image_context(image)

    contextualized.validate_source_alignment("Nuclei")
    assert contextualized.source_image_provenance_planes.paths == (
        source_path,
        source_path,
    )
    assert tuple(
        dict(metadata)["z_index"]
        for metadata in contextualized.source_image_provenance_planes.component_metadata
    ) == ("1", "2")


def test_source_aligned_object_labels_replace_stale_plane_provenance() -> None:
    source_paths = (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    source_components = (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths,
                component_metadata=source_components,
            )
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(source_paths[0], source_paths[0]),
            component_metadata=(source_components[0], source_components[0]),
        ),
    )

    contextualized = labels.with_source_image_context(image)

    assert contextualized.source_image_provenance_planes.paths == source_paths
    assert (
        tuple(
            dict(item)
            for item in contextualized.source_image_provenance_planes.component_metadata
        )
        == source_components
    )


def test_source_image_loading_semantics_attaches_component_metadata() -> None:
    image = np.zeros((4, 5), dtype=np.uint16)

    payload = apply_source_binding_payload(
        image,
        NamedSourceBinding(alias="DNA"),
        ImagePayloadSourceMetadataContext(
            SourceImageIdentity(
                "01_POS002_D.TIF",
                {"well": "01", "site": "POS002", "channel": "D"},
            )
        ),
    )

    metadata = image_payload_metadata(payload)
    assert metadata.source_path == "01_POS002_D.TIF"
    assert dict(metadata.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }
    assert metadata.source_image_names == ("DNA",)
    assert metadata.source_provenance.represented_source_image_names == ("DNA",)


def test_source_image_loading_keeps_identity_separate_from_storage_address(
    tmp_path: Path,
) -> None:
    import tifffile

    storage_path = tmp_path / "physical-stack.tif"
    tifffile.imwrite(storage_path, np.zeros((4, 5), dtype=np.uint16))
    virtual_path = "/workspace/A01_s001_w2_z017_t001.tif"

    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(
            virtual_path,
            {"well": "A01", "site": 1, "channel": 2, "z_index": 17},
        ),
        source_address=str(storage_path),
    ).metadata(tifffile.imread(storage_path))

    assert metadata.source_path == virtual_path
    assert metadata.source_dtype == "uint16"
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": 1,
        "channel": 2,
        "z_index": 17,
    }


def test_source_image_loading_preserves_existing_nominal_source_name() -> None:
    image = np.zeros((4, 5), dtype=np.uint16)
    projected = ImagePayloadMetadata(
        source_image_names=("DNA",),
    ).payload_with(image)

    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(
            "/workspace/A01_s001_w1_z001_t001.tif",
            {
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "timepoint": "1",
            },
        )
    ).metadata(projected)

    assert metadata.source_image_names == ("DNA",)
    assert metadata.source_provenance.represented_source_image_names == ("DNA",)


def test_source_image_loading_uses_array_metadata_for_non_image_container(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "illumination.mat"
    storage_path.write_bytes(b"matlab-container")
    image = np.zeros((4, 5), dtype=np.float32)

    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity("/workspace/Illumination.mat"),
        source_address=str(storage_path),
    ).metadata(image)

    assert metadata.source_path == "/workspace/Illumination.mat"
    assert metadata.source_dtype == "float32"
    assert metadata.source_spatial_shape_yx == (4, 5)


def test_declared_source_image_projection_selects_pixels_and_metadata_together() -> (
    None
):
    planes = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 29, dtype=np.float32),
        )
    )
    payload = ImagePayloadMetadata(
        source_image_names=("Blue", "Green"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("blue.tif", "green.tif"),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(planes, None)

    projected = image_payload_metadata(payload).project_declared_source_image(
        payload,
        "Green",
    )

    np.testing.assert_array_equal(image_payload_data(projected), planes[1])
    metadata = image_payload_metadata(projected)
    assert metadata.plane_axis is None
    assert metadata.source_image_names == ("Green",)
    assert metadata.source_path == "green.tif"


def test_declared_source_pixel_channel_axis_owns_source_spatial_domain() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(
            source_dtype="uint8",
            source_channel_axis=-1,
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=(5, 3),
            ),
        ),
    )

    bound = apply_source_binding_payload(
        payload,
        NamedSourceBinding(
            alias="Color",
            source_channel_axis=-1,
            source_channel_counts=frozenset((3, 4)),
        ),
        ImagePayloadSourceMetadataContext(SourceImageIdentity("color.tif")),
    )

    metadata = image_payload_metadata(bound)
    assert metadata.source_channel_axis == -1
    assert metadata.source_spatial_shape_yx == (4, 5)


def test_object_label_source_image_semantics_treats_rgb_image_as_label_plane() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0:2, 0:2] = (255, 0, 0)
    image[2:4, 3:5] = (0, 255, 0)
    source = ImageMetadataPayload(
        image,
        ImagePayloadMetadata(source_channel_axis=-1),
    )

    payload = apply_source_binding_payload(
        source,
        NamedSourceBinding(
            alias="Objects",
            artifact_kind=ObjectLabelsArtifactType,
            projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            source_channel_axis=-1,
            source_channel_counts=frozenset((3, 4)),
        ),
        ImagePayloadSourceMetadataContext(SourceImageIdentity("objects.png")),
    )

    labels = image_payload_data(payload)
    assert labels.shape == (4, 5)
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert image_payload_metadata(payload).source_path == "objects.png"


def test_source_file_pixel_semantics_declare_ingestion_channel_axis(
    tmp_path: Path,
) -> None:
    import imageio.v3 as iio

    path = tmp_path / "source.png"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    iio.imwrite(path, image)

    payload = apply_source_binding_payload(
        iio.imread(path),
        NamedSourceBinding(
            alias="Color",
            source_channel_axis=-1,
            source_channel_counts=frozenset((3, 4)),
        ),
        ImagePayloadSourceMetadataContext(SourceImageIdentity(str(path))),
    )

    assert image_payload_metadata(payload).source_channel_axis == -1


def test_source_file_pixel_semantics_fill_missing_projected_channel_axis(
    tmp_path: Path,
) -> None:
    import imageio.v3 as iio

    path = tmp_path / "source.png"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    iio.imwrite(path, image)
    projected = ImagePayloadMetadata(
        source_image_names=("Color",),
    ).payload_with(iio.imread(path))

    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity(str(path))
    ).metadata(projected)

    assert metadata.source_channel_axis == -1
    assert metadata.source_image_names == ("Color",)


def test_source_file_pixel_semantics_allow_monochrome_without_declared_channel_axis(
    tmp_path: Path,
) -> None:
    import imageio.v3 as iio

    path = tmp_path / "source.png"
    iio.imwrite(path, np.zeros((4, 5), dtype=np.uint8))

    payload = apply_source_binding_payload(
        iio.imread(path),
        NamedSourceBinding(
            alias="Color",
            source_channel_axis=-1,
            source_channel_counts=frozenset((3, 4)),
        ),
        ImagePayloadSourceMetadataContext(SourceImageIdentity(str(path))),
    )

    assert image_payload_metadata(payload).source_channel_axis is None


def test_object_label_set_replacement_preserves_sparse_ijv_representation() -> None:
    source_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))
    replacement_rows = SparseIJVLabelRows(np.array([[0, 1, 1]], dtype=np.int32))
    source = ObjectLabelSet(
        name="OverlappingWorms",
        variant_data=ObjectLabelVariantData(labels=source_rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    replacement = ObjectLabelSet(
        name="OverlappingWorms",
        variant_data=ObjectLabelVariantData(labels=replacement_rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    rebuilt = source.with_labels(
        ObjectLabelSetReplacementStrategy.for_enum_member(
            source.representation
        ).replacement_labels(replacement)
    )

    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert rebuilt.labels is replacement_rows


def test_object_label_payload_rebuild_preserves_sparse_ijv_representation() -> None:
    source_rows = SparseIJVLabelRows(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32))
    replacement_rows = SparseIJVLabelRows(np.array([[0, 1, 1]], dtype=np.int32))
    source = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=source_rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    rebuilt = source.with_labels(replacement_rows)

    assert isinstance(rebuilt, ObjectLabelPayload)
    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert rebuilt.labels is replacement_rows


def test_object_label_payload_source_context_preserves_sparse_ijv_representation() -> (
    None
):
    image = ImagePayloadMetadata().payload_with(
        np.zeros((4, 5), dtype=np.float32), None
    )
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
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows(np.array([[0, 0, 1]], dtype=np.int32))
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    dense_replacement = np.array([[0, 2], [3, 0]], dtype=np.int32)

    rebuilt = source.with_labels(
        ObjectLabelSetReplacementStrategy.for_enum_member(
            source.representation
        ).replacement_labels(dense_replacement)
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
        variant_data=ObjectLabelVariantData(
            labels=labels, unedited_labels=unedited, small_removed_labels=small_removed
        ),
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

    rebuilt = payload.with_measurement_labels(selected)

    assert isinstance(rebuilt, ObjectLabelPayload)
    assert rebuilt.labels is selected
    assert rebuilt.unedited_labels is None
    assert rebuilt.small_removed_labels is None
    assert rebuilt.domain.declared_object_count == 2
    assert rebuilt.domain.declared_object_ids == (1, 2)
    assert rebuilt.spatial_origin_yx == (4, 5)
    assert rebuilt.source_spatial_shape_yx == (10, 11)


def test_object_label_variant_compatibility_uses_storage_authority() -> None:
    variant = np.ones((1, 2, 2), dtype=np.int32)
    matching_labels = np.zeros((1, 2, 2), dtype=np.int32)
    selected_labels = matching_labels[0]

    assert (
        object_label_variant_matching_labels(
            variant,
            matching_labels,
        )
        is variant
    )
    assert (
        object_label_variant_matching_labels(
            variant,
            selected_labels,
        )
        is None
    )


def test_runtime_projection_requirement_projects_singleton_object_label_stack() -> None:
    labels = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2, 3),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
    assert item.runtime_plane_metadata is not None
    assert item.runtime_plane_metadata.plane_indices == (0,)
    assert item.runtime_plane_metadata.plane_shape == (1,)
    assert item.runtime_plane_metadata.source_plane_indices == (0,)


def test_runtime_projection_requirement_tracks_multi_plane_runtime_coordinates() -> (
    None
):
    labels = np.arange(50, dtype=np.int32).reshape(2, 5, 5)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=(tuple(range(1, 25)), tuple(range(25, 50))),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
    assert first.runtime_plane_metadata.plane_indices == (0,)
    assert first.runtime_plane_metadata.plane_shape == (2,)
    assert second.runtime_plane_metadata.plane_indices == (1,)
    assert second.runtime_plane_metadata.plane_shape == (2,)
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


def test_runtime_projection_does_not_expand_indexed_scalar_source_metadata() -> None:
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    payload = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    ).payload_with(np.arange(12, dtype=np.uint16).reshape(3, 2, 2), None)

    projected = (
        RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description=source_path,
        )
    )

    assert len(projected) == 1
    assert projected[0].runtime_plane_metadata is None
    assert projected[0].source_component_metadata["z_index"] == "1"
    assert projected[0].source_component_metadata[SOURCE_PLANE_INDEX_FIELD] == "0"


def test_runtime_projection_keeps_unindexed_scalar_volume_as_one_payload() -> None:
    source_path = "/input/A01_s001_w1_z001_t001.tif"
    payload = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
        },
    ).payload_with(np.arange(8, dtype=np.uint16).reshape(2, 2, 2), None)

    (projected,) = (
        RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
    ).project_payload_items(
        RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description=source_path,
        )
    )

    assert projected.value is payload
    assert projected.runtime_plane_metadata is None


def test_runtime_slice_axis_rejects_multi_plane_projection() -> None:
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
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,), (4,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    with pytest.raises(ValueError, match="must select exactly one plane"):
        payload.with_runtime_slice_projection(
            slice_index=0,
            slice_count=2,
            label_plane_indices=(0, 2),
            source_plane_indices=None,
        )


def test_object_label_runtime_slice_projection_keeps_volume_plane_domain() -> None:
    labels = np.zeros((2, 3, 4, 4), dtype=np.int32)
    labels[0, :, 0:2, 0:2] = 1
    labels[1, :, 1:4, 1:4] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert projected is payload


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
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
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
        label_plane_indices=(0, 1),
        source_plane_indices=(0, 1),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
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


def test_normalize_artifact_value_builds_key_for_nominal_payload():
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("DAPI",),
        group_component=AllComponents.CHANNEL,
    )
    table = MeasurementTable(
        name="measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_id": 1, "area": 12.0},),
            fields=(
                FieldSpec("object_id", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Cells",
            "object_id",
        ),
    )

    value = RuntimeValue.normalize(
        output_plan,
        table,
        axis_id="A01",
    )

    assert value.name == "measurements"
    assert value.artifact_type is MeasurementsArtifactType
    assert value.key.scope.axis_id == "A01"
    assert value.key.scope.value_text == "DAPI"
    assert value.data is table


def _declared_runtime_slice_image(slice_count: int) -> ImageMetadataPayload:
    return ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((slice_count, 4, 5), dtype=np.float32), None)


def test_group_scoped_artifact_consumes_singleton_payload_axis() -> None:
    output_plan = ArtifactOutputPlan(
        name="Corrected",
        path="/memory/Corrected.tif",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
    )

    value = RuntimeValue.normalize(
        output_plan,
        _declared_runtime_slice_image(1),
        axis_id="A01",
    )

    assert image_payload_data(value.data).shape == (4, 5)
    assert image_payload_metadata(value.data).plane_axis is None


def test_group_scoped_artifact_rejects_multi_plane_payload_axis() -> None:
    output_plan = ArtifactOutputPlan(
        name="Corrected",
        path="/memory/Corrected.tif",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
    )

    with pytest.raises(ValueError, match="cannot retain a declared runtime-slice"):
        RuntimeValue.normalize(
            output_plan,
            _declared_runtime_slice_image(2),
            axis_id="A01",
        )


def test_group_scoped_stack_artifact_preserves_declared_payload_axis() -> None:
    output_plan = ArtifactOutputPlan(
        name="Corrected",
        path="/memory/Corrected.tif",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
    )

    value = RuntimeValue.normalize(
        output_plan,
        _declared_runtime_slice_image(2),
        axis_id="A01",
    )

    assert image_payload_data(value.data).shape == (2, 4, 5)
    assert (
        image_payload_metadata(value.data).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    )


def test_group_scoped_measurements_preserve_runtime_row_axis() -> None:
    output_plan = ArtifactOutputPlan(
        name="Measurements",
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
    )
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "value": 1.0},
                {"slice_index": 1, "value": 2.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("value", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
    )

    value = RuntimeValue.normalize(output_plan, table, axis_id="A01")

    assert value.data is table
    assert value.materialization_payload() == table.rows


def test_grouped_scalar_artifact_records_compose_one_runtime_axis() -> None:
    records = tuple(
        RuntimeValue.normalize(
            ArtifactOutputPlan(
                name="Corrected",
                path=f"/memory/Corrected_{site}.tif",
                artifact_type=ImageArtifactType,
                group_keys=(site,),
                group_component=AllComponents.SITE,
            ),
            ImagePayloadMetadata().payload_with(
                np.zeros((4, 5), dtype=np.float32),
                None,
            ),
            axis_id="A01",
        )
        for site in ("1", "2")
    )

    composed = RuntimeValue.compose(records)

    assert image_payload_data(composed).shape == (2, 4, 5)
    assert image_payload_metadata(composed).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_normalize_artifact_value_accepts_dataclass_measurement_rows():
    @dataclass(frozen=True, slots=True)
    class MeasurementRow:
        feature_name: str
        value: float

    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )

    row = MeasurementRow(feature_name="Area", value=12.0)
    rows = DataclassMeasurementColumnarRows((row,), row_type=MeasurementRow)
    table = MeasurementTable(
        name="measurements",
        rows=rows,
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "measurements"),
    )
    value = RuntimeValue.normalize(
        output_plan,
        table,
        axis_id="A01",
    )

    assert value.data is table
    assert table.rows is rows
    assert table.rows.fields == (
        FieldSpec("feature_name", str),
        FieldSpec("value", float),
    )
    assert value.artifact_type is MeasurementsArtifactType


def test_normalize_artifact_value_aggregates_slice_aligned_object_label_domains():
    output_plan = ArtifactOutputPlan(
        name="GridObjects",
        path="/memory/GridObjects.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [0, 3]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 2], [4, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )

    value = RuntimeValue.normalize(
        output_plan,
        RuntimeSliceAlignedValues((first, second)),
        axis_id="A01",
    )
    payload = value.data

    assert isinstance(payload, ObjectLabelSet)
    assert payload.name == output_plan.name
    assert payload.dimensions == ()
    assert payload.source_image_name is None
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert payload.dtype == np.dtype(np.int32)
    assert payload.domain == ObjectLabelDomain(
        declared_object_id_domains=((1, 2, 3, 4), (1, 2, 3, 4)),
        scope=ObjectLabelDomainScope.PLANE,
    )
    assert payload.source_spatial_domain == SourceSpatialDomain()
    assert payload.source_path is None
    assert payload.source_component_metadata is None
    assert payload.source_image_names == ()
    assert payload.source_image_provenance_planes.count == 0
    assert value.materialization_payload() is payload
    assert output_plan.materialization_payload(value) is payload
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
    expected_labels = np.array(
        [
            [[0, 1], [0, 0]],
            [[0, 2], [0, 0]],
        ],
        dtype=np.int32,
    )
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=expected_labels[0]),
        source_path="/input/A02_s001_w1_z001_t001.tif",
        source_component_metadata={"well": "A02", "site": 1, "channel": 1},
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=expected_labels[1]),
        source_path="/input/A02_s002_w1_z001_t001.tif",
        source_component_metadata={"well": "A02", "site": 2, "channel": 1},
        domain=ObjectLabelDomain(declared_object_ids=(2,)),
    )

    value = RuntimeValue.normalize(
        output_plan,
        RuntimeSliceAlignedValues((first, second)),
        axis_id="A02",
    )
    payload = value.data

    assert isinstance(payload, ObjectLabelSet)
    assert payload.name == output_plan.name
    assert payload.dimensions == ()
    assert payload.source_image_name is None
    assert payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert payload.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert payload.dtype == np.dtype(np.int32)
    assert payload.domain == ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,)),
        scope=ObjectLabelDomainScope.PLANE,
    )
    assert payload.source_spatial_domain == SourceSpatialDomain()
    assert payload.source_path is None
    assert dict(payload.source_component_metadata or {}) == {
        "well": "A02",
        "channel": 1,
        "extension": ".tif",
    }
    assert payload.source_image_names == ()
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
    np.testing.assert_array_equal(payload.labels, expected_labels)
    assert value.materialization_payload() is payload
    assert output_plan.materialization_payload(value) is payload


def test_normalize_artifact_value_rejects_metadata_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        artifact_type=MetadataArtifactType,
    )

    with pytest.raises(TypeError, match="expected metadata mapping"):
        RuntimeValue.normalize(output_plan, ["not", "metadata"], axis_id="A01")


def test_normalize_artifact_value_rejects_object_label_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    with pytest.raises(TypeError, match="expected object_labels payload"):
        RuntimeValue.normalize(output_plan, {"not": "labels"}, axis_id="A01")


def test_object_label_payload_validator_accepts_nominal_slice_aggregate():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[0, 1], [0, 2]],
                    [[3, 0], [4, 0]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    value = RuntimeValue(
        key=ArtifactKey(
            name="nuclei",
            artifact_type=ObjectLabelsArtifactType,
            scope=RuntimeExecutionAxisScope(axis_id="A01"),
        ),
        data=payload,
    )

    normalized = RuntimeValue.normalize(output_plan, value, axis_id="A01")

    assert normalized.data is payload


def test_spatial_grid_normalizes_to_nominal_runtime_value():
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

    value = RuntimeValue.normalize(output_plan, grid, axis_id="A01")

    assert value.artifact_type is SpatialGridArtifactType
    assert value.data is grid
    assert grid.rows == 30
    assert grid.x_origin == 27.0
    assert grid.ordering is SpatialGridOrdering.BY_ROWS
    assert value.materialization_payload()["rows"] == 30


def test_slice_aligned_spatial_grid_preserves_nominal_value_sequence():
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

    value = RuntimeValue.normalize(output_plan, grids, axis_id="A01")
    runtime_grids = value.data

    assert value.artifact_type is SpatialGridArtifactType
    assert isinstance(runtime_grids, RuntimeSliceAlignedValues)
    assert runtime_grids.slice_count == 2
    assert [runtime_grids.value_for_slice(index).x_origin for index in range(2)] == [
        1.0,
        2.0,
    ]


def test_spatial_grid_normalizes_pure_2d_sequence_from_canonical_mappings():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        artifact_type=SpatialGridArtifactType,
    )
    grids = tuple(
        SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=x_origin,
            y_origin=4.0,
        ).as_mapping()
        for x_origin in (1.0, 2.0)
    )

    value = RuntimeValue.normalize(output_plan, list(grids), axis_id="A01")

    assert isinstance(value.data, RuntimeSliceAlignedValues)
    assert [
        value.data.value_for_slice(slice_index).name for slice_index in range(2)
    ] == ["Grid", "Grid"]
    assert [
        value.data.value_for_slice(slice_index).x_origin for slice_index in range(2)
    ] == [1.0, 2.0]
    assert [mapping["x_origin"] for mapping in value.materialization_payload()] == [
        1.0,
        2.0,
    ]


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

    value = RuntimeValue.normalize(output_plan, grid, axis_id="A01")

    runtime_grid = cast(SpatialGrid, value.data)
    assert runtime_grid.ordering is SpatialGridOrdering.BY_COLUMNS
    assert value.materialization_payload()["ordering"] == (
        SpatialGridOrdering.BY_COLUMNS.value
    )


def test_normalize_artifact_value_rejects_generic_array_payload_as_object_labels():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    with pytest.raises(TypeError, match="expected object_labels payload"):
        RuntimeValue.normalize(output_plan, ArrayLike(), axis_id="A01")


def test_normalize_artifact_value_rejects_ndarray_as_object_labels():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.zeros((3, 3), dtype=np.uint16)

    with pytest.raises(TypeError, match="expected object_labels payload"):
        RuntimeValue.normalize(output_plan, labels, axis_id="A01")


def test_normalize_image_metadata_payload_applies_declared_identity():
    output_plan = ArtifactOutputPlan(
        name="DNA",
        path="/memory/DNA.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.CHANNEL,),
    )
    image = np.zeros((2, 2), dtype=np.float32)
    payload = ImagePayloadMetadata(
        source_image_names=("raw_DNA",),
    ).payload_with(image, None)

    value = RuntimeValue.normalize(
        output_plan,
        payload,
        axis_id="A01",
    )

    assert value.artifact_type is ImageArtifactType
    assert image_payload_data(value.data) is image
    metadata = image_payload_metadata(value.data)
    assert metadata.source_image_names == ("DNA",)
    assert metadata.source_provenance.represented_source_image_names == (
        "DNA",
        "raw_DNA",
    )


def test_object_label_runtime_value_preserves_nominal_source_provenance():
    output_plan = ArtifactOutputPlan(
        name="Mitochondria",
        path="/memory/Mitochondria.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    value = RuntimeValue.normalize(
        output_plan,
        ObjectLabelSet(
            name="Mitochondria",
            variant_data=ObjectLabelVariantData(labels=labels),
            source_path="/input/A02_s001_w5_z001_t001.tif",
            source_component_metadata={
                "well": "A02",
                "site": "1",
                "channel": "5",
            },
        ),
        axis_id="A02",
    )
    restored = cast(ObjectLabelSet, value.data)

    assert restored.source_path == "/input/A02_s001_w5_z001_t001.tif"
    assert restored.source_component_metadata == {
        "well": "A02",
        "site": "1",
        "channel": "5",
    }


def test_object_label_set_preserves_payload_parent_image_spacing() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [0, 0]], dtype=np.int32)
        ),
        parent_image_source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    )

    labels = ObjectLabelSet.from_payload("Cells", payload)
    runtime_payload = labels

    assert labels.parent_image_source_voxel_spacing == SourceVoxelSpacing(
        (2.0, 1.0, 1.0)
    )
    assert isinstance(runtime_payload, ObjectLabelSet)
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


def test_derived_image_payload_context_rejects_undeclared_mask_axis_reduction() -> None:
    image = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, :, 0, 0] = False
    source = MaskedImagePayload(data=np.stack((image, image)), mask=mask)

    with pytest.raises(ValueError, match="Mask shape"):
        image_payload_metadata(source).derive_payload(source, image)


def test_default_image_output_context_preserves_explicit_intensity_proof_invalidation() -> (
    None
):
    source = (
        ImagePayloadMetadata(
            source_dtype="uint8",
        )
        .with_unit_interval_intensity_scale(255)
        .payload_with(np.zeros((4, 5), dtype=np.float32))
    )
    output = (
        ImagePayloadMetadata(source_dtype="float32")
        .without_unit_interval_intensity_scale()
        .payload_with(np.ones((4, 5), dtype=np.float32))
    )

    result = DefaultImageOutputSourceContextStrategy().contextualize(
        source,
        output,
        None,
    )

    result_metadata = image_payload_metadata(result)
    assert result_metadata.unit_interval_intensity == (
        ImageUnitIntervalIntensityMetadata()
    )
    assert result_metadata.unit_interval_intensity_scale is None


def test_default_image_output_context_inherits_unspecified_intensity_proof() -> None:
    source = (
        ImagePayloadMetadata(
            source_dtype="uint8",
        )
        .with_unit_interval_intensity_scale(255)
        .payload_with(np.zeros((4, 5), dtype=np.float32))
    )
    output = ImagePayloadMetadata(source_dtype="float32").payload_with(
        np.ones((4, 5), dtype=np.float32)
    )

    result = DefaultImageOutputSourceContextStrategy().contextualize(
        source,
        output,
        None,
    )

    result_metadata = image_payload_metadata(result)
    assert result_metadata.unit_interval_intensity == (
        ImageUnitIntervalIntensityMetadata(scale=255)
    )
    assert result_metadata.unit_interval_intensity_scale == 255


def test_derived_image_payload_context_preserves_declared_resized_spatial_domain() -> (
    None
):
    source = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(4, 5),
        )
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    output = image_payload_metadata(source).with_spatial_resize((4, 15)).payload_with(
        np.ones((4, 15), dtype=np.float32),
        None,
    )

    result = image_payload_metadata(source).derive_payload(source, output)

    assert image_payload_metadata(result).source_spatial_domain == SourceSpatialDomain(
        origin_yx=(0, 0),
        source_shape_yx=(4, 15),
    )


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

    result = image_payload_metadata(source).derive_payload(source, output)

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


def test_derived_image_payload_context_preserves_explicit_channel_axis_removal() -> (
    None
):
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )
    source = ImageMetadataPayload(
        data=np.zeros((1, 4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_dtype="uint8",
            source_channel_axis=3,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    output = ImageMetadataPayload(
        data=np.ones((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_dtype="float32",
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )

    result = image_payload_metadata(source).derive_payload(
        source, output, plane_projection=projection
    )

    assert np.shape(image_payload_data(result)) == (1, 4, 5)
    assert image_payload_metadata(result).source_channel_axis is None


def test_derived_image_payload_context_inherits_channel_axis_for_plain_array() -> None:
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )
    source = ImageMetadataPayload(
        data=np.zeros((1, 4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_dtype="uint8",
            source_channel_axis=3,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )

    result = image_payload_metadata(source).derive_payload(
        source,
        np.ones((1, 4, 5, 3), dtype=np.float32),
        plane_projection=projection,
    )

    assert image_payload_metadata(result).source_channel_axis == 3


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

    result = image_payload_metadata(source).derive_payload(source, output)
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


def test_derived_image_payload_context_preserves_output_name_while_replacing_stale_plane_source_identity() -> (
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

    result = image_payload_metadata(source).derive_payload(source, output)
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
    assert metadata.source_image_names == ("OrigHoechst",)


def test_derived_image_payload_context_preserves_output_name_while_replacing_stale_scalar_source_identity() -> (
    None
):
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

    result = image_payload_metadata(source).derive_payload(source, output)
    metadata = image_payload_metadata(result)

    assert metadata.source_path == "/input/A01_s001_w3_z001_t001.tif"
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "3",
    }
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w3_z001_t001.tif",
    )
    assert metadata.source_image_provenance_planes.contributor_count == 1
    assert metadata.source_provenance.represented_source_image_names == (
        "OrigHoechst",
        "OrigSyto",
    )
    assert metadata.source_image_names == ("OrigHoechst",)


def test_derived_image_payload_context_preserves_output_name_while_replacing_singleton_stack_source_identity() -> (
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

    result = image_payload_metadata(source).derive_payload(source, output)
    metadata = image_payload_metadata(result)

    assert metadata.source_path == "/input/Sequence1_s001_w1_z001_t007.tif"
    assert metadata.source_component_metadata == {
        "well": "Sequence1",
        "site": "1",
        "channel": "1",
        "timepoint": "7",
    }
    assert metadata.source_image_provenance_planes.paths == (
        "/input/Sequence1_s001_w1_z001_t007.tif",
    )
    assert metadata.source_image_provenance_planes.contributor_count == 1
    assert metadata.source_provenance.represented_source_image_names == (
        "AdjacentImage",
        "OrigColor",
    )
    assert metadata.source_image_names == ("AdjacentImage",)


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

    result = image_payload_metadata(source).derive_payload(source, output)
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
    source = np.zeros((3, 4, 5), dtype=np.float32)
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

    result = image_payload_metadata(source).derive_payload(
        source,
        output,
    )
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


def test_derived_image_payload_context_uses_source_provenance_atomically() -> None:
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

    result = image_payload_metadata(source).derive_payload(source, output)
    metadata = image_payload_metadata(result)

    assert metadata.source_path is None
    assert metadata.source_component_metadata is None
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s001_w2_z001_t001.tif",
    )
    assert metadata.source_image_names == ("Grayscale",)


def test_object_label_source_context_keeps_source_aligned_stack_planes() -> None:
    paths = tuple(f"/input/A01_s001_w1_z{index:03d}_t001.tif" for index in range(1, 61))
    component_metadata = tuple(
        {"well": "A01", "site": "1", "channel": "1", "z_index": str(index)}
        for index in range(1, 61)
    )
    image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=paths,
            component_metadata=component_metadata,
        ),
    ).payload_with(np.zeros((60, 4, 5), dtype=np.float32), None)
    labels = ObjectLabelSet(
        name="downsizedNuclei",
        variant_data=ObjectLabelVariantData(labels=np.ones((60, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=tuple((1,) for _path in paths),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
    image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=paths,
            component_metadata=component_metadata,
        ),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((3, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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


def test_image_mask_for_data_domain_broadcasts_only_declared_channel_axis() -> None:
    image = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        np.zeros((4, 5, 2), dtype=np.float32), None
    )
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False

    projected = image_mask_for_data_domain(
        source_payload=image,
        data=image,
        explicit_mask=mask,
    )

    assert projected.shape == image_payload_data(image).shape
    np.testing.assert_array_equal(projected[..., 0], mask)
    np.testing.assert_array_equal(projected[..., 1], mask)


def test_masked_image_payload_rejects_undeclared_singleton_axis_broadcast() -> None:
    data = np.zeros((1, 4, 5), dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False

    with pytest.raises(ValueError, match="mask shape"):
        MaskedImagePayload(data=data, mask=mask)


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

    result = image_payload_metadata(payload).project_channel_payload(payload, data, 1)

    assert isinstance(result, MaskedImagePayload)
    assert result.data.shape == (1, 3, 4)
    np.testing.assert_array_equal(result.mask, mask[1:2])
    assert result.metadata.intensity_scale == 65535.0
    assert result.metadata.source_dtype == "uint16"


def test_color_channel_projection_removes_channel_axis_without_selecting_runtime_plane() -> (
    None
):
    data = np.zeros((4, 5, 3), dtype=np.float32)
    source = ImageMetadataPayload(
        data,
        ImagePayloadMetadata(
            source_path="/input/color.jpg",
            source_component_metadata={"site": "1"},
            source_channel_axis=-1,
        ),
    )

    selected = image_payload_metadata(source).project_channel_payload(
        source_payload=source,
        source_data=data,
        channel_index=1,
        channel_data=data[..., 1],
        channel_axis=-1,
    )

    metadata = image_payload_metadata(selected)
    assert np.shape(image_payload_data(selected)) == (4, 5)
    assert metadata.source_channel_axis is None
    assert metadata.source_path == "/input/color.jpg"
    assert dict(metadata.source_component_metadata or {}) == {"site": "1"}


def test_selected_channel_projection_keeps_its_declared_scalar_identity() -> None:
    data = np.stack(
        (
            np.full((3, 4), 1.0, dtype=np.float32),
            np.full((3, 4), 2.0, dtype=np.float32),
        )
    )
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif", "/input/A01_s001_w2.tif"),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "1", "channel": "2"},
            ),
        )
    ).payload_with(data, None)
    selected = image_payload_metadata(source).project_channel_payload(source, data, 0)

    metadata = image_payload_metadata(selected)

    assert image_payload_metadata(selected).has_complete_source_identity(selected)
    assert metadata.source_path == "/input/A01_s001_w1.tif"
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    assert metadata.source_image_provenance_planes.count == 0


def test_masked_image_payload_requires_exact_undeclared_volume_mask_domain() -> None:
    data = np.zeros((1, 3, 4, 5), dtype=np.float32)

    payload = MaskedImagePayload(data=data, mask=np.ones(data.shape, dtype=bool))
    assert payload.mask.shape == data.shape

    for mask_shape in ((3, 4, 5), (1, 4, 5), (4, 5)):
        with pytest.raises(ValueError, match="mask shape"):
            MaskedImagePayload(data=data, mask=np.ones(mask_shape, dtype=bool))


def test_masked_image_payload_omits_only_declared_channel_axis() -> None:
    data = np.zeros((2, 4, 5, 3), dtype=np.float32)
    payload = MaskedImagePayload(
        data=data,
        mask=np.ones((2, 4, 5), dtype=bool),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_channel_axis=-1,
        ),
    )

    assert payload.mask.shape == (2, 4, 5)
    with pytest.raises(ValueError, match="mask shape"):
        MaskedImagePayload(
            data=data,
            mask=np.ones((4, 5), dtype=bool),
            metadata=payload.metadata,
        )


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

    metadata = ImagePayloadSourceMetadataContext(
        SourceImageIdentity("/plate/A01_s001_w1.png")
    ).metadata(image)

    assert metadata.spatial_origin_yx == (0, 0)
    assert metadata.source_spatial_shape_yx == (520, 696)


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
    first = ImagePayloadMetadata(
        intensity_scale=65535.0, source_dtype="uint16"
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        intensity_scale=255.0, source_dtype="uint8"
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)

    metadata = ImagePayloadMetadata.compose((first, second))

    assert metadata.source_plane_intensity_scales == (65535.0, 255.0)
    assert metadata.for_source_plane(0).intensity_scale == 65535.0
    assert metadata.for_source_plane(1).source_dtype == "uint8"


def test_image_payload_metadata_composition_omits_incomplete_source_alias_axis() -> (
    None
):
    first = ImagePayloadMetadata(source_image_names=("rawDNA",)).payload_with(
        np.zeros((2, 3), dtype=np.float32), None
    )
    second = ImagePayloadMetadata(source_dtype="float32").payload_with(
        np.ones((2, 3), dtype=np.float32), None
    )

    metadata = ImagePayloadMetadata.compose((first, second))

    assert metadata.source_image_names == ()


def test_image_payload_metadata_composition_rejects_ambiguous_plane_aliases() -> None:
    payload = ImagePayloadMetadata(
        source_image_names=("rawDNA", "rawGFP")
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)

    with pytest.raises(ValueError, match="at most one source alias"):
        ImagePayloadMetadata.compose((payload,))


def test_image_payload_metadata_composition_tracks_unit_interval_proof() -> None:
    first = (
        ImagePayloadMetadata(
            intensity_scale=65535.0,
            source_dtype="uint16",
        )
        .with_unit_interval_intensity_scale(65535)
        .payload_with(np.zeros((2, 3), dtype=np.float32), None)
    )
    second = (
        ImagePayloadMetadata(
            intensity_scale=255.0,
            source_dtype="uint8",
        )
        .with_unit_interval_intensity_scale(255)
        .payload_with(np.zeros((2, 3), dtype=np.float32), None)
    )

    metadata = ImagePayloadMetadata.compose((first, second))

    assert metadata.source_plane_unit_interval_intensity_scales == (65535, 255)
    assert metadata.for_source_plane(0).unit_interval_intensity_scale == 65535
    assert metadata.for_source_plane(1).unit_interval_intensity_scale == 255


def test_image_payload_metadata_common_unit_interval_uses_scalar_fallback() -> None:
    metadata = ImagePayloadMetadata(
        unit_interval_intensity=ImageUnitIntervalIntensityMetadata(
            scale=65535,
            source_plane_scales=(None, None, None),
        )
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
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 5), dtype=np.int32)),
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
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=3
        ),
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


def test_runtime_slice_projection_does_not_infer_raw_array_axis() -> None:
    value = np.arange(2 * 1 * 4 * 5, dtype=np.int32).reshape((2, 1, 4, 5))

    projected = RuntimeSliceProjection.value_for_slice(
        value,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert projected is value


def test_object_label_set_runtime_slice_projection_projects_source_path() -> None:
    label_set = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=2, axis_size=3
        ),
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
    assert image_payload_metadata(projected).source_path == (
        "/input/A01_s003_w1_z001_t001.TIF"
    )


def test_measurement_table_runtime_slice_projection_projects_source_identity() -> None:
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "ObjectNumber": 1, "AreaShape_Area": 10.0},
                {"slice_index": 1, "ObjectNumber": 2, "AreaShape_Area": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("ObjectNumber", int),
                FieldSpec("AreaShape_Area", float),
            ),
        ),
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
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei", "ObjectNumber"),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        table,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(projected, MeasurementTable)
    assert projected.rows.row_mappings() == (
        {"slice_index": 1, "ObjectNumber": 2, "AreaShape_Area": 20.0},
    )
    assert projected.source_path == "/input/A01_s002_w1_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "1",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.source_provenance.source_path == (
        "/input/A01_s002_w1_z001_t001.TIF"
    )


def test_relationship_runtime_slice_projection_projects_source_identity() -> None:
    relationship = ObjectRelationship(
        name="ParentChild",
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
        declaration=ObjectRelationshipDeclaration(
            source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
            target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
            relationship_type="parent_child",
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=1,
            source_runtime_slice_offset=0,
            target_runtime_slice_offset=0,
        ),
        payload=DirectedObjectRelationshipPayload(
            source_ids=(10, 11), target_ids=(1, 2), slice_indices=(0, 1), slice_count=2
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        relationship,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(projected, ObjectRelationship)
    assert projected.payload.source_ids == (11,)
    assert projected.payload.target_ids == (2,)
    assert projected.source_path == "/input/A01_s002_w4_z001_t001.TIF"
    assert dict(projected.source_component_metadata) == {
        "well": "A01",
        "site": "2",
        "channel": "4",
    }
    assert projected.source_image_provenance_planes.paths == ()
    assert projected.source_provenance.source_path == (
        "/input/A01_s002_w4_z001_t001.TIF"
    )


def test_image_payload_metadata_composition_preserves_shared_spatial_context() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 12),
        output_shape_yx=(4, 5),
        offset_yx=(0, 2),
        physical_border_edges_yx=(True, False, False, False),
    )
    first = metadata.payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = metadata.payload_with(np.zeros((4, 5), dtype=np.float32), None)

    composed = ImagePayloadMetadata.compose((first, second))

    assert composed.spatial_origin_yx == (0, 2)
    assert composed.source_spatial_shape_yx == (10, 12)
    assert composed.physical_border_edges_for_shape((4, 5)) == (
        True,
        False,
        False,
        False,
    )


def test_pure_2d_auxiliary_aggregator_preserves_image_payload_metadata() -> None:
    first = ImagePayloadMetadata(
        intensity_scale=65535.0, source_dtype="uint16"
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        intensity_scale=255.0, source_dtype="uint8"
    ).payload_with(np.ones((2, 3), dtype=np.float32), None)

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 2, 3)
    assert (
        image_payload_metadata(stacked).for_source_plane(0).intensity_scale == 65535.0
    )
    assert image_payload_metadata(stacked).for_source_plane(1).source_dtype == "uint8"


def test_pure_2d_auxiliary_aggregator_preserves_stacked_object_labels() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((2, 3, 4), dtype=np.int32),
            unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 2,
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((2, 3, 4), dtype=np.int32) * 3,
            unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 4,
        ),
        domain=ObjectLabelDomain(declared_object_ids=(3,)),
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
        {"object_label": (1, 2), "feature_name": ("a", "b")},
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("feature_name", str),
        ),
    )
    second = _RuntimeValueTestColumnarRows(
        {"object_label": (3,), "feature_name": ("c",)},
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("feature_name", str),
        ),
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


def test_pure_2d_auxiliary_aggregator_preserves_slice_aligned_spatial_grids() -> None:
    grids = [
        SpatialGrid(
            name="Grid",
            rows=1,
            columns=1,
            x_spacing=4.0,
            y_spacing=4.0,
            x_origin=x_origin,
            y_origin=2.0,
        )
        for x_origin in (1.0, 3.0)
    ]

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(grids, "numpy")

    assert isinstance(aggregated, RuntimeSliceAlignedValues)
    assert tuple(
        aggregated.value_for_slice(index).x_origin
        for index in range(aggregated.slice_count)
    ) == (1.0, 3.0)


def test_pure_2d_auxiliary_aggregator_projects_columnar_row_axis() -> None:
    rows = _RuntimeValueTestColumnarRows(
        {
            "object_label": (1, 2),
            "feature_name": ("a", "b"),
            "slice_index": (3, 4),
        },
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("feature_name", str),
            FieldSpec("slice_index", int),
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate([rows], "numpy")

    assert isinstance(aggregated, ColumnarRows)
    assert tuple(aggregated.column_values("slice_index")) == (0, 0)


def test_pure_2d_auxiliary_aggregator_does_not_infer_sequence_row_semantics() -> None:
    @dataclass(frozen=True)
    class SliceMeasurementRow:
        slice_index: int
        object_label: int
        value: float

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [
            (SliceMeasurementRow(slice_index=7, object_label=1, value=10.0),),
            (SliceMeasurementRow(slice_index=7, object_label=1, value=20.0),),
        ],
        "numpy",
    )

    assert [(row.slice_index, row.object_label, row.value) for row in aggregated] == [
        (7, 1, 10.0),
        (7, 1, 20.0),
    ]


def test_pure_2d_auxiliary_aggregator_uses_runtime_object_label_domains() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 4], [0, 0]], dtype=np.int32)
        ),
        source_image_names=("rawDNA",),
        domain=ObjectLabelDomain(
            declared_object_ids=(4,),
        ),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 7], [0, 0]], dtype=np.int32)
        ),
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


def test_object_label_domain_rejects_square_diagonal_plane_inference() -> None:
    labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
    labels[0, 0] = np.asarray([[1, 0], [0, 2]], dtype=np.int32)
    labels[1, 1] = np.asarray([[3, 0], [0, 4]], dtype=np.int32)

    with pytest.raises(ValueError, match=r"declares 4 plane\(s\)"):
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_names=("A", "B"),
            domain=ObjectLabelDomain(
                declared_object_id_domains=((1, 2), (10,), (20,), (3, 4, 5)),
                scope=ObjectLabelDomainScope.PLANE,
            ),
        )


def test_pure_2d_slice_identity_does_not_select_object_label_planes() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 7), (0, 0)),
                    ((0, 9), (0, 0)),
                ),
                dtype=np.int32,
            )
        ),
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
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=3
        ),
    )

    assert projected is payload


def test_pure_2d_slice_index_projector_preserves_explicit_payload_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 7], [0, 0]], dtype=np.int32)
        ),
        source_image_names=("rawActin",),
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )

    projected = RuntimeSliceProjectionStrategy.strategy_for_value(
        payload,
    ).identity_projected_value(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=1
        ),
    )

    assert isinstance(projected, ObjectLabelPayload)
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.domain.declared_object_count == 9
    assert projected.domain.declared_object_ids == ()
    assert projected.source_image_names == ("rawActin",)


def test_normalize_image_payload_intensity_uses_semantic_scale() -> None:
    image = np.array([[0, 4095]], dtype=np.uint16)
    payload = ImagePayloadMetadata(
        intensity_scale=4095.0, source_dtype="uint16"
    ).payload_with(image, None)

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


def test_normalize_object_label_set_preserves_nominal_semantics():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.zeros((3, 3), dtype=np.int32)

    value = RuntimeValue.normalize(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(labels=labels),
            source_image_name="DNA",
            dimensions=("y", "x"),
        ),
        axis_id="A01",
    )
    label_set = cast(ObjectLabelSet, value.data)

    assert label_set.labels is labels
    assert label_set.name == "Nuclei"
    assert label_set.source_image_name == "DNA"
    assert label_set.dimensions == ("y", "x")
    assert label_set.representation is ObjectLabelRepresentation.DENSE_LABELS


def test_normalize_object_label_set_preserves_dense_label_variants():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    unedited_labels = np.array([[3, 1], [2, 0]], dtype=np.int32)
    small_removed_labels = np.array([[0, 1], [2, 4]], dtype=np.int32)

    value = RuntimeValue.normalize(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(
                labels=labels,
                unedited_labels=unedited_labels,
                small_removed_labels=small_removed_labels,
            ),
        ),
        axis_id="A01",
    )

    restored = cast(ObjectLabelSet, value.data)
    assert restored.variant_data.present_variants == (
        ObjectLabelVariant.FINAL,
        ObjectLabelVariant.UNEDITED,
        ObjectLabelVariant.SMALL_REMOVED,
    )
    np.testing.assert_array_equal(restored.labels, labels)
    np.testing.assert_array_equal(restored.unedited_labels, unedited_labels)
    np.testing.assert_array_equal(restored.small_removed_labels, small_removed_labels)


def test_normalize_object_label_set_accepts_sparse_ijv_representation():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    labels = SparseIJVLabelRows.from_label_slice(np.array([[0, 1, 7]], dtype=np.int32))

    value = RuntimeValue.normalize(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(labels=labels),
            representation=ObjectLabelRepresentation.SPARSE_IJV,
        ),
        axis_id="A01",
    )
    label_set = cast(ObjectLabelSet, value.data)

    assert label_set.labels is labels
    assert label_set.representation is ObjectLabelRepresentation.SPARSE_IJV


def test_normalize_measurement_table_preserves_declared_fields_and_subject():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"object_id": 1, "area": 12.0},),
        fields=(FieldSpec("object_id", int), FieldSpec("area", float)),
    )
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=rows,
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Nuclei",
            "object_id",
        ),
    )

    value = RuntimeValue.normalize(
        output_plan,
        table,
        axis_id="A01",
    )

    assert value.data is table
    assert table.rows is rows
    assert table.subject.object_name == "Nuclei"
    assert table.subject.object_id_field == "object_id"
    assert table.subject == MeasurementSubject(
        MeasurementScope.OBJECT,
        "Nuclei",
        "object_id",
    )
    assert table.rows.fields == (
        FieldSpec("object_id", int),
        FieldSpec("area", float),
    )


def test_measurement_table_preserves_exact_mixed_long_and_wide_schema():
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [
                {"object_label": 1, "area": 12.0},
                {
                    "object_label": 1,
                    "feature_name": "Perimeter",
                    "result_value": 8.0,
                },
            ],
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("area", float, required=False),
                FieldSpec("feature_name", str, required=False),
                FieldSpec("result_value", float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )

    assert table.rows.fields == (
        FieldSpec("object_label", int),
        FieldSpec("area", float, required=False),
        FieldSpec("feature_name", str, required=False),
        FieldSpec("result_value", float, required=False),
    )
    assert table.rows.row_mappings() == (
        {
            "object_label": 1,
            "area": 12.0,
        },
        {
            "object_label": 1,
            "feature_name": "Perimeter",
            "result_value": 8.0,
        },
    )


def test_measurement_table_preserves_value_named_columns_in_declared_schema():
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"slice_index": 0, "mean_value": 0.5, "min_value": 0.1},),
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("mean_value", float),
            FieldSpec("min_value", float),
        ),
    )
    table = MeasurementTable(
        name="IntensityMeasurements",
        rows=rows,
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "IntensityMeasurements"),
    )

    assert table.rows is rows


def test_normalize_measurement_table_accepts_nominal_columnar_rows():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = _RuntimeValueTestColumnarRows(
        {"slice_index": (0,), "object_id": (1,), "area": (12.0,)},
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_id", int),
            FieldSpec("area", float),
        ),
    )
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=rows,
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Nuclei",
            "object_id",
        ),
    )

    value = RuntimeValue.normalize(
        output_plan,
        table,
        axis_id="A01",
    )

    assert value.data is table
    assert table.rows is rows
    assert tuple(field.name for field in table.rows.fields) == (
        "slice_index",
        "object_id",
        "area",
    )


def test_measurement_table_rejects_negative_runtime_slice_index():
    with pytest.raises(ValueError, match="negative slice_index -1"):
        MeasurementTable(
            name="InvalidMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": -1, "value": 0.5},),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("value", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.ARTIFACT, "InvalidMeasurements"
            ),
        )


def test_normalize_measurement_table_accepts_generic_subject():
    output_plan = ArtifactOutputPlan(
        name="ImageMeasurements",
        path="/memory/ImageMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"mean_intensity": 12.0},),
        fields=(FieldSpec("mean_intensity", float),),
    )
    table = MeasurementTable(
        name="ImageMeasurements",
        rows=rows,
        subject=MeasurementSubject(MeasurementScope.IMAGE, "DNA"),
    )

    value = RuntimeValue.normalize(
        output_plan,
        table,
        axis_id="A01",
    )

    assert value.data is table
    assert table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        "DNA",
    )
    assert table.subject.object_name is None


def test_object_measurement_subject_allows_implicit_object_ids():
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Nuclei")

    assert subject.id_field is None


def test_normalize_object_relationship_materializes_table_columns():
    output_plan = ArtifactOutputPlan(
        name="ParentChild",
        path="/memory/ParentChild.pkl",
        artifact_type=RelationshipsArtifactType,
    )

    relationship = ObjectRelationship(
        name="ParentChild",
        declaration=ObjectRelationshipDeclaration(
            source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
            target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
            relationship_type="parent_child",
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=1,
            source_runtime_slice_offset=0,
            target_runtime_slice_offset=0,
        ),
        payload=DirectedObjectRelationshipPayload(
            source_ids=[10, 11], target_ids=[1, 2], slice_indices=(), slice_count=None
        ),
    )
    value = RuntimeValue.normalize(
        output_plan,
        relationship,
        axis_id="A01",
    )

    assert value.data is relationship
    assert value.materialization_payload() == {
        "relationship_type": "parent_child",
        "source_role": "parent",
        "target_role": "child",
        "source_object": "Cells",
        "target_object": "Nuclei",
        "producer_module_number": 1,
        "parent_id": (10, 11),
        "child_id": (1, 2),
    }
    assert relationship.declaration.source.name == "Cells"
    assert relationship.declaration.target.name == "Nuclei"


def test_normalize_object_relationship_preserves_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="ParentChild",
        path="/memory/ParentChild.pkl",
        artifact_type=RelationshipsArtifactType,
    )

    value = RuntimeValue.normalize(
        output_plan,
        ObjectRelationship(
            name="ParentChild",
            declaration=ObjectRelationshipDeclaration(
                source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
                target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
                relationship_type="parent_child",
                source_role="parent",
                target_role="child",
                source_id_field="parent_id",
                target_id_field="child_id",
                producer_module_number=1,
                source_runtime_slice_offset=0,
                target_runtime_slice_offset=0,
            ),
            payload=DirectedObjectRelationshipPayload(
                source_ids=(10, 11),
                target_ids=(1, 2),
                slice_indices=(0, 1),
                slice_count=2,
            ),
        ),
        axis_id="A01",
    )

    relationship = cast(ObjectRelationship, value.data)
    materialized = value.materialization_payload()

    assert materialized["slice_index"] == (0, 1)
    assert materialized["slice_count"] == 2
    assert relationship.payload.slice_indices == (0, 1)
    assert relationship.payload.slice_count == 2
    assert relationship.row_mappings() == (
        {
            "relationship_type": "parent_child",
            "source_role": "parent",
            "target_role": "child",
            "source_object": "Cells",
            "target_object": "Nuclei",
            "producer_module_number": 1,
            "parent_id": 10,
            "child_id": 1,
            "slice_index": 0,
            "slice_count": 2,
        },
        {
            "relationship_type": "parent_child",
            "source_role": "parent",
            "target_role": "child",
            "source_object": "Cells",
            "target_object": "Nuclei",
            "producer_module_number": 1,
            "parent_id": 11,
            "child_id": 2,
            "slice_index": 1,
            "slice_count": 2,
        },
    )


def test_object_relationship_rows_do_not_repeat_payload_provenance():
    relationship = ObjectRelationship(
        name="ParentChild",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01", "site": "1"},)
        ),
        declaration=ObjectRelationshipDeclaration(
            source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
            target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
            relationship_type="parent_child",
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=1,
            source_runtime_slice_offset=0,
            target_runtime_slice_offset=0,
        ),
        payload=DirectedObjectRelationshipPayload(
            source_ids=(10,), target_ids=(1,), slice_indices=(), slice_count=None
        ),
    )

    assert "source_image_provenance_planes" in relationship.as_table()
    assert relationship.row_mappings() == (
        {
            "relationship_type": "parent_child",
            "source_role": "parent",
            "target_role": "child",
            "source_object": "Cells",
            "target_object": "Nuclei",
            "producer_module_number": 1,
            "parent_id": 10,
            "child_id": 1,
        },
    )


def test_native_runtime_value_name_must_match_output_plan():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )

    with pytest.raises(ValueError, match="does not match planned artifact"):
        RuntimeValue.normalize(
            output_plan,
            ObjectLabelSet(
                name="Cells",
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((3, 3), dtype=np.int32)
                ),
            ),
            axis_id="A01",
        )


def test_object_relationship_rejects_mismatched_id_lengths():
    with pytest.raises(ValueError, match="equal length"):
        ObjectRelationship(
            name="ParentChild",
            declaration=ObjectRelationshipDeclaration(
                source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
                target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
                relationship_type="related",
                source_role="parent",
                target_role="child",
                source_id_field="parent_id",
                target_id_field="child_id",
                producer_module_number=1,
                source_runtime_slice_offset=0,
                target_runtime_slice_offset=0,
            ),
            payload=DirectedObjectRelationshipPayload(
                source_ids=[1], target_ids=[1, 2], slice_indices=(), slice_count=None
            ),
        )
