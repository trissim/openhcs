"""Nominal runtime object-label values and transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, cast

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core import (
    runtime_array_values,
    runtime_image_values,
)
from openhcs.core.artifacts import (
    NamedArtifactPayload,
    ObjectLabelsArtifactType,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeStrategyFamilyMixin,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainDeclaration,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelPlaneDomainStrategy,
    PreserveSourceObjectLabelDomainDeclaration,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisStrategy,
    RuntimePlaneAxisValueProjection,
    RuntimeSliceIdentityProjectableValue,
)
from openhcs.core.runtime_tabular_values import MeasurementObjectRowIdentity
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenanceAddressRequirement,
    SourceImageProvenanceFields,
    SourceImageProvenancePlaneCountRequirement,
    SourcePlaneIndexedProvenanceExpansion,
)
from openhcs.core.source_metadata import (
    SourceVoxelSpacing,
)
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainFields,
)

from enum import Enum
from openhcs.core.alias_property import AliasProperty
from openhcs.core.artifacts import ArtifactPayloadShape
from openhcs.core.registry_strategies import str_enum_member_with_payload

_PRESERVE_PLANE_AXIS = object()


def normalize_source_label_data(data: object, channel_axis: int | None) -> object:
    """Convert a source color label plane into stable positive integer IDs."""

    if channel_axis is None:
        return data
    rgb = np.moveaxis(np.asarray(data), channel_axis, -1)
    flat = rgb[..., :3].reshape(-1, 3)
    labels = np.zeros(flat.shape[0], dtype=np.int32)
    foreground = np.any(flat != 0, axis=1)
    if np.any(foreground):
        _colors, inverse = np.unique(flat[foreground], axis=0, return_inverse=True)
        labels[foreground] = inverse.astype(np.int32, copy=False) + 1
    return labels.reshape(rgb.shape[:-1])


class ObjectLabelRepresentation(str, Enum):
    """Storage representation used by an object-label artifact payload."""

    def __new__(cls, value: str, payload_shape: ArtifactPayloadShape):
        return str_enum_member_with_payload(
            cls, value, payload_attribute="_payload_shape", payload=payload_shape
        )

    DENSE_LABELS = ("dense_labels", ArtifactPayloadShape.ARRAY)
    SPARSE_IJV = ("sparse_ijv", ArtifactPayloadShape.TABLE)
    payload_shape = AliasProperty[ArtifactPayloadShape]("_payload_shape")


class ObjectLabelVariant(str, Enum):
    """Named semantic variants carried by an object-label artifact."""

    FINAL = "final"
    UNEDITED = "unedited"
    SMALL_REMOVED = "small_removed"


@dataclass(frozen=True, slots=True)
class ObjectLabelVariantData:
    """Final and optional CellProfiler object-label variant arrays."""

    labels: ObjectLabelData
    unedited_labels: ObjectLabelData | None = None
    small_removed_labels: ObjectLabelData | None = None

    @classmethod
    def compatible_replacement(
        cls,
        source: "ObjectLabelValue",
        labels: ObjectLabelData,
    ) -> "ObjectLabelVariantData":
        """Return replacement labels with source variants that still match."""
        source_variants = source.variant_data
        storage_authority = ObjectLabelStorageStrategy.for_value(source)
        return cls(
            labels=labels,
            unedited_labels=storage_authority.matching_variant(
                source,
                source_variants.unedited_labels,
                labels,
            ),
            small_removed_labels=storage_authority.matching_variant(
                source,
                source_variants.small_removed_labels,
                labels,
            ),
        )

    @property
    def present_variants(self) -> tuple[ObjectLabelVariant, ...]:
        variants = [ObjectLabelVariant.FINAL]
        if self.unedited_labels is not None:
            variants.append(ObjectLabelVariant.UNEDITED)
        if self.small_removed_labels is not None:
            variants.append(ObjectLabelVariant.SMALL_REMOVED)
        return tuple(variants)

    def labels_for_variant(
        self,
        variant: ObjectLabelVariant | str,
    ) -> ObjectLabelData:
        normalized = ObjectLabelVariant(
            variant,
        )
        match normalized:
            case ObjectLabelVariant.FINAL:
                return self.labels
            case ObjectLabelVariant.UNEDITED:
                return (
                    self.unedited_labels
                    if self.unedited_labels is not None
                    else self.labels
                )
            case ObjectLabelVariant.SMALL_REMOVED:
                return (
                    self.small_removed_labels
                    if self.small_removed_labels is not None
                    else self.labels
                )

    @classmethod
    def variant_is_present(
        cls,
        variant: ObjectLabelVariant,
        payloads: Sequence["ObjectLabelVariantData"],
    ) -> bool:
        """Return whether a semantic variant has material data."""

        match variant:
            case ObjectLabelVariant.FINAL:
                return True
            case ObjectLabelVariant.UNEDITED:
                return any(payload.unedited_labels is not None for payload in payloads)
            case ObjectLabelVariant.SMALL_REMOVED:
                return any(
                    payload.small_removed_labels is not None for payload in payloads
                )

    def with_labels(self, labels: ObjectLabelData) -> "ObjectLabelVariantData":
        """Return these variants with replacement final labels."""
        return type(self)(
            labels=labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
        )

    def project(
        self,
        projector: Callable[[ObjectLabelData], ObjectLabelData],
    ) -> "ObjectLabelVariantData":
        """Project every present variant through the same label operation."""
        return ObjectLabelVariantData(
            labels=projector(self.labels),
            unedited_labels=(
                None
                if self.unedited_labels is None
                else projector(self.unedited_labels)
            ),
            small_removed_labels=(
                None
                if self.small_removed_labels is None
                else projector(self.small_removed_labels)
            ),
        )

    def in_representation(
        self,
        representation: ObjectLabelRepresentation,
    ) -> "ObjectLabelVariantData":
        """Normalize every present variant through one declared representation."""
        return self.project(
            ObjectLabelSetReplacementStrategy.for_enum_member(
                representation
            ).replacement_labels
        )

    def project_runtime_slice(
        self,
        *,
        slice_index: int,
        slice_count: int,
    ) -> "ObjectLabelVariantData":
        """Project every present variant onto one runtime slice."""
        return self.project_plane(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=slice_index,
            plane_count=slice_count,
        )

    def project_plane(
        self,
        *,
        plane_axis: RuntimePlaneAxis,
        plane_index: int,
        plane_count: int,
    ) -> "ObjectLabelVariantData":
        """Project every present variant onto one declared label plane."""
        return self.project(
            lambda labels: self.project_label_data_plane(
                labels,
                plane_axis=plane_axis,
                plane_index=plane_index,
                plane_count=plane_count,
            )
        )

    @staticmethod
    def project_label_data_plane(
        labels: ObjectLabelData,
        *,
        plane_axis: RuntimePlaneAxis,
        plane_index: int,
        plane_count: int,
    ) -> ObjectLabelData:
        """Project one object-label variant through nominal plane semantics."""
        return object_label_project_plane(
            labels,
            plane_index,
            plane_count=plane_count,
        )


@dataclass(kw_only=True)
class ObjectLabelValue(
    SourceImageProvenanceFields,
    SourceSpatialDomainFields,
    ObjectLabelDomainMetadata,
    runtime_image_values.ImagePayloadMetadataCarrier,
    RuntimeSliceIdentityProjectableValue,
    runtime_array_values.RuntimeArrayPayload,
    ABC,
):
    """Nominal object-label carrier with dense labels and domain metadata."""

    variant_data: ObjectLabelVariantData
    representation: ObjectLabelRepresentation
    domain: ObjectLabelDomain
    plane_axis: RuntimePlaneAxis | None
    parent_image_source_voxel_spacing: SourceVoxelSpacing

    def object_label_domain(self) -> ObjectLabelDomain:
        return self.domain

    def measurement_object_row_identity(
        self,
        declared_identity: MeasurementObjectRowIdentity,
    ) -> MeasurementObjectRowIdentity:
        """Project declared row identity through this label value's domain scope."""
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            self.domain.scope
        ).measurement_object_row_identity(declared_identity)

    def validate_object_label_variants(self) -> None:
        if not isinstance(self.variant_data, ObjectLabelVariantData):
            raise TypeError(
                f"{type(self).__name__}.variant_data requires ObjectLabelVariantData."
            )

    @property
    def labels(self) -> ObjectLabelData:
        return self.variant_data.labels

    @property
    def shape(self) -> Any:
        return self.labels.shape

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def dtype(self) -> Any:
        return self.labels.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        return np.asarray(self.labels, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.labels

    def with_data(self, data: Any) -> Self:
        return self.with_labels(data)

    def __getitem__(self, key: Any) -> Any:
        return self.labels[key]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def unedited_labels(self) -> ObjectLabelData | None:
        return self.variant_data.unedited_labels

    @property
    def small_removed_labels(self) -> ObjectLabelData | None:
        return self.variant_data.small_removed_labels

    @property
    def metadata(self) -> runtime_image_values.ImagePayloadMetadata:
        """Return image-domain metadata carried by this object-label value."""

        return runtime_image_values.ImagePayloadMetadata(
            source_provenance=self.source_provenance,
            source_spatial_domain=self.source_spatial_domain,
            source_voxel_spacing=self.parent_image_source_voxel_spacing,
            plane_axis=self.plane_axis,
        )

    def measurement_reference_image(self) -> runtime_array_values.RuntimeArrayData:
        """Return an image payload in this value's exact object-label domain."""

        labels = object_label_dense_array(self)
        domain_strategy = ObjectLabelPlaneDomainStrategy.for_enum_member(
            self.object_label_domain().scope
        )
        metadata = self.metadata.with_source_provenance(
            domain_strategy.measurement_reference_source_provenance(self)
        )
        return metadata.payload_with(
            np.zeros(labels.shape, dtype=np.float32),
        )

    def declared_plane_projection(self) -> RuntimePlaneAxisValueProjection | None:
        """Return the exact runtime plane axis declared by this label value."""
        domain = self.object_label_domain()
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            domain.scope
        ).value_projection(self)

    def measurement_planes(self) -> tuple["ObjectLabelValue", ...]:
        """Return values projected through the declared label-plane domain."""
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            self.object_label_domain().scope
        ).declared_measurement_planes(self)

    def measurement_plane_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return object-ID domains in declared measurement-plane order."""
        domain = self.object_label_domain()
        return ObjectLabelPlaneDomainStrategy.for_enum_member(domain.scope).plane_domains(
            self,
            domain=domain,
        )

    def runtime_slice_plane_count(self) -> int | None:
        """Return the declared runtime-slice plane count, when present."""
        projection = self.declared_plane_projection()
        if projection is None or projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        return projection.axis_size

    def declared_plane_count(self) -> int | None:
        """Return this value's validated plane-scoped label cardinality."""
        if self.domain.scope is not ObjectLabelDomainScope.PLANE:
            return None
        declared_domains = self.domain.declared_object_id_domains
        if not declared_domains:
            raise ValueError(
                f"{type(self).__name__} declares plane-scoped labels without one "
                "object-ID domain per plane."
            )
        plane_count = len(declared_domains)
        object_label_validate_plane_count(
            self.labels,
            plane_count=plane_count,
            context=type(self).__name__,
        )
        return plane_count

    def validate_source_alignment(self, label_name: str) -> None:
        """Validate source-addressable provenance for every declared label plane."""
        if self.domain.scope is ObjectLabelDomainScope.PAYLOAD:
            return
        if self.domain.scope is not ObjectLabelDomainScope.PLANE:
            raise TypeError(
                f"Unsupported object-label domain scope {self.domain.scope!r}."
            )
        plane_count = self.declared_plane_count()
        if plane_count is None:
            raise ValueError(
                f"Plane-scoped object labels {label_name!r} require a declared "
                "object-label plane stack."
            )
        SourceImageProvenancePlaneCountRequirement(
            provenance=self.source_provenance,
            expected_count=plane_count,
            label_name=label_name,
        ).validate()
        for plane_index in range(plane_count):
            SourceImageProvenanceAddressRequirement(
                provenance=self.source_provenance.for_source_plane(plane_index),
                label_name=label_name,
                plane_index=plane_index,
            ).validate()

    @property
    def source_image_name(self) -> str | None:
        """Return the semantic source image name when this carrier has one."""
        return None

    @property
    def source_aliases(self) -> tuple[str, ...]:
        """Return source-binding aliases carried by this object-label value."""
        aliases = tuple(
            dict.fromkeys(alias for alias in self.source_image_names if alias)
        )
        if aliases:
            return aliases
        source_image_name = self.source_image_name
        if source_image_name is not None:
            return (source_image_name,)
        return ()

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Return schema dimensions carried by native object-label values."""
        return ()

    def with_variants(
        self,
        variants: "ObjectLabelVariantData",
        *,
        representation: ObjectLabelRepresentation | None = None,
        domain: ObjectLabelDomain | None = None,
        source_provenance: SourceImageProvenance | None = None,
        source_spatial_domain: SourceSpatialDomain | None = None,
        parent_image_source_voxel_spacing: SourceVoxelSpacing | None = None,
        plane_axis: RuntimePlaneAxis | None | object = _PRESERVE_PLANE_AXIS,
    ) -> Self:
        """Return this nominal carrier with exact replacement label semantics."""
        selected_representation = (
            self.representation if representation is None else representation
        )
        selected_domain = self.domain if domain is None else domain
        selected_plane_axis = (
            ObjectLabelPlaneDomainStrategy.for_enum_member(
                selected_domain.scope
            ).value_plane_axis(self.plane_axis)
            if plane_axis is _PRESERVE_PLANE_AXIS
            else plane_axis
        )
        normalized_variants = variants.in_representation(selected_representation)
        return self.replace_fields(
            variant_data=normalized_variants,
            representation=selected_representation,
            domain=selected_domain,
            source_provenance=(
                self.source_provenance
                if source_provenance is None
                else source_provenance
            ),
            source_spatial_domain=(
                self.source_spatial_domain
                if source_spatial_domain is None
                else source_spatial_domain
            ),
            parent_image_source_voxel_spacing=(
                self.parent_image_source_voxel_spacing
                if parent_image_source_voxel_spacing is None
                else parent_image_source_voxel_spacing
            ),
            plane_axis=selected_plane_axis,
        )

    def with_replacement_labels(
        self,
        labels: ObjectLabelData,
        *,
        representation: ObjectLabelRepresentation | None = None,
        domain: ObjectLabelDomain | None = None,
        source_spatial_domain: SourceSpatialDomain | None = None,
    ) -> Self:
        """Return compatible replacement labels in this nominal carrier."""
        return self.with_variants(
            ObjectLabelVariantData.compatible_replacement(self, labels),
            representation=representation,
            domain=domain,
            source_spatial_domain=source_spatial_domain,
        )

    def with_labels(
        self,
        labels: ObjectLabelData,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelValue":
        """Return this carrier's metadata with replacement labels."""
        return self.with_variants(
            ObjectLabelVariantData(labels, unedited_labels, small_removed_labels),
        )

    def with_projected_plane(
        self,
        labels: ObjectLabelData,
        plane_index: int,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelValue":
        """Return one selected label plane with projected domain metadata."""
        return self.with_variants(
            ObjectLabelVariantData(labels, unedited_labels, small_removed_labels),
            domain=object_label_domain_for_projected_label_plane(self, plane_index),
            source_provenance=self.source_provenance.for_source_plane(plane_index),
            plane_axis=None,
        )

    def with_measurement_labels(self, labels: ObjectLabelData) -> Self:
        """Return measurement-time labels in this value's declared domain."""
        variants = ObjectLabelVariantData.compatible_replacement(self, labels)
        if self.domain.scope is ObjectLabelDomainScope.PLANE:
            plane_count = self.declared_plane_count()
            if plane_count is None:
                raise ValueError(
                    "Plane-scoped object-label replacement has no declared plane count."
                )
            object_label_validate_plane_count(
                variants.labels,
                plane_count=plane_count,
                context="Object-label measurement replacement",
            )
        if (
            variants.labels is self.variant_data.labels
            and variants.unedited_labels is self.variant_data.unedited_labels
            and variants.small_removed_labels is self.variant_data.small_removed_labels
        ):
            return self
        return self.with_variants(variants)

    def with_source_plane_measurement_labels(
        self,
        labels: ObjectLabelData,
        plane_index: int,
    ) -> Self:
        """Return measurement labels projected from one declared source plane."""
        plane_count = self.declared_plane_count()
        if plane_count is None:
            raise ValueError(
                "Source-plane measurement projection requires plane-scoped labels."
            )
        if plane_index < 0 or plane_index >= plane_count:
            raise IndexError(plane_index)
        variants = self.variant_data.project_plane(
            plane_axis=self.plane_axis,
            plane_index=plane_index,
            plane_count=plane_count,
        ).with_labels(labels)
        return self.with_variants(
            variants,
            domain=object_label_domain_for_projected_label_plane(self, plane_index),
            source_provenance=self.source_provenance.for_source_plane(plane_index),
            plane_axis=None,
        )

    def with_runtime_slice_projection(
        self,
        *,
        slice_index: int,
        slice_count: int,
        label_plane_indices: tuple[int, ...] | None,
        source_plane_indices: tuple[int, ...] | None,
    ) -> "ObjectLabelValue":
        """Return this carrier projected onto one runtime slice."""
        if self.plane_axis is None:
            return self
        source_variants = self.variant_data
        variants = (
            source_variants.project(
                lambda labels: (
                    object_label_project_plane(
                        labels,
                        label_plane_indices[0],
                        plane_count=len(self.domain.declared_object_id_domains),
                    )
                    if len(label_plane_indices) == 1
                    else object_label_project_planes(
                        labels,
                        label_plane_indices,
                        plane_count=len(self.domain.declared_object_id_domains),
                    )
                )
            )
            if label_plane_indices is not None
            else source_variants.project_runtime_slice(
                slice_index=slice_index,
                slice_count=slice_count,
            )
        )
        projected_domain = self.runtime_slice_domain(
            slice_index=slice_index,
            slice_count=slice_count,
            plane_indices=label_plane_indices,
        )
        projected_plane_count = (
            len(projected_domain.declared_object_id_domains)
            if projected_domain.scope is ObjectLabelDomainScope.PLANE
            else 1
        )
        projected_axis = RuntimePlaneAxisStrategy.for_enum_member(
            self.plane_axis
        ).projected_axis(projected_plane_count)
        slice_metadata = runtime_image_values.image_payload_metadata(
            self
        ).for_grouped_source_plane_projection(
            source_plane_indices=source_plane_indices,
            runtime_plane_index=slice_index,
            runtime_plane_count=slice_count,
        )
        return self.with_variants(
            variants,
            domain=projected_domain,
            source_provenance=slice_metadata.source_provenance,
            source_spatial_domain=slice_metadata.object_label_source_spatial_domain(),
            plane_axis=projected_axis,
        )

    def with_plane_projection(
        self,
        plane_indices: Sequence[int],
    ) -> "ObjectLabelValue":
        """Return the ordered subset of planes selected by component scope."""
        normalized_indices = tuple(int(index) for index in plane_indices)
        if not normalized_indices:
            raise ValueError("Object-label plane projection cannot be empty.")
        plane_count = self.declared_plane_count()
        if plane_count is None:
            raise ValueError("Object-label plane projection requires a plane stack.")
        invalid_indices = tuple(
            index for index in normalized_indices if index < 0 or index >= plane_count
        )
        if invalid_indices:
            raise IndexError(
                "Object-label plane projection indices must be within "
                f"0..{plane_count - 1}; got {invalid_indices!r}."
            )
        if normalized_indices == tuple(range(plane_count)):
            return self
        metadata = runtime_image_values.image_payload_metadata(self).for_source_planes(
            normalized_indices
        )
        variants = self.variant_data.project(
            lambda labels: object_label_project_planes(
                labels,
                normalized_indices,
                plane_count=plane_count,
            )
        )
        return self.with_variants(
            variants,
            domain=self.object_label_domain().project_planes(normalized_indices),
            source_provenance=metadata.source_provenance,
        )

    def with_runtime_slice_identity(
        self,
        *,
        slice_index: int,
        slice_count: int,
    ) -> Self:
        """Return this object-label carrier stamped with execution-slice identity."""
        del slice_index, slice_count
        return self

    def normalize_object_label_metadata(
        self,
        value_label: str,
    ) -> ObjectLabelRepresentation:
        """Normalize shared object-label domain and provenance fields."""
        if not isinstance(self.domain, ObjectLabelDomain):
            raise TypeError(
                f"{value_label}.domain requires ObjectLabelDomain, "
                f"got {type(self.domain).__name__}."
            )
        self.representation = ObjectLabelRepresentation(
            self.representation,
        )
        if self.plane_axis is not None:
            self.plane_axis = RuntimePlaneAxis(
                self.plane_axis,
            )
        if self.domain.scope is ObjectLabelDomainScope.PAYLOAD:
            if self.plane_axis is not None:
                raise ValueError(
                    f"{value_label} payload-scoped labels cannot declare a plane axis."
                )
        elif self.domain.scope is ObjectLabelDomainScope.PLANE:
            if self.plane_axis is None:
                raise ValueError(
                    f"{value_label} plane-scoped labels require a declared plane axis."
                )
            declared_domains = self.domain.declared_object_id_domains
            object_label_validate_plane_count(
                self.labels,
                plane_count=len(declared_domains),
                context=value_label,
            )
        else:
            raise TypeError(
                f"{value_label} has unsupported object-label domain scope "
                f"{self.domain.scope!r}."
            )
        self.normalize_source_spatial_domain_fields()
        self.normalize_source_provenance_fields()
        ObjectLabelStorageStrategy.for_value(self).validate_representation(
            self,
            representation=self.representation,
            value_label=value_label,
        )
        return self.representation

    def runtime_slice_domain(
        self,
        *,
        slice_index: int,
        slice_count: int,
        plane_indices: tuple[int, ...] | None = None,
    ) -> ObjectLabelDomain:
        """Return the object-id domain represented by one runtime slice."""
        domain = self.object_label_domain()
        if plane_indices is not None:
            return domain.project_planes(plane_indices)
        return domain.project_slice(slice_index, slice_count)

    def object_label_source_spatial_domain(self) -> SourceSpatialDomain:
        """Return this value's source-image coordinate domain."""
        return self.source_spatial_domain.with_value_name(
            runtime_image_values.OBJECT_LABEL_SOURCE_SPATIAL_VALUE_NAME,
        )

    def apply_source_spatial_coordinate_offset(
        self,
        feature_values: MutableMapping[str, np.ndarray],
        *,
        x_fields: Sequence[str],
        y_fields: Sequence[str],
        local_offset_yx: tuple[int, int] = (0, 0),
    ) -> None:
        """Project local object-coordinate feature arrays into source-image XY."""
        source_origin = self.object_label_source_spatial_domain().origin_yx
        origin_yx = source_origin if source_origin is not None else (0, 0)
        offset_y = int(origin_yx[0]) + int(local_offset_yx[0])
        offset_x = int(origin_yx[1]) + int(local_offset_yx[1])
        if offset_x:
            for field in x_fields:
                if field in feature_values:
                    feature_values[field] = (
                        np.asarray(feature_values[field], dtype=float) + offset_x
                    )
        if offset_y:
            for field in y_fields:
                if field in feature_values:
                    feature_values[field] = (
                        np.asarray(feature_values[field], dtype=float) + offset_y
                    )

    def object_label_semantic_identity(self) -> tuple[tuple[str, Hashable], ...]:
        """Return the declared semantic identity for label-domain batching."""
        source_spatial_domain = self.object_label_source_spatial_domain()
        return (
            ("carrier", (type(self).__module__, type(self).__qualname__)),
            ("representation", self.representation),
            ("domain", self.object_label_domain()),
            ("plane_axis", self.plane_axis),
            ("source_provenance", self.source_provenance.equality_identity),
            (
                "source_spatial_domain",
                (
                    source_spatial_domain.origin_yx,
                    source_spatial_domain.source_shape_yx,
                    repr(source_spatial_domain.fill_value),
                    source_spatial_domain.value_name,
                ),
            ),
        )

    def object_label_dense_projection_identity(
        self,
    ) -> tuple[tuple[str, Hashable], ...]:
        """Return label identity for caches that already hold aligned dense data."""
        return (
            ("carrier", (type(self).__module__, type(self).__qualname__)),
            ("representation", self.representation),
            ("domain", self.object_label_domain()),
            ("plane_axis", self.plane_axis),
            ("source_provenance", self.source_provenance.equality_identity),
        )

    def with_source_image_context(
        self, image: runtime_array_values.RuntimeArrayData
    ) -> Self:
        """Return this object-label value with missing provenance filled from image."""
        metadata = runtime_image_values.image_payload_metadata(image)
        return self.with_variants(
            self.variant_data,
            source_provenance=object_label_source_context_provenance(self, image),
            source_spatial_domain=(
                self.object_label_source_spatial_domain().with_missing_from(
                    metadata.object_label_source_spatial_domain()
                )
            ),
        )

    def with_parent_image_context(
        self,
        image: runtime_array_values.RuntimeArrayData,
    ) -> "ObjectLabelValue":
        """Return this value with missing CellProfiler parent-image spacing filled."""
        parent_spacing = self.parent_image_source_voxel_spacing.with_missing_from(
            runtime_image_values.image_payload_metadata(image).source_voxel_spacing
        )
        return self.with_variants(
            self.variant_data,
            parent_image_source_voxel_spacing=parent_spacing,
        )


def object_label_source_context_provenance(
    label: ObjectLabelValue,
    image: runtime_array_values.RuntimeArrayData,
) -> SourceImageProvenance:
    """Merge image provenance into labels without reviving stale stack axes."""
    label_provenance = label.source_provenance
    image_provenance = runtime_image_values.image_payload_metadata(
        image
    ).source_provenance
    if label.domain.scope is ObjectLabelDomainScope.PLANE:
        plane_count = label.declared_plane_count()
        if plane_count is None:
            raise ValueError(
                "Plane-scoped object-label source context requires a declared "
                "label-plane count."
            )
        if image_provenance.source_plane_count == 0:
            if plane_count == 1:
                image_provenance = runtime_image_values.ImagePayloadMetadata.compose(
                    (image,),
                    mode=(
                        runtime_image_values.ImagePayloadMetadataCompositionMode.STACK
                    ),
                ).source_provenance
            else:
                image_provenance = SourcePlaneIndexedProvenanceExpansion(
                    image_provenance,
                    expected_plane_count=plane_count,
                ).expanded()
        return image_provenance.with_missing_from(
            label_provenance
        ).with_common_scalar_identity_from_planes()
    merged = label_provenance.with_missing_from(image_provenance)
    if not (label_provenance.addressable and label_provenance.source_plane_count == 0):
        return merged
    names = merged.source_image_names
    if len(names) > 1:
        unique_names = tuple(dict.fromkeys(names))
        names = unique_names if len(unique_names) == 1 else ()
    return SourceImageProvenance(
        source_path=merged.source_path,
        source_component_metadata=merged.source_component_metadata,
        source_image_names=names,
    )


@dataclass(slots=True)
class ObjectLabelPayload(ObjectLabelValue):
    """Dense object labels plus optional semantic label variants."""

    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS
    domain: ObjectLabelDomain = field(default_factory=ObjectLabelDomain)
    plane_axis: RuntimePlaneAxis | None = None
    parent_image_source_voxel_spacing: SourceVoxelSpacing = field(
        default_factory=SourceVoxelSpacing
    )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.validate_object_label_variants()
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.normalize_object_label_metadata("ObjectLabelPayload")

ObjectLabelData = np.ndarray | SparseIJVLabelRows

ObjectLabelValueBuildSource = (
    ObjectLabelValue | ObjectLabelDomainMetadata | ObjectLabelData
)

ObjectLabelMeasurementSource = ObjectLabelValue | ObjectLabelData


class ObjectLabelDataDomainMetadataStrategy(ObjectLabelDomainMetadataStrategy):
    """Dense and sparse label data declares no independent identity domain."""

    value_type = (np.ndarray, SparseIJVLabelRows)

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        del value
        return ObjectLabelDomain()


@dataclass(slots=True, kw_only=True)
class ObjectLabelSet(ObjectLabelValue, NamedArtifactPayload):
    """Native OpenHCS object-label value."""

    name: str
    dimensions: tuple[str, ...] = ()
    source_image_name: str | None = None
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS
    domain: ObjectLabelDomain = field(default_factory=ObjectLabelDomain)
    plane_axis: RuntimePlaneAxis | None = None
    parent_image_source_voxel_spacing: SourceVoxelSpacing = field(
        default_factory=SourceVoxelSpacing
    )

    @classmethod
    def from_payload(
        cls,
        name: str,
        payload: ObjectLabelValue,
        *,
        dimensions: tuple[str, ...] = (),
        source_image_name: str | None = None,
    ) -> Self:
        """Bind artifact identity to an already nominal object-label payload."""
        return cls(
            name=name,
            dimensions=dimensions,
            source_image_name=source_image_name,
            variant_data=payload.variant_data,
            representation=payload.representation,
            domain=payload.domain,
            plane_axis=payload.plane_axis,
            source_spatial_domain=payload.source_spatial_domain,
            parent_image_source_voxel_spacing=(
                payload.parent_image_source_voxel_spacing
            ),
            source_provenance=payload.source_provenance,
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.validate_object_label_variants()
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.validate_artifact_name()
        if self.source_image_name == "":
            raise ValueError("ObjectLabelSet.source_image_name cannot be empty.")
        self.normalize_object_label_metadata(f"ObjectLabelSet '{self.name}'")

    def source_alias_plane_index(
        self,
        source_aliases: tuple[str, ...],
        axis_size: int,
    ) -> int | None:
        """Resolve source aliases only when this label set declares that axis."""
        if self.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
            return None
        return self.source_provenance.source_alias_plane_index(
            source_aliases,
            axis_size,
        )

    def object_label_semantic_identity(self) -> tuple[tuple[str, Hashable], ...]:
        """Return native object-label identity fields in addition to payload metadata."""
        return (
            *ObjectLabelValue.object_label_semantic_identity(self),
            ("object_name", self.name),
            ("dimensions", self.dimensions),
            ("source_image_name", self.source_image_name),
        )

    def object_label_dense_projection_identity(
        self,
    ) -> tuple[tuple[str, Hashable], ...]:
        """Return native dense-projection identity fields for label caches."""
        return (
            *ObjectLabelValue.object_label_dense_projection_identity(self),
            ("object_name", self.name),
            ("dimensions", self.dimensions),
            ("source_image_name", self.source_image_name),
        )


class ObjectLabelValueIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from nominal object-label values."""

    value_type = ObjectLabelValue

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        label_value = cast(ObjectLabelValue, labels)
        return ObjectLabelIdDomainStrategy.for_value(label_value.labels).present_ids(
            label_value.labels
        )


class ObjectLabelStorageStrategy(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Single nominal authority for object-label storage behavior."""

    @classmethod
    def for_value(cls, labels: object) -> "ObjectLabelStorageStrategy":
        return cls.require_nominal_value(
            labels,
            context="Object-label storage",
        )

    @abstractmethod
    def storage_representation(
        self,
        labels: object,
    ) -> ObjectLabelRepresentation:
        """Return the representation owned by the label storage."""

    @abstractmethod
    def dense_data(
        self,
        labels: object,
        *,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> np.ndarray:
        """Materialize dense labels from this storage."""

    @abstractmethod
    def sparse_ijv_rows(self, labels: object) -> SparseIJVLabelRows:
        """Materialize sparse-IJV rows from this storage."""

    @abstractmethod
    def stack_planes(
        self,
        labels: Sequence[object],
        memory_type: str,
    ) -> ObjectLabelData:
        """Stack homogeneous semantic planes in this storage representation."""

    def axis_centers(
        self,
        labels: object,
        *,
        domain: Sequence[int],
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """Reduce object coordinates without erasing storage representation."""
        sparse_labels = self.sparse_ijv_rows(labels)
        array = sparse_labels.as_array()
        object_ids = array[:, sparse_labels.label_column].astype(
            np.int64,
            copy=False,
        )
        max_domain_label = max(domain, default=0)
        max_label = max(int(object_ids.max(initial=0)), max_domain_label)
        counts = np.bincount(object_ids, minlength=max_label + 1)
        coordinate_columns = (
            (
                sparse_labels.slice_column,
                sparse_labels.y_column,
                sparse_labels.x_column,
            )
            if sparse_labels.has_slice_index
            else (sparse_labels.y_column, sparse_labels.x_column)
        )
        axis_centers: list[np.ndarray] = []
        for coordinate_column in coordinate_columns:
            sums = np.bincount(
                object_ids,
                weights=array[:, coordinate_column],
                minlength=max_label + 1,
            )
            centers = np.full(max_label + 1, np.nan, dtype=np.float64)
            np.divide(sums, counts, out=centers, where=counts > 0)
            axis_centers.append(centers)
        return tuple(axis_centers), counts

    def validate_representation(
        self,
        labels: object,
        *,
        representation: ObjectLabelRepresentation,
        value_label: str,
    ) -> None:
        """Validate storage and variants against a declared representation."""
        if representation is not self.storage_representation(labels):
            raise TypeError(
                f"{value_label} requires {representation.value} payload, got "
                f"{type(labels).__name__}."
            )

    @abstractmethod
    def validate_plane_count(
        self,
        labels: object,
        *,
        plane_count: int,
        context: str,
    ) -> None:
        """Validate storage against an already-declared semantic plane count."""

    @abstractmethod
    def project_planes(
        self,
        labels: object,
        plane_indices: tuple[int, ...],
    ) -> ObjectLabelData:
        """Return selected semantic planes in requested order."""

    @abstractmethod
    def project_plane(
        self,
        labels: object,
        plane_index: int,
    ) -> ObjectLabelData:
        """Return one semantic plane."""

    @abstractmethod
    def matching_variant(
        self,
        payload: object,
        variant: object | None,
        labels: object,
    ) -> object | None:
        """Return a variant when compatible with replacement labels."""

    @abstractmethod
    def label_shape(self, labels: object) -> tuple[int, ...] | None:
        """Return a shape when this storage has dense shape semantics."""


class DenseArrayObjectLabelStorageStrategy(ObjectLabelStorageStrategy):
    """Nominal storage behavior for dense object-label arrays."""

    value_type = np.ndarray

    def storage_representation(self, labels: object) -> ObjectLabelRepresentation:
        del labels
        return ObjectLabelRepresentation.DENSE_LABELS

    def dense_data(
        self,
        labels: object,
        *,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> np.ndarray:
        del source_spatial_shape_yx
        return cast(np.ndarray, labels)

    def sparse_ijv_rows(self, labels: object) -> SparseIJVLabelRows:
        return SparseIJVLabelRows.from_dense_stack(cast(np.ndarray, labels))

    def stack_planes(
        self,
        labels: Sequence[object],
        memory_type: str,
    ) -> ObjectLabelData:
        from openhcs.core.memory import stack_runtime_slices

        return stack_runtime_slices(
            tuple(cast(np.ndarray, value) for value in labels),
            memory_type,
            0,
        )

    def validate_plane_count(
        self,
        labels: object,
        *,
        plane_count: int,
        context: str,
    ) -> None:
        array = cast(np.ndarray, labels)
        if array.ndim < 3 or int(array.shape[0]) != plane_count:
            raise ValueError(
                f"{context} declares {plane_count} plane(s), but dense label "
                f"storage has shape {array.shape!r}."
            )

    def project_planes(
        self,
        labels: object,
        plane_indices: tuple[int, ...],
    ) -> ObjectLabelData:
        return cast(np.ndarray, labels)[np.asarray(plane_indices, dtype=np.intp)]

    def project_plane(
        self,
        labels: object,
        plane_index: int,
    ) -> ObjectLabelData:
        return cast(np.ndarray, labels)[plane_index]

    def matching_variant(
        self,
        payload: object,
        variant: object | None,
        labels: object,
    ) -> object | None:
        del payload
        if variant is None:
            return None
        replacement_shape = ObjectLabelStorageStrategy.for_value(labels).label_shape(
            labels
        )
        if replacement_shape is None or self.label_shape(variant) == replacement_shape:
            return variant
        return None

    def label_shape(self, labels: object) -> tuple[int, ...]:
        return tuple(cast(np.ndarray, labels).shape)


class SparseIJVObjectLabelStorageStrategy(ObjectLabelStorageStrategy):
    """Nominal storage behavior for sparse-IJV label rows."""

    value_type = SparseIJVLabelRows

    def storage_representation(self, labels: object) -> ObjectLabelRepresentation:
        del labels
        return ObjectLabelRepresentation.SPARSE_IJV

    def dense_data(
        self,
        labels: object,
        *,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> np.ndarray:
        return cast(SparseIJVLabelRows, labels).to_dense(
            source_spatial_shape_yx=source_spatial_shape_yx,
        )

    def sparse_ijv_rows(self, labels: object) -> SparseIJVLabelRows:
        return cast(SparseIJVLabelRows, labels)

    def stack_planes(
        self,
        labels: Sequence[object],
        memory_type: str,
    ) -> ObjectLabelData:
        del memory_type
        return SparseIJVLabelRows.from_slices(
            tuple(cast(SparseIJVLabelRows, value) for value in labels)
        )

    def validate_plane_count(
        self,
        labels: object,
        *,
        plane_count: int,
        context: str,
    ) -> None:
        sparse_labels = cast(SparseIJVLabelRows, labels)
        observed_count = (
            sparse_labels.label_data_runtime_slice_count()
            if sparse_labels.has_slice_index
            else 1
        )
        if observed_count != plane_count:
            raise ValueError(
                f"{context} declares {plane_count} plane(s), but sparse label "
                f"storage carries {observed_count}."
            )

    def project_planes(
        self,
        labels: object,
        plane_indices: tuple[int, ...],
    ) -> ObjectLabelData:
        sparse_labels = cast(SparseIJVLabelRows, labels)
        if not sparse_labels.has_slice_index:
            if plane_indices != (0,):
                raise IndexError(plane_indices)
            return sparse_labels
        return SparseIJVLabelRows.from_slices(
            tuple(sparse_labels.slice(plane_index) for plane_index in plane_indices)
        )

    def project_plane(
        self,
        labels: object,
        plane_index: int,
    ) -> ObjectLabelData:
        sparse_labels = cast(SparseIJVLabelRows, labels)
        if not sparse_labels.has_slice_index:
            if plane_index != 0:
                raise IndexError(plane_index)
            return sparse_labels
        return sparse_labels.slice(plane_index)

    def matching_variant(
        self,
        payload: object,
        variant: object | None,
        labels: object,
    ) -> object | None:
        del payload
        if variant is None:
            return None
        replacement_authority = ObjectLabelStorageStrategy.for_value(labels)
        if (
            replacement_authority.storage_representation(labels)
            is ObjectLabelRepresentation.SPARSE_IJV
        ):
            return variant
        return None

    def label_shape(self, labels: object) -> None:
        del labels
        return None


class ObjectLabelValueStorageStrategy(ObjectLabelStorageStrategy):
    """ObjectLabelValue delegates storage behavior to its label-data authority."""

    value_type = ObjectLabelValue

    @staticmethod
    def label_data(labels: object) -> ObjectLabelData:
        label_value = cast(ObjectLabelValue, labels)
        if isinstance(label_value.labels, ObjectLabelValue):
            raise TypeError(
                f"{type(label_value).__name__}.labels requires label data, not "
                "another ObjectLabelValue. Pass its variant_data explicitly instead."
            )
        return label_value.labels

    def storage_representation(self, labels: object) -> ObjectLabelRepresentation:
        label_data = self.label_data(labels)
        return ObjectLabelStorageStrategy.for_value(label_data).storage_representation(
            label_data
        )

    def dense_data(
        self,
        labels: object,
        *,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> np.ndarray:
        del source_spatial_shape_yx
        label_value = cast(ObjectLabelValue, labels)
        label_data = self.label_data(label_value)
        return ObjectLabelStorageStrategy.for_value(label_data).dense_data(
            label_data,
            source_spatial_shape_yx=label_value.source_spatial_domain.source_shape_yx,
        )

    def sparse_ijv_rows(self, labels: object) -> SparseIJVLabelRows:
        label_data = self.label_data(labels)
        return ObjectLabelStorageStrategy.for_value(label_data).sparse_ijv_rows(
            label_data
        )

    def stack_planes(
        self,
        labels: Sequence[object],
        memory_type: str,
    ) -> ObjectLabelData:
        label_data = tuple(self.label_data(value) for value in labels)
        if not label_data:
            raise ValueError("Object-label plane stacking requires values.")
        return ObjectLabelStorageStrategy.for_value(label_data[0]).stack_planes(
            label_data,
            memory_type,
        )

    def validate_representation(
        self,
        labels: object,
        *,
        representation: ObjectLabelRepresentation,
        value_label: str,
    ) -> None:
        label_value = cast(ObjectLabelValue, labels)
        label_data = self.label_data(label_value)
        final_authority = ObjectLabelStorageStrategy.for_value(label_data)
        final_authority.validate_representation(
            label_data,
            representation=representation,
            value_label=value_label,
        )
        for variant_name, variant in (
            ("unedited_labels", label_value.unedited_labels),
            ("small_removed_labels", label_value.small_removed_labels),
        ):
            if variant is None:
                continue
            variant_authority = ObjectLabelStorageStrategy.for_value(variant)
            variant_authority.validate_representation(
                variant,
                representation=representation,
                value_label=f"{value_label} {variant_name}",
            )
            if variant_authority.matching_variant(variant, variant, label_data) is None:
                raise ValueError(
                    f"{value_label} {variant_name} shape "
                    f"{variant_authority.label_shape(variant)!r} does not match "
                    f"final labels shape {final_authority.label_shape(label_data)!r}."
                )

    def validate_plane_count(
        self,
        labels: object,
        *,
        plane_count: int,
        context: str,
    ) -> None:
        label_data = self.label_data(labels)
        ObjectLabelStorageStrategy.for_value(label_data).validate_plane_count(
            label_data,
            plane_count=plane_count,
            context=context,
        )

    def project_planes(
        self,
        labels: object,
        plane_indices: tuple[int, ...],
    ) -> ObjectLabelData:
        label_data = self.label_data(labels)
        return ObjectLabelStorageStrategy.for_value(label_data).project_planes(
            label_data,
            plane_indices,
        )

    def project_plane(
        self,
        labels: object,
        plane_index: int,
    ) -> ObjectLabelData:
        label_data = self.label_data(labels)
        return ObjectLabelStorageStrategy.for_value(label_data).project_plane(
            label_data,
            plane_index,
        )

    def matching_variant(
        self,
        payload: object,
        variant: object | None,
        labels: object,
    ) -> object | None:
        del payload
        if variant is None:
            return None
        return ObjectLabelStorageStrategy.for_value(variant).matching_variant(
            variant,
            variant,
            labels,
        )

    def label_shape(self, labels: object) -> tuple[int, ...] | None:
        label_data = self.label_data(labels)
        return ObjectLabelStorageStrategy.for_value(label_data).label_shape(label_data)


def object_label_dense_array(
    payload: object,
    *,
    dtype: object | None = None,
    copy: bool | None = None,
) -> np.ndarray:
    """Materialize object-label dense data through its nominal storage authority."""
    dense_data = ObjectLabelStorageStrategy.for_value(payload).dense_data(
        payload,
        source_spatial_shape_yx=None,
    )
    if copy is None:
        return np.asarray(dense_data, dtype=dtype)
    return np.array(dense_data, dtype=dtype, copy=copy)


def object_label_sparse_ijv_rows(payload: object) -> SparseIJVLabelRows:
    """Materialize sparse-IJV rows through the nominal storage authority."""
    return ObjectLabelStorageStrategy.for_value(payload).sparse_ijv_rows(payload)


def object_label_stack_planes(
    labels: Sequence[ObjectLabelData],
    memory_type: str,
) -> ObjectLabelData:
    """Stack homogeneous label planes through their nominal storage authority."""
    values = tuple(labels)
    if not values:
        raise ValueError("Object-label plane stacking requires values.")
    authority = ObjectLabelStorageStrategy.for_value(values[0])
    if any(type(value) is not type(values[0]) for value in values[1:]):
        raise TypeError(
            "Object-label plane stacking requires one nominal storage type."
        )
    return authority.stack_planes(values, memory_type)


def object_label_axis_centers(
    payload: object,
    *,
    domain: Sequence[int],
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Reduce object coordinates through the nominal storage authority."""
    return ObjectLabelStorageStrategy.for_value(payload).axis_centers(
        payload,
        domain=domain,
    )


def object_label_storage_is_sparse_ijv(payload: object) -> bool:
    """Return whether the nominal storage authority owns sparse-IJV rows."""
    authority = ObjectLabelStorageStrategy.for_value(payload)
    return (
        authority.storage_representation(payload)
        is ObjectLabelRepresentation.SPARSE_IJV
    )


def object_label_validate_plane_count(
    labels: ObjectLabelData,
    *,
    plane_count: int,
    context: str,
) -> None:
    """Validate label storage against a declared semantic plane count."""
    ObjectLabelStorageStrategy.for_value(labels).validate_plane_count(
        labels,
        plane_count=plane_count,
        context=context,
    )


def object_label_project_planes(
    labels: ObjectLabelData,
    plane_indices: tuple[int, ...],
    *,
    plane_count: int,
) -> ObjectLabelData:
    """Return an ordered plane subset through the storage authority."""
    authority = ObjectLabelStorageStrategy.for_value(labels)
    authority.validate_plane_count(
        labels,
        plane_count=plane_count,
        context="Object-label data plane projection",
    )
    return authority.project_planes(labels, plane_indices)


def object_label_project_plane(
    labels: ObjectLabelData,
    plane_index: int,
    *,
    plane_count: int,
) -> ObjectLabelData:
    """Return one exact plane through the storage authority."""
    authority = ObjectLabelStorageStrategy.for_value(labels)
    authority.validate_plane_count(
        labels,
        plane_count=plane_count,
        context="Object-label data plane projection",
    )
    return authority.project_plane(labels, plane_index)


def object_label_value_with_dense_labels(
    source: ObjectLabelValueBuildSource,
    labels: ObjectLabelData,
    *,
    domain_declaration: ObjectLabelDomainDeclaration = (
        PreserveSourceObjectLabelDomainDeclaration()
    ),
    representation: ObjectLabelRepresentation | None = None,
    source_spatial_domain: SourceSpatialDomain | None = None,
) -> ObjectLabelValue:
    """Build transformed object labels preserving the source value category."""
    declared_domain = domain_declaration.declared_domain(source, labels)
    if isinstance(source, ObjectLabelValue):
        return source.with_replacement_labels(
            labels,
            representation=representation,
            domain=declared_domain,
            source_spatial_domain=source_spatial_domain,
        )
    if not isinstance(
        source, ObjectLabelDomainMetadata
    ) and not ObjectLabelsArtifactType.payload_shape.accepts(source):
        raise TypeError(
            "Object-label replacement requires nominal object-label domain "
            "metadata or declared label data, got "
            f"{type(source).__name__}."
        )
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=declared_domain,
        representation=(
            ObjectLabelRepresentation.DENSE_LABELS
            if representation is None
            else representation
        ),
    )


class ObjectLabelSetReplacementStrategy(
    EnumKeyedStrategyMixin[ObjectLabelRepresentation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered replacement policy for ObjectLabelSet label representations."""

    representation: ClassVar[ObjectLabelRepresentation | None] = None
    representation_label: ClassVar[str | None] = None
    __registry_key__ = "representation_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "representation"
    __enum_label_attr__ = "representation_label"

    @abstractmethod
    def replacement_labels(self, labels: object) -> object:
        """Return labels compatible with this representation."""


class DenseObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Dense replacements convert through the nominal storage authority."""

    representation = ObjectLabelRepresentation.DENSE_LABELS

    def replacement_labels(self, labels: object) -> np.ndarray:
        return object_label_dense_array(labels)


class SparseIJVObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Sparse-IJV replacements use the sparse rows carried by nominal label sets."""

    representation = ObjectLabelRepresentation.SPARSE_IJV

    def replacement_labels(self, labels: object) -> object:
        return object_label_sparse_ijv_rows(labels)


def object_label_domain_for_projected_label_plane(
    source: ObjectLabelValue,
    plane_index: int,
) -> ObjectLabelDomain:
    """Return the payload-domain object IDs carried by one selected plane."""
    domain = source.object_label_domain()
    if domain.scope is ObjectLabelDomainScope.PAYLOAD:
        return domain
    plane_domain = domain.project_planes((plane_index,))
    return ObjectLabelDomain.declared(
        scope=ObjectLabelDomainScope.PAYLOAD,
        declared_object_count=plane_domain.declared_object_count,
        declared_object_ids=plane_domain.declared_object_ids,
    )


def object_label_variant_matching_labels(
    variant: object | None,
    labels: object,
) -> object | None:
    """Return a variant only when it is compatible with replacement labels."""
    if variant is None:
        return None
    return ObjectLabelStorageStrategy.for_value(variant).matching_variant(
        variant,
        variant,
        labels,
    )
