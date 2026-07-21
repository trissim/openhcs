"""Object-label reduction and PURE_2D slice aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from openhcs.core import runtime_image_values
from openhcs.core.runtime_object_labels import (
    ObjectLabelData,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
    object_label_stack_planes,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ObjectLabelPlaneDomainStrategy,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
    ObjectLabelVariant,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainAdapter,
)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelAggregation:
    """Vectorized reductions over dense object-label IDs."""

    labels: np.ndarray
    object_count: int

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        dtype: object = np.int32,
    ) -> "DenseObjectLabelAggregation":
        """Build reductions from any nominal object-label payload."""
        labels = object_label_dense_array(payload, dtype=dtype)
        if labels.size:
            object_count = int(np.max(labels))
        else:
            object_count = 0
        return cls(
            labels=labels,
            object_count=object_count,
        )

    def counts(self) -> np.ndarray:
        """Return per-object pixel counts excluding the background label."""
        return np.bincount(
            self.labels,
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def sum(self, values: object) -> np.ndarray:
        """Return per-object sums for values aligned with ``labels``."""
        return np.bincount(
            self.labels,
            weights=np.asarray(values, dtype=float),
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def maximum(self, values: object) -> np.ndarray:
        """Return per-object maxima for values aligned with ``labels``."""
        maxima = np.zeros(self.object_count + 1, dtype=float)
        np.maximum.at(maxima, self.labels, np.asarray(values, dtype=float))
        return maxima[1:]


class ObjectLabelPure2DSliceAggregator:
    """Aggregate object-label slices while preserving label-domain semantics."""

    def __init__(
        self,
        values: Sequence[ObjectLabelValue],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> None:
        self.values: tuple[ObjectLabelValue, ...] = tuple(values)
        self.memory_type = memory_type
        self.requested_plane_axis = plane_axis

    @classmethod
    def aggregate(
        cls,
        values: Sequence[ObjectLabelValue],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> ObjectLabelValue:
        values_tuple = tuple(values)
        if not values_tuple:
            raise ValueError("Object-label slice aggregation requires values.")
        value_type = type(values_tuple[0])
        if not all(type(value) is value_type for value in values_tuple):
            raise TypeError(
                "Object-label slice aggregation requires one nominal value type."
            )
        return cls(
            values_tuple,
            memory_type,
            plane_axis=plane_axis,
        ).aggregate_values()

    @property
    def first(self) -> ObjectLabelValue:
        return self.values[0]

    @property
    def representation(self) -> ObjectLabelRepresentation:
        representations = {value.representation for value in self.values}
        if len(representations) != 1:
            raise ValueError(
                "Cannot aggregate mixed object-label representations across "
                "PURE_2D slices."
            )
        return representations.pop()

    @property
    def declared_object_id_domains(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            plane_domain
            for value in self.values
            for plane_domain in self.value_plane_id_domains(value)
        )

    def value_plane_id_domains(
        self,
        value: ObjectLabelValue,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the object-id domain represented by one PURE_2D output slice."""
        domain = value.object_label_domain()
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            domain.scope
        ).identity_domains(
            value,
            domain=domain,
        )

    @property
    def domain_scope(self) -> ObjectLabelDomainScope:
        return ObjectLabelDomainScope.PLANE

    @property
    def domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(
            declared_object_id_domains=self.declared_object_id_domains,
            scope=self.domain_scope,
        )

    @property
    def plane_axis(self) -> RuntimePlaneAxis:
        return self.requested_plane_axis

    @property
    def expands_to_source_domain(self) -> bool:
        return SourceSpatialDomain.domains_have_varying_complete_placement(
            tuple(value.object_label_source_spatial_domain() for value in self.values)
        )

    def aggregate_values(self) -> ObjectLabelValue:
        return self.output_value(
            ObjectLabelVariantData(
                labels=self.aggregate_variant(ObjectLabelVariant.FINAL),
                unedited_labels=self.aggregate_optional_variant(
                    ObjectLabelVariant.UNEDITED
                ),
                small_removed_labels=self.aggregate_optional_variant(
                    ObjectLabelVariant.SMALL_REMOVED
                ),
            )
        )

    def aggregate_optional_variant(
        self,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelData | None:
        if ObjectLabelVariantData.variant_is_present(variant, self.values):
            return self.aggregate_variant(variant)
        return None

    def aggregate_variant(self, variant: ObjectLabelVariant) -> ObjectLabelData:
        return object_label_stack_planes(
            tuple(self.slice_labels(value, variant) for value in self.values),
            self.memory_type,
        )

    def slice_labels(
        self,
        value: ObjectLabelValue,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelData:
        if not self.expands_to_source_domain:
            return value.variant_data.labels_for_variant(variant)
        domain_value = self.domain_value_for_variant(value, variant)
        adapter = SourceSpatialDomainAdapter.for_value(domain_value)
        if adapter is None:
            raise TypeError(
                "Object-label aggregation requires a registered source-spatial "
                "adapter for its nominal label value."
            )
        return adapter.materialize()

    def domain_value_for_variant(
        self,
        value: ObjectLabelValue,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelValue:
        """Return a typed label value carrying the selected variant and domain."""
        return value.with_replacement_labels(
            value.variant_data.labels_for_variant(variant),
            representation=value.representation,
        )

    def output_value(
        self,
        variants: ObjectLabelVariantData,
    ) -> ObjectLabelValue:
        """Build the aggregated object-label value."""
        metadata = runtime_image_values.ImagePayloadMetadata.compose(
            self.values,
            mode=runtime_image_values.ImagePayloadMetadataCompositionMode.STACK,
        )
        return self.first.with_variants(
            variants,
            representation=self.representation,
            domain=self.domain,
            source_provenance=metadata.source_provenance,
            source_spatial_domain=metadata.source_spatial_domain,
            parent_image_source_voxel_spacing=metadata.source_voxel_spacing,
            plane_axis=self.plane_axis,
        )
