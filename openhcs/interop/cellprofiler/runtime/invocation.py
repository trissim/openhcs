"""Typed CellProfiler invocation records and execution-mode utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Mapping
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
)
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
from openhcs.core.measurement_image_alignment import (
    MeasurementImageAlignmentSource,
    MeasurementImageLabelAlignmentStrategy,
    MeasurementLabelSourceAlignmentStrategy,
)
from openhcs.core.registry_strategies import GeneratedLeafClassSpec
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.runtime_semantics import (
    MeasurementImageReferenceDomain,
    RuntimePlaneAxisProjector,
)
from openhcs.core.source_spatial_domain import CommonRuntimeValue
from openhcs.core.runtime_invocation import (
    ResolvedRuntimeInputRequest,
    RuntimeFunctionInvocationRequest,
    RuntimeImageExecutionContext,
    RuntimeImageRequest,
    RuntimeInvocationOptions,
    RuntimeSliceAlignedValues,
    requested_image_execution_mode,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionRequest,
    ImageMetadataPayload,
    MaskedImagePayload,
    ObjectLabelMeasurementSource,
    ObjectLabelValue,
    image_payload_data,
    image_payload_metadata,
    image_payload_slice_context,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerRuntimeValue,
    MeasurementRowMapping,
)


CELLPROFILER_GRID_CYCLE_SCOPE_KWARG = "_cellprofiler_grid_cycle_scope"
CellProfilerRuntimeSettingValue = CellProfilerRuntimeValue | Enum


class CellProfilerGridCycleScope(str, Enum):
    """Closed DefineGrid execution scopes from CellProfiler."""

    EACH_CYCLE = "each_cycle"
    ONCE = "once"


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInvocationOptions(RuntimeInvocationOptions):
    """Typed CellProfiler controls that are not absorbed function arguments."""

    grid_cycle_scope: CellProfilerGridCycleScope = CellProfilerGridCycleScope.EACH_CYCLE

    def __post_init__(self) -> None:
        if not isinstance(self.grid_cycle_scope, CellProfilerGridCycleScope):
            raise TypeError(
                "CellProfilerInvocationOptions.grid_cycle_scope must be "
                "CellProfilerGridCycleScope, got "
                f"{type(self.grid_cycle_scope).__name__}."
            )


CellProfilerImageExecutionContext = RuntimeImageExecutionContext
CellProfilerResolvedInputRequest = ResolvedRuntimeInputRequest


class CellProfilerSourceIdentityMixin:
    """Shared source-image identity behavior for CellProfiler runtime records."""

    SOURCE_ALIASES_FIELD_NAME: ClassVar[str]

    source_image_name: str | None
    source_aliases: tuple[str, ...]

    @property
    def has_source_identity(self) -> bool:
        """Return whether this record names a CellProfiler source-image surface."""
        return self.source_image_name is not None or bool(self.source_aliases)

    def primary_source_image_pair(self) -> "CellProfilerSourceImagePair | None":
        """Return the source pair represented by this record, when unambiguous."""
        return CellProfilerSourceImagePair.from_source_identity(
            self.source_image_name,
            self.source_aliases,
        )

    def source_image_pairs(self) -> tuple["CellProfilerSourceImagePair", ...]:
        """Return ordered pairwise source invocations for composed image payloads."""
        return tuple(
            CellProfilerSourceImagePair.from_parts(
                first_index=first_index,
                second_index=second_index,
                first_name=first_name,
                second_name=second_name,
            )
            for first_index, first_name in enumerate(self.source_aliases)
            for second_index, second_name in enumerate(self.source_aliases)
            if first_index < second_index
        )

    def matches_single_source_name(self, name: str) -> bool:
        """Return whether this source identity is exactly one named image."""
        return name in self.single_source_names()

    def single_source_names(self) -> frozenset[str]:
        """Return source names that identify this payload as one image."""
        names: list[str] = []
        if self.source_image_name is not None:
            names.append(self.source_image_name)
        match self.source_aliases:
            case (single_alias,):
                names.append(single_alias)
            case ():
                pass
            case _:
                pass
        return frozenset(names)

    def validate_source_identity(self) -> None:
        """Validate source alias declarations without module-level helper escape."""
        if not isinstance(self.source_aliases, tuple):
            raise TypeError(
                f"{self.SOURCE_ALIASES_FIELD_NAME} must be a tuple, "
                f"got {type(self.source_aliases).__name__}."
            )
        invalid = tuple(
            type(alias).__name__
            for alias in self.source_aliases
            if not isinstance(alias, str)
        )
        if invalid:
            raise TypeError(
                f"{self.SOURCE_ALIASES_FIELD_NAME} must contain str values, "
                f"got {invalid!r}."
            )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceAxisProjectionScope:
    """Declared source-axis projection facts for one CellProfiler image alias."""

    axis_index: int
    source_axis_size: int
    source_alias: str

    @classmethod
    def from_aliases(
        cls,
        source_aliases: tuple[str, ...],
        source_alias: str,
    ) -> "CellProfilerSourceAxisProjectionScope":
        """Build a validated projection scope from an ordered source axis."""
        try:
            axis_index = source_aliases.index(source_alias)
        except ValueError as exc:
            raise ValueError(
                f"CellProfiler image request cannot project source alias "
                f"{source_alias!r} from composed source aliases {source_aliases!r}."
            ) from exc
        return cls(
            axis_index=axis_index,
            source_axis_size=len(source_aliases),
            source_alias=source_alias,
        )

    def __post_init__(self) -> None:
        if self.source_axis_size < 1:
            raise ValueError("CellProfiler source-axis size must be positive.")
        if self.axis_index < 0 or self.axis_index >= self.source_axis_size:
            raise ValueError(
                "CellProfiler source-axis projection index must be within the "
                f"declared axis size; got index {self.axis_index} for size "
                f"{self.source_axis_size}."
            )
        if not self.source_alias:
            raise ValueError("CellProfiler source-axis projection requires an alias.")

    @property
    def is_single_source_axis(self) -> bool:
        """Return whether no material source stack is needed."""
        return self.source_axis_size == 1

    @property
    def is_composed_source_axis(self) -> bool:
        """Return whether payload data must carry a source-binding axis."""
        return self.source_axis_size > 1

    def project_image_payload(self, payload: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        """Project this source-axis plane from an image payload."""
        data = image_payload_data(payload)
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "CellProfiler source-axis projection requires ndarray image data, "
                f"got {type(data).__name__} for source alias {self.source_alias!r}."
            )
        if data.ndim < 3 or data.shape[0] != self.source_axis_size:
            raise ValueError(
                "CellProfiler source-axis projection requires a leading source axis "
                f"of size {self.source_axis_size}, got shape {data.shape!r} "
                f"for source alias {self.source_alias!r}."
            )
        return image_payload_slice_context(payload, data[self.axis_index], self.axis_index)


class CellProfilerSourceAxisCardinalityProjection(
    MostDerivedContextStrategyMixin[CellProfilerSourceAxisProjectionScope],
    ABC,
):
    """Projection behavior for singleton versus composed source axes."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    stable_key_axis: ClassVar[str] = "source_axis_cardinality"
    strategy_key: ClassVar[str | None] = None

    @classmethod
    def project(
        cls,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        """Project ``payload`` according to the declared source-axis cardinality."""
        strategy = cls.for_context(
            scope,
            error_subject="CellProfiler source-axis cardinality projection",
        )
        if strategy is None:
            raise ValueError("CellProfiler source-axis cardinality requires a strategy.")
        return strategy.project_payload(payload, scope)

    @abstractmethod
    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        """Return the source-axis payload for this cardinality."""


class SingletonCellProfilerSourceAxisCardinalityProjection(
    CellProfilerSourceAxisCardinalityProjection
):
    """A single source alias already is the requested measurement image."""

    strategy_key = "single_source_axis"

    def matches(self, scope: CellProfilerSourceAxisProjectionScope) -> bool:
        return scope.is_single_source_axis

    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        del scope
        return payload


class ComposedCellProfilerSourceAxisCardinalityProjection(
    CellProfilerSourceAxisCardinalityProjection
):
    """A composed source axis must be projected from payload data."""

    strategy_key = "composed_source_axis"

    def matches(self, scope: CellProfilerSourceAxisProjectionScope) -> bool:
        return scope.is_composed_source_axis

    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        return CellProfilerSourceAxisPayloadProjection.project(payload, scope)


class CellProfilerSourceAxisPayloadProjection(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project one source-binding plane from a composed image payload."""

    stable_key_axis: ClassVar[str] = "value_type_label"

    @classmethod
    def project(
        cls,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        """Return one source-axis image while preserving payload context."""
        strategy = cls.for_nominal_value(payload)
        if strategy is None:
            raise TypeError(
                "CellProfiler source-axis projection has no registered strategy for "
                f"{type(payload).__name__}."
            )
        return strategy.project_payload(payload, scope)

    @abstractmethod
    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        """Return one source-axis image from a supported payload value."""


class ImagePayloadSourceAxisProjection(CellProfilerSourceAxisPayloadProjection):
    """Project raw/contextual image payloads with a leading source axis."""

    value_type = (np.ndarray, ImageMetadataPayload, MaskedImagePayload)

    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        return scope.project_image_payload(payload)


class AlignedImageStackSourceAxisProjection(CellProfilerSourceAxisPayloadProjection):
    """Project each aligned slice, then re-compose the aligned payload."""

    value_type = AlignedImageStack

    def project_payload(
        self,
        payload: CellProfilerRuntimeValue,
        scope: CellProfilerSourceAxisProjectionScope,
    ) -> CellProfilerRuntimeValue:
        if not isinstance(payload, AlignedImageStack):
            raise TypeError(
                "Aligned source-axis projection requires AlignedImageStack, "
                f"got {type(payload).__name__}."
            )
        projected_slices = tuple(
            CellProfilerSourceAxisPayloadProjection.project(
                slice_payload,
                scope,
            )
            for slice_payload in payload.slices
        )
        return compose_aligned_image_payload(
            f"{scope.source_alias} measurement source-axis projection",
            projected_slices,
        ).payload


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerImageRequest(
    CellProfilerSourceIdentityMixin,
    RuntimeImageRequest,
):
    """CellProfiler image invocation payload with ordered source-plane aliases."""

    SOURCE_ALIASES_FIELD_NAME: ClassVar[str] = "CellProfilerImageRequest.source_aliases"

    source_aliases: tuple[str, ...] = ()
    projects_runtime_slice_kwargs: bool = True

    def __post_init__(self) -> None:
        self.validate_source_identity()

    def owns_source_axis(self, source_aliases: tuple[str, ...]) -> bool:
        """Return whether this request declares exactly the requested source axis."""
        return bool(source_aliases) and tuple(source_aliases) == self.source_aliases

    def source_axis_payload(self, source_alias: str) -> CellProfilerRuntimeValue:
        """Return one source image projected from this request's source axis."""
        return CellProfilerSourceAxisCardinalityProjection.project(
            self.payload,
            CellProfilerSourceAxisProjectionScope.from_aliases(
                self.source_aliases,
                source_alias,
            ),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceImageEndpoint:
    """One source image endpoint inside an ordered CellProfiler source pair."""

    index: int
    display_name: str

    name = AliasProperty("display_name")

    def invocation_kwarg(self) -> int:
        """Return this endpoint's CellProfiler channel index."""
        return self.index


@dataclass(frozen=True, slots=True)
class CellProfilerSourceImagePair:
    """Ordered pair of source-image endpoints inside a composed CP payload."""

    first: CellProfilerSourceImageEndpoint
    second: CellProfilerSourceImageEndpoint
    runtime_pair: RuntimeMeasurementSourcePair

    @classmethod
    def from_source_identity(
        cls,
        source_image_name: str | None,
        source_aliases: tuple[str, ...],
    ) -> "CellProfilerSourceImagePair | None":
        """Return the primary source pair implied by source identity fields."""
        match source_aliases:
            case (first_name, second_name):
                return cls.from_parts(
                    first_index=0,
                    second_index=1,
                    first_name=first_name,
                    second_name=second_name,
                )
            case _:
                return cls.from_source_image_name(source_image_name)

    @classmethod
    def from_parts(
        cls,
        *,
        first_index: int,
        second_index: int,
        first_name: str,
        second_name: str,
    ) -> "CellProfilerSourceImagePair":
        """Build a source pair from endpoint indexes and display names."""
        return cls(
            first=CellProfilerSourceImageEndpoint(first_index, first_name),
            second=CellProfilerSourceImageEndpoint(second_index, second_name),
            runtime_pair=RuntimeMeasurementSourcePair(first_name, second_name),
        )

    @classmethod
    def from_source_image_name(
        cls,
        source_image_name: str | None,
    ) -> "CellProfilerSourceImagePair | None":
        """Decode a composed source-image name into a pair invocation identity."""
        if source_image_name is None:
            return None
        source_parts = tuple(part for part in source_image_name.split("__") if part)
        if len(source_parts) != 2:
            return None
        first_name, second_name = source_parts
        return cls.from_parts(
            first_index=0,
            second_index=1,
            first_name=first_name,
            second_name=second_name,
        )

    @property
    def source_image_name(self) -> str:
        """Return CellProfiler's table-level source identity for this pair."""
        return RuntimeMeasurementSourcePair.source_pair_name(
            self.first.display_name,
            self.second.display_name,
        )

    def invocation_kwargs(
        self,
        *,
        first_channel_kwarg: str,
        second_channel_kwarg: str,
    ) -> dict[str, int]:
        """Lower this source-pair invocation to CellProfiler channel kwargs."""
        return {
            first_channel_kwarg: self.first.invocation_kwarg(),
            second_channel_kwarg: self.second.invocation_kwarg(),
        }


class CellProfilerSourcePairFeature(ABC, metaclass=AutoRegisterMeta):
    """CellProfiler feature naming semantics for ordered source-image pairs."""

    __registry_key__ = "source_field"
    __skip_if_no_key__ = True

    source_field: ClassVar[str | None] = None
    feature_family: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def all(cls) -> tuple["CellProfilerSourcePairFeature", ...]:
        """Return registered source-pair feature policies in declaration order."""
        return tuple(
            feature_type()
            for feature_type in cls.__registry__.values()
            if feature_type.source_field is not None
        )

    @classmethod
    @lru_cache(maxsize=None)
    def source_field_names(cls) -> frozenset[str]:
        """Return raw result fields owned by source-pair feature policies."""
        return frozenset(feature.source_field_name for feature in cls.all())

    @classmethod
    @lru_cache(maxsize=None)
    def runtime_feature_names_for_pair(
        cls,
        source_pair: CellProfilerSourceImagePair,
    ) -> Mapping[str, str]:
        """Return raw-field to runtime-feature names for one source pair."""
        return {
            feature.source_field_name: feature.runtime_feature_name(source_pair)
            for feature in cls.all()
        }

    @classmethod
    def project_row_for_pair(
        cls,
        row_mapping: MeasurementRowMapping,
        source_pair: CellProfilerSourceImagePair,
        *,
        retain_field: Callable[[str], bool],
    ) -> dict[str, CellProfilerRuntimeValue]:
        """Return one row with source-pair fields projected to runtime names."""
        source_field_names = cls.source_field_names()
        if not (source_field_names & row_mapping.keys()):
            return dict(row_mapping)
        projected = {
            field_name: value
            for field_name, value in row_mapping.items()
            if retain_field(field_name)
        }
        for source_field_name, runtime_feature_name in (
            cls.runtime_feature_names_for_pair(source_pair).items()
        ):
            if source_field_name not in row_mapping:
                continue
            projected[runtime_feature_name] = row_mapping[source_field_name]
        return projected

    @property
    def source_field_name(self) -> str:
        """Return the raw absorbed-function field represented by this feature."""
        if self.source_field is None:
            raise TypeError(f"{type(self).__name__} does not declare source_field.")
        return self.source_field

    @property
    def feature_family_name(self) -> str:
        """Return the CellProfiler measurement feature family."""
        if self.feature_family is None:
            raise TypeError(f"{type(self).__name__} does not declare feature_family.")
        return self.feature_family

    def runtime_feature_name(self, source_pair: CellProfilerSourceImagePair) -> str:
        """Return the CellProfiler measurement column for this source pair."""
        first_name, second_name = self.source_names(source_pair)
        return f"Correlation_{self.feature_family_name}_{first_name}_{second_name}"

    @abstractmethod
    def source_names(
        self,
        source_pair: CellProfilerSourceImagePair,
    ) -> tuple[str, str]:
        """Return source display names in CellProfiler's feature orientation."""


class FirstSecondCellProfilerSourcePairFeature(CellProfilerSourcePairFeature):
    """Feature policy whose CellProfiler column uses first, then second source."""

    source_field = None

    def source_names(
        self,
        source_pair: CellProfilerSourceImagePair,
    ) -> tuple[str, str]:
        return source_pair.first.name, source_pair.second.name


class SecondFirstCellProfilerSourcePairFeature(CellProfilerSourcePairFeature):
    """Feature policy whose CellProfiler column uses second, then first source."""

    source_field = None

    def source_names(
        self,
        source_pair: CellProfilerSourceImagePair,
    ) -> tuple[str, str]:
        return source_pair.second.name, source_pair.first.name


for _source_pair_feature_spec in (
    GeneratedLeafClassSpec(
        class_name="CellProfilerCorrelationFeature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={
            "source_field": "correlation",
            "feature_family": "Correlation",
        },
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerSlopeFeature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "slope", "feature_family": "Slope"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerReverseSlopeFeature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={"source_field": "slope_reverse", "feature_family": "Slope"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerOverlapFeature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "overlap", "feature_family": "Overlap"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerK1Feature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "k1", "feature_family": "K"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerK2Feature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={"source_field": "k2", "feature_family": "K"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerMandersM1Feature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "manders_m1", "feature_family": "Manders"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerMandersM2Feature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={"source_field": "manders_m2", "feature_family": "Manders"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerRWC1Feature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "rwc1", "feature_family": "RWC"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerRWC2Feature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={"source_field": "rwc2", "feature_family": "RWC"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerCostesM1Feature",
        base_type=FirstSecondCellProfilerSourcePairFeature,
        attributes={"source_field": "costes_m1", "feature_family": "Costes"},
    ),
    GeneratedLeafClassSpec(
        class_name="CellProfilerCostesM2Feature",
        base_type=SecondFirstCellProfilerSourcePairFeature,
        attributes={"source_field": "costes_m2", "feature_family": "Costes"},
    ),
):
    _source_pair_feature_spec.declare_in(globals())


CellProfilerInvocationRequest = RuntimeFunctionInvocationRequest


CellProfilerMeasurementImageDomain = MeasurementImageReferenceDomain


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerMeasurementImage(
    CellProfilerSourceIdentityMixin,
    RuntimeImageExecutionContext,
    MeasurementImageAlignmentSource,
):
    """One resolved image payload used by object measurement modules."""

    registry_key: ClassVar[str] = "cellprofiler_measurement_image"
    SOURCE_ALIASES_FIELD_NAME: ClassVar[str] = (
        "CellProfilerMeasurementImage.source_aliases"
    )

    payload: CellProfilerRuntimeValue
    source_aliases: tuple[str, ...] = ()
    align_to_labels: bool = True
    reference_domain: CellProfilerMeasurementImageDomain = (
        CellProfilerMeasurementImageDomain.SOURCE_IMAGE
    )

    def __post_init__(self) -> None:
        self.validate_source_identity()
        if not isinstance(self.reference_domain, CellProfilerMeasurementImageDomain):
            raise TypeError(
                "CellProfilerMeasurementImage.reference_domain must be "
                "CellProfilerMeasurementImageDomain, got "
                f"{type(self.reference_domain).__name__}."
            )

    @classmethod
    def shared_source_image_name(
        cls,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
    ) -> str | None:
        """Return table-level source identity only when all images share one source."""
        return CommonRuntimeValue.from_values(
            image.source_image_name for image in measurement_images
        ).single

    @classmethod
    def shared_source_payload(
        cls,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
    ) -> CellProfilerRuntimeValue | None:
        """Return a table-level source payload only for one shared provenance."""
        if not measurement_images:
            return None
        provenances = tuple(
            image_payload_metadata(image.payload).source_provenance.equality_identity
            for image in measurement_images
        )
        if CommonRuntimeValue.from_values(provenances).single is None:
            return None
        return measurement_images[0].payload

    @classmethod
    def composed_source_metadata(
        cls,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
    ) -> ImagePayloadMetadata | None:
        """Return source metadata composed in runtime measurement-image order."""
        if not measurement_images:
            return None
        metadata = ImagePayloadMetadataCompositionRequest(
            tuple(image.payload for image in measurement_images)
        ).metadata()
        if not metadata.has_values:
            return None
        return metadata

    @property
    def alignment_image(self) -> CellProfilerRuntimeValue:
        """Return this measurement image payload for alignment."""
        return self.payload

    @property
    def alignment_reference_domain(self) -> CellProfilerMeasurementImageDomain:
        """Return the reference domain that owns alignment shape decisions."""
        return self.reference_domain

    @property
    def alignment_source_aliases(self) -> tuple[str, ...]:
        """Return source aliases for source-binding projection."""
        return self.source_aliases

    @property
    def alignment_image_name(self) -> str:
        """Return a stable name for measurement-image alignment diagnostics."""
        source_image_name = self.source_image_name
        if source_image_name is not None:
            return source_image_name
        return "measurement image"

    def with_alignment_image(
        self,
        image: CellProfilerRuntimeValue,
    ) -> "CellProfilerMeasurementImage":
        """Return this source with projected image data and identical provenance."""
        return replace(self, payload=image)

    def align_labels_to_source(
        self,
        label_payload: ObjectLabelValue,
        labels: ObjectLabelMeasurementSource | None = None,
        *,
        plane_projector: RuntimePlaneAxisProjector | None = None,
    ) -> CellProfilerRuntimeValue:
        """Project labels into this measurement image's source domain."""
        label_data = labels
        if label_data is None:
            label_data = object_label_dense_array(label_payload)
        return MeasurementLabelSourceAlignmentStrategy.align_request_labels_to_image_source(
            self.alignment_request(
                labels=label_data,
                label_payload=label_payload,
                plane_projector=plane_projector,
            )
        )

    def align_image_to_labels(
        self,
        label_payload: ObjectLabelValue,
        labels: ObjectLabelMeasurementSource,
        *,
        plane_projector: RuntimePlaneAxisProjector | None = None,
    ) -> CellProfilerRuntimeValue:
        """Project this measurement image payload into the supplied label domain."""
        return MeasurementImageLabelAlignmentStrategy.align(
            self.alignment_request(
                labels=labels,
                label_payload=label_payload,
                plane_projector=plane_projector,
            )
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSliceAlignedValues(RuntimeSliceAlignedValues[np.ndarray]):
    """Non-image vector payload with one value array per object-label slice."""

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("CellProfilerSliceAlignedValues.slices cannot be empty.")
        if not isinstance(self.slices, tuple):
            raise TypeError(
                "CellProfilerSliceAlignedValues.slices must be a tuple, got "
                f"{type(self.slices).__name__}."
            )
        invalid = tuple(
            type(value).__name__
            for value in self.slices
            if not isinstance(value, np.ndarray)
        )
        if invalid:
            raise TypeError(
                "CellProfilerSliceAlignedValues.slices must contain ndarray values, "
                f"got {invalid!r}."
            )


def illumination_scope_uses_all_images(value: CellProfilerRuntimeSettingValue) -> bool:
    """Return whether a CellProfiler illumination scope means all images."""
    if value is None:
        return False
    if isinstance(value, Enum):
        value = value.value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized.startswith("all")


def coerce_cellprofiler_grid_cycle_scope(
    value: CellProfilerRuntimeSettingValue,
    *,
    default: CellProfilerGridCycleScope = CellProfilerGridCycleScope.EACH_CYCLE,
) -> CellProfilerGridCycleScope:
    """Coerce CellProfiler's grid scope setting into a closed runtime enum."""
    if value is None:
        return default
    if isinstance(value, CellProfilerGridCycleScope):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if normalized in {"each", "each_cycle"}:
        return CellProfilerGridCycleScope.EACH_CYCLE
    if normalized == "once":
        return CellProfilerGridCycleScope.ONCE
    return CellProfilerGridCycleScope(normalized)
