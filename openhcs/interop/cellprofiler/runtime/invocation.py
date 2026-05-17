"""Typed CellProfiler invocation records and execution-mode utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
from openhcs.core.registry_strategies import GeneratedLeafClassSpec
from openhcs.core.runtime_semantics import (
    CommonRuntimeValue,
    MeasurementImageReferenceDomain,
)
from openhcs.core.runtime_invocation import (
    ResolvedRuntimeInputRequest,
    RuntimeFunctionInvocationRequest,
    RuntimeImageExecutionContext,
    RuntimeImageRequest,
    RuntimeInvocationOptions,
    RuntimeSliceAlignedValues,
    requested_image_execution_mode,
)


CELLPROFILER_GRID_CYCLE_SCOPE_KWARG = "_cellprofiler_grid_cycle_scope"


class CellProfilerGridCycleScope(str, Enum):
    """Closed DefineGrid execution scopes from CellProfiler."""

    EACH_CYCLE = "each_cycle"
    ONCE = "once"


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInvocationOptions(RuntimeInvocationOptions):
    """Typed CellProfiler controls that are not absorbed function arguments."""

    grid_cycle_scope: CellProfilerGridCycleScope = CellProfilerGridCycleScope.EACH_CYCLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grid_cycle_scope",
            coerce_cellprofiler_grid_cycle_scope(self.grid_cycle_scope),
        )


CellProfilerImageExecutionContext = RuntimeImageExecutionContext
CellProfilerResolvedInputRequest = ResolvedRuntimeInputRequest
CellProfilerImageRequest = RuntimeImageRequest


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
        row_mapping: Mapping[str, Any],
        source_pair: CellProfilerSourceImagePair,
        *,
        retain_field: Callable[[str], bool],
    ) -> dict[str, Any]:
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
        base_type=FirstSecondCellProfilerSourcePairFeature,
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
class CellProfilerMeasurementImage(RuntimeImageExecutionContext):
    """One resolved image payload used by object measurement modules."""

    registry_key: ClassVar[str] = "cellprofiler_measurement_image"

    payload: object
    source_image_names: tuple[str, ...] = ()
    align_to_labels: bool = True
    reference_domain: CellProfilerMeasurementImageDomain = (
        CellProfilerMeasurementImageDomain.SOURCE_IMAGE
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_image_names",
            tuple(str(name) for name in self.source_image_names),
        )
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

    def source_image_pairs(self) -> tuple[CellProfilerSourceImagePair, ...]:
        """Return ordered pairwise source invocations for composed image payloads."""
        return tuple(
            CellProfilerSourceImagePair.from_parts(
                first_index=first_index,
                second_index=second_index,
                first_name=first_name,
                second_name=second_name,
            )
            for first_index, first_name in enumerate(self.source_image_names)
            for second_index, second_name in enumerate(self.source_image_names)
            if first_index < second_index
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSliceAlignedValues(RuntimeSliceAlignedValues[np.ndarray]):
    """Non-image vector payload with one value array per object-label slice."""

    def __post_init__(self) -> None:
        slices = tuple(np.asarray(value) for value in self.slices)
        if not slices:
            raise ValueError("CellProfilerSliceAlignedValues.slices cannot be empty.")
        object.__setattr__(self, "slices", slices)


def illumination_scope_uses_all_images(value: object) -> bool:
    """Return whether a CellProfiler illumination scope means all images."""
    if value is None:
        return False
    if isinstance(value, Enum):
        value = value.value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized.startswith("all")


def coerce_cellprofiler_grid_cycle_scope(
    value: object,
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
