"""Zernike backend strategies for CellProfiler-compatible measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import hashlib
import logging
import math
import os
from pathlib import Path
import pickle
import time
from types import MappingProxyType
from typing import ClassVar, TypeAlias

import numpy as np
import centrosome.cpmorphology
import centrosome.zernike
from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from numba import njit
from scipy import ndimage as ndi

from openhcs.constants.constants import MemoryType
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementRowValueField,
    RuntimeMeasurementFeature,
    RuntimeMeasurementIndexedDescriptorDeclaration,
    coerce_enum,
)
from openhcs.core.equivalence.measurement_features import (
    RuntimeMeasurementIndexedDescriptorEquivalence,
)
from openhcs.core.runtime_equivalence import (
    MeasurementFeatureStabilityPolicy,
    ShapeDescriptorSparseNumericTolerance,
    SparseNumericCounterToleranceProfile,
)
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.runtime_values import ColumnarRows, project_image_mask_to_data_domain
from openhcs.core.public_api import public_names_from_objects
from openhcs.interop.cellprofiler.module_declarations import (
    IntensityFeature,
    ShapeDescriptorFeature,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.label_geometry import (
    minimum_enclosing_circle_from_labels,
)
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)

_INTENSITY_DEBUG_TRACE_DIR_ENV = "OPENHCS_ZERNIKE_INTENSITY_DEBUG_TRACE_DIR"
logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


ZernikeMomentIndexes: TypeAlias = tuple[tuple[int, int], ...]
ShapeZernikeMoments: TypeAlias = tuple[ZernikeMomentIndexes, np.ndarray]
IntensityZernikeMoments: TypeAlias = tuple[
    ZernikeMomentIndexes,
    np.ndarray,
    np.ndarray,
]


class ObjectZernikeDescriptorFeature(str, Enum):
    """CellProfiler object Zernike descriptor families owned by this backend."""

    SHAPE = "Zernike"
    INTENSITY_MAGNITUDE = "IntensityDistribution_ZernikeMagnitude"
    INTENSITY_PHASE = "IntensityDistribution_ZernikePhase"


class ObjectIntensityZernikeFeatureNameStrategy(
    EnumKeyedStrategyMixin[ObjectZernikeDescriptorFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Render intensity Zernike feature families with nominal dispatch."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "feature"
    feature: ClassVar[ObjectZernikeDescriptorFeature]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def family_name(self) -> str:
        """Return the external feature-family name for this descriptor."""

    def descriptor_family_names(self) -> tuple[str, ...]:
        """Return feature-family spellings accepted for descriptor parsing."""
        family_name = self.family_name()
        source_split_family_name = "".join(
            (
                f"_{char}" if index and char.isupper() else char
                for index, char in enumerate(family_name)
            )
        )
        return (
            f"IntensityDistribution_{family_name}",
            family_name,
            source_split_family_name,
        )

    def feature_name(self, *, degree: int, repetition: int) -> str:
        """Return CP-compatible long-form intensity Zernike feature identity."""
        return f"IntensityDistribution_{self.family_name()}_{int(degree)}_{int(repetition)}"


class ObjectIntensityZernikeMagnitudeFeatureNameStrategy(
    ObjectIntensityZernikeFeatureNameStrategy
):
    """Render intensity Zernike magnitude rows."""

    feature = ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE

    def family_name(self) -> str:
        return "ZernikeMagnitude"


class ObjectIntensityZernikePhaseFeatureNameStrategy(
    ObjectIntensityZernikeFeatureNameStrategy
):
    """Render intensity Zernike phase rows."""

    feature = ObjectZernikeDescriptorFeature.INTENSITY_PHASE

    def family_name(self) -> str:
        return "ZernikePhase"


@lru_cache(maxsize=None)
def indexed_object_intensity_zernike_feature_name(
    feature: ObjectZernikeDescriptorFeature | str, *, degree: int, repetition: int
) -> str:
    """Return CP-compatible long-form intensity Zernike feature identity."""
    feature = coerce_enum(
        ObjectZernikeDescriptorFeature,
        feature,
        "indexed_object_intensity_zernike_feature_name.feature",
    )
    return ObjectIntensityZernikeFeatureNameStrategy.for_enum_member(
        feature
    ).feature_name(degree=degree, repetition=repetition)


@dataclass(frozen=True, slots=True)
class IndexedObjectZernikeDescriptor:
    """Parsed identity for an indexed CellProfiler object Zernike descriptor."""

    family: ObjectZernikeDescriptorFeature
    degree: int
    repetition: int

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
        *,
        family: ObjectZernikeDescriptorFeature | None = None,
        family_names: Sequence[str] | None = None,
        allow_source_qualified: bool = False,
    ) -> "IndexedObjectZernikeDescriptor | None":
        normalized_parts = tuple(
            (
                part
                for part in str(feature_name)
                .strip()
                .lower()
                .replace("-", "_")
                .split("_")
                if part
            )
        )
        for candidate_family in (
            (family,) if family is not None else tuple(ObjectZernikeDescriptorFeature)
        ):
            candidate_family_names = (
                tuple(family_names)
                if family_names is not None
                else (candidate_family.value,)
            )
            for family_name in candidate_family_names:
                family_parts = tuple(
                    (
                        part
                        for part in str(family_name)
                        .lower()
                        .replace("-", "_")
                        .split("_")
                        if part
                    )
                )
                family_prefixes = (family_parts, ("".join(family_parts),))
                for family_prefix in family_prefixes:
                    minimum_length = len(family_prefix) + 2
                    if len(normalized_parts) < minimum_length:
                        continue
                    if (
                        len(normalized_parts) != minimum_length
                        and not allow_source_qualified
                    ):
                        continue
                    if normalized_parts[: len(family_prefix)] != family_prefix:
                        continue
                    degree_text, repetition_text = normalized_parts[-2:]
                    if not degree_text.isdecimal() or not repetition_text.isdecimal():
                        continue
                    return cls(
                        family=candidate_family,
                        degree=int(degree_text),
                        repetition=int(repetition_text),
                    )
        return None


class ZernikeDescriptorSparseNumericTolerance(SparseNumericCounterToleranceProfile):
    """Zernike descriptor sparse tolerance owned by CellProfiler Zernike support."""

    profile_key = "cellprofiler_zernike_descriptor"

    @classmethod
    def matches_descriptor(cls, descriptor: object) -> bool:
        """Return whether this profile owns ``descriptor`` tolerance."""
        return isinstance(descriptor, IndexedObjectZernikeDescriptor)

    def tolerance(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        descriptor: object | None,
    ) -> tuple[float, float, int, float]:
        if not isinstance(descriptor, IndexedObjectZernikeDescriptor):
            raise TypeError(
                "Zernike descriptor sparse tolerance requires "
                "IndexedObjectZernikeDescriptor."
            )
        abs_tolerance = (
            policy.zernike_descriptor_phase_abs_tolerance
            if descriptor.family is ObjectZernikeDescriptorFeature.INTENSITY_PHASE
            else policy.zernike_descriptor_magnitude_abs_tolerance
        )
        return (
            abs_tolerance,
            policy.zernike_descriptor_rel_tolerance,
            policy.object_boundary_jitter_max_unstable_values,
            policy.object_boundary_jitter_max_unstable_fraction,
        )


class ObjectZernikeDescriptorDeclaration(
    RuntimeMeasurementIndexedDescriptorDeclaration,
    RuntimeMeasurementIndexedDescriptorEquivalence,
    ABC,
):
    """Registered parser/render/equivalence declaration for one Zernike family."""

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> IndexedObjectZernikeDescriptor | None:
        descriptor = IndexedObjectZernikeDescriptor.from_feature_name(
            feature_name,
            family=cls.descriptor_family(),
            family_names=cls.descriptor_family_names(),
            allow_source_qualified=cls.allows_source_qualified_feature_name(),
        )
        if descriptor is None or descriptor.family is not cls.descriptor_family():
            return None
        return descriptor

    @classmethod
    def feature_name(
        cls,
        descriptor: object,
    ) -> str:
        if not isinstance(descriptor, IndexedObjectZernikeDescriptor):
            raise TypeError(
                f"{cls.__name__}.feature_name requires IndexedObjectZernikeDescriptor."
            )
        if descriptor.family is not cls.descriptor_family():
            raise ValueError(
                f"{cls.__name__} cannot render descriptor family "
                f"{descriptor.family.value!r}."
            )
        return cls._feature_name(descriptor)

    @classmethod
    def indexed_suffix_token_width(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return trailing degree/repetition token width for Zernike descriptors."""
        descriptor = cls.from_feature_name("_".join(feature_tokens))
        if descriptor is None:
            return None
        return 2

    @classmethod
    def allows_source_qualified_feature_name(cls) -> bool:
        """Return whether this descriptor family carries source tokens in names."""
        return False

    @classmethod
    @abstractmethod
    def descriptor_family(cls) -> ObjectZernikeDescriptorFeature:
        """Return the concrete Zernike family this declaration owns."""

    @classmethod
    def descriptor_family_names(cls) -> tuple[str, ...]:
        """Return external feature-family prefixes accepted by this declaration."""
        return (cls.descriptor_family().value,)

    @classmethod
    @abstractmethod
    def _feature_name(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
    ) -> str:
        """Return the CP feature name for ``descriptor``."""

    @classmethod
    def descriptor_snapshots_comparable(
        cls,
        descriptor: object,
        key: object,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        if not isinstance(descriptor, IndexedObjectZernikeDescriptor):
            raise TypeError(
                f"{cls.__name__}.descriptor_snapshots_comparable requires "
                "IndexedObjectZernikeDescriptor."
            )
        if descriptor.family is not cls.descriptor_family():
            return False
        return cls._descriptor_snapshots_stable(
            descriptor,
            key,
            reference,
            candidate,
            policy,
        )

    @classmethod
    @abstractmethod
    def _descriptor_snapshots_stable(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
        key: object,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether the descriptor's supporting facts are stable."""


class ShapeObjectZernikeDescriptorDeclaration(ObjectZernikeDescriptorDeclaration):
    """Shape Zernike descriptor declaration owned by CellProfiler Zernike support."""

    descriptor_key = "cellprofiler_shape_zernike"

    @classmethod
    def descriptor_family(cls) -> ObjectZernikeDescriptorFeature:
        return ObjectZernikeDescriptorFeature.SHAPE

    @classmethod
    def _feature_name(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
    ) -> str:
        return "Zernike_{0}_{1}".format(descriptor.degree, descriptor.repetition)

    @classmethod
    def descriptor_values_equivalent(
        cls,
        descriptor: object,
        key: object,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del descriptor, key
        return ShapeDescriptorSparseNumericTolerance.equivalent(left, right, policy)

    @classmethod
    def _descriptor_snapshots_stable(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
        key: object,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del descriptor
        if not policy.allow_unstable_shape_descriptors:
            return False
        return MeasurementFeatureStabilityPolicy(
            key,
            reference,
            candidate,
            policy,
        ).shape_descriptor_geometry_is_stable()


class ShapeZernikeFeatureAuthority(ShapeDescriptorFeature):
    """Shape-Zernike feature taxonomy owned by CellProfiler Zernike support."""

    @classmethod
    def shape_zernike_feature_name(cls, *, degree: int, repetition: int) -> str:
        """Return one CP shape-Zernike feature name through this authority."""
        return ShapeObjectZernikeDescriptorDeclaration.feature_name(
            IndexedObjectZernikeDescriptor(
                ObjectZernikeDescriptorFeature.SHAPE,
                int(degree),
                int(repetition),
            )
        )

    @classmethod
    def shape_zernike_feature_names(cls, *, max_order: int) -> tuple[str, ...]:
        """Return CP shape-Zernike feature names through this authority."""
        return tuple(
            cls.shape_zernike_feature_name(degree=degree, repetition=repetition)
            for degree in range(int(max_order) + 1)
            for repetition in range(degree % 2, degree + 1, 2)
        )


class ShapeZernikeMeasurementFeature(RuntimeMeasurementFeature):
    """Feature families emitted for CellProfiler shape Zernike descriptors."""

    ZERNIKE = (
        "Zernike",
        (),
        (ShapeZernikeFeatureAuthority,),
        (ShapeObjectZernikeDescriptorDeclaration,),
    )


ShapeZernikeFeatureAuthority.MeasurementFeature = ShapeZernikeMeasurementFeature


class IntensityMagnitudeObjectZernikeDescriptorDeclaration(
    ObjectZernikeDescriptorDeclaration
):
    """Intensity Zernike magnitude descriptor declaration."""

    descriptor_key = "cellprofiler_intensity_zernike_magnitude"

    @classmethod
    def descriptor_family(cls) -> ObjectZernikeDescriptorFeature:
        return ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE

    @classmethod
    def allows_source_qualified_feature_name(cls) -> bool:
        return True

    @classmethod
    def _feature_name(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
    ) -> str:
        return indexed_object_intensity_zernike_feature_name(
            cls.descriptor_family(),
            degree=descriptor.degree,
            repetition=descriptor.repetition,
        )

    @classmethod
    def descriptor_family_names(cls) -> tuple[str, ...]:
        return ObjectIntensityZernikeFeatureNameStrategy.for_enum_member(
            cls.descriptor_family()
        ).descriptor_family_names()

    @classmethod
    def descriptor_values_equivalent(
        cls,
        descriptor: object,
        key: object,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del key
        return ZernikeDescriptorSparseNumericTolerance.equivalent(
            left,
            right,
            policy,
            descriptor=descriptor,
        )

    @classmethod
    def _descriptor_snapshots_stable(
        cls,
        descriptor: IndexedObjectZernikeDescriptor,
        key: object,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del descriptor
        if not policy.allow_unstable_zernike_descriptors:
            return False
        return MeasurementFeatureStabilityPolicy(
            key,
            reference,
            candidate,
            policy,
        ).object_count_values_stable()


class IntensityPhaseObjectZernikeDescriptorDeclaration(
    IntensityMagnitudeObjectZernikeDescriptorDeclaration
):
    """Intensity Zernike phase descriptor declaration."""

    descriptor_key = "cellprofiler_intensity_zernike_phase"

    @classmethod
    def descriptor_family(cls) -> ObjectZernikeDescriptorFeature:
        return ObjectZernikeDescriptorFeature.INTENSITY_PHASE


class IntensityZernikeFeatureAuthority(IntensityFeature):
    """Intensity-Zernike feature taxonomy owned by CellProfiler Zernike support."""


class IntensityZernikeMeasurementFeature(RuntimeMeasurementFeature):
    """Feature families emitted for CellProfiler intensity Zernike descriptors."""

    ZERNIKE_MAGNITUDE = (
        "IntensityDistribution_ZernikeMagnitude",
        (),
        (IntensityZernikeFeatureAuthority,),
        (IntensityMagnitudeObjectZernikeDescriptorDeclaration,),
    )
    ZERNIKE_PHASE = (
        "IntensityDistribution_ZernikePhase",
        (),
        (IntensityZernikeFeatureAuthority,),
        (IntensityPhaseObjectZernikeDescriptorDeclaration,),
    )


IntensityZernikeFeatureAuthority.MeasurementFeature = IntensityZernikeMeasurementFeature


@dataclass(frozen=True)
class _ZernikeLabelGeometry:
    centers: np.ndarray
    radii: np.ndarray
    y_coords: np.ndarray
    x_coords: np.ndarray
    label_values: np.ndarray
    raw_label_values: np.ndarray


@dataclass(frozen=True)
class ZernikeIntensityDebugArrayTrace:
    """Content identity for one array role in an intensity-Zernike debug trace."""

    shape: tuple[int, ...]
    dtype: str
    digest: bytes

    @classmethod
    def from_array(cls, array: np.ndarray) -> "ZernikeIntensityDebugArrayTrace":
        dtype, shape, digest = _array_content_key(array)
        return cls(shape=shape, dtype=dtype, digest=digest)


@dataclass(frozen=True)
class ZernikeIntensityDebugTrace:
    """Object-indexed Zernike state emitted only when debug tracing is enabled."""

    backend_provider: CellProfilerBackendProvider
    image: ZernikeIntensityDebugArrayTrace
    labels: ZernikeIntensityDebugArrayTrace
    max_order: int
    object_ids: np.ndarray
    zernike_numbers: tuple[tuple[int, int], ...]
    centers: np.ndarray
    radii: np.ndarray
    areas: np.ndarray
    y_coords: np.ndarray
    x_coords: np.ndarray
    label_values: np.ndarray
    pixel_values: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray

    @classmethod
    def from_intensity_measurement(
        cls,
        *,
        backend_provider: CellProfilerBackendProvider,
        image: np.ndarray,
        labels: np.ndarray,
        max_order: int,
        object_ids: np.ndarray,
        zernike_numbers: tuple[tuple[int, int], ...],
        centers: np.ndarray,
        radii: np.ndarray,
        areas: np.ndarray,
        y_coords: np.ndarray,
        x_coords: np.ndarray,
        label_values: np.ndarray,
        pixel_values: np.ndarray,
        magnitudes: np.ndarray,
        phases: np.ndarray,
    ) -> "ZernikeIntensityDebugTrace":
        return cls(
            backend_provider=backend_provider,
            image=ZernikeIntensityDebugArrayTrace.from_array(image),
            labels=ZernikeIntensityDebugArrayTrace.from_array(labels),
            max_order=int(max_order),
            object_ids=np.ascontiguousarray(object_ids, dtype=np.int32),
            zernike_numbers=zernike_numbers,
            centers=np.ascontiguousarray(centers, dtype=np.float64),
            radii=np.ascontiguousarray(radii, dtype=np.float64),
            areas=np.ascontiguousarray(areas, dtype=np.float64),
            y_coords=np.ascontiguousarray(y_coords, dtype=np.int64),
            x_coords=np.ascontiguousarray(x_coords, dtype=np.int64),
            label_values=np.ascontiguousarray(label_values, dtype=np.int32),
            pixel_values=np.ascontiguousarray(pixel_values, dtype=np.float64),
            magnitudes=np.ascontiguousarray(magnitudes, dtype=np.float64),
            phases=np.ascontiguousarray(phases, dtype=np.float64),
        )

    def write_if_enabled(self) -> Path | None:
        trace_dir_text = os.environ.get(_INTENSITY_DEBUG_TRACE_DIR_ENV)
        if trace_dir_text is None or not trace_dir_text.strip():
            return None
        trace_dir = Path(trace_dir_text)
        trace_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"zernike_intensity_{os.getpid()}_{time.time_ns()}_"
            f"{self.backend_provider.value}_{self.object_ids.size}_"
            f"{self.max_order}_{self.image.digest.hex()}_"
            f"{self.labels.digest.hex()}.pkl"
        )
        path = trace_dir / filename
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path


@dataclass(frozen=True, slots=True)
class IntensityZernikeMeasurementRowsRequest:
    """Backend request for long-form intensity-Zernike measurement rows."""

    image: np.ndarray
    labels: np.ndarray
    max_order: int
    include_phase: bool
    image_mask: np.ndarray | None = None
    object_ids: Sequence[int] | None = None
    slice_index: int | None = None
    row_identity: MeasurementObjectRowIdentity | None = None
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION

    def rows(self) -> ColumnarRows:
        labels_array = np.asarray(self.labels, dtype=np.int32)
        object_ids = (
            np.asarray(tuple(int(object_id) for object_id in self.object_ids), dtype=np.int32)
            if self.object_ids is not None
            else np.arange(
                1,
                (int(labels_array.max()) if labels_array.size else 0) + 1,
                dtype=np.int32,
            )
        )
        if object_ids.size <= 0:
            return ObjectIntensityZernikeMeasurementColumnarRows.empty()

        zernike_indexes, magnitudes, phases = intensity_zernike_moments(
            self.image,
            labels_array,
            object_ids,
            image_mask=self.image_mask,
            max_order=int(self.max_order),
            backend_provider=self.backend_provider,
        )
        return ObjectIntensityZernikeMeasurementColumnarRows(
            object_ids=object_ids,
            zernike_indexes=zernike_indexes,
            magnitudes=magnitudes,
            phases=phases,
            include_phase=self.include_phase,
            phase_zero_extent=int(labels_array.max(initial=0)),
            slice_index=self.slice_index,
            row_identity=self.row_identity,
        )


@dataclass(slots=True)
class ObjectIntensityZernikeMeasurementColumnarRows(ObjectMeasurementColumnarRows):
    """Columnar intensity-Zernike measurement rows."""

    object_ids: Sequence[int]
    zernike_indexes: Sequence[tuple[int, int]]
    magnitudes: object
    phases: object
    include_phase: bool
    phase_zero_extent: int | None = None
    slice_index: int | None = None
    row_identity: MeasurementObjectRowIdentity | None = None
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls) -> "ObjectIntensityZernikeMeasurementColumnarRows":
        return cls(
            object_ids=(),
            zernike_indexes=(),
            magnitudes=np.zeros((0, 0), dtype=np.float64),
            phases=np.zeros((0, 0), dtype=np.float64),
            include_phase=False,
        )

    def __post_init__(self) -> None:
        object_ids = np.asarray(self.object_ids, dtype=np.int32)
        zernike_indexes = tuple((int(n), int(m)) for n, m in self.zernike_indexes)
        if object_ids.size == 0 or len(zernike_indexes) == 0:
            self._columns = MappingProxyType({})
            return

        magnitude_values = np.asarray(self.magnitudes, dtype=np.float64)
        phase_values = np.asarray(self.phases, dtype=np.float64)
        if self.phase_zero_extent is not None:
            zero_phase_rows = object_ids <= int(self.phase_zero_extent)
            if np.any(zero_phase_rows):
                phase_values = phase_values.copy()
                phase_values[
                    zero_phase_rows[:, np.newaxis] & np.isnan(phase_values)
                ] = 0.0
        descriptor_count = 2 if self.include_phase else 1
        row_count = int(object_ids.size) * len(zernike_indexes) * descriptor_count
        feature_sequence: list[str] = []
        value_columns: list[np.ndarray] = []
        for index, (degree, repetition) in enumerate(zernike_indexes):
            feature_sequence.append(
                indexed_object_intensity_zernike_feature_name(
                    ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE,
                    degree=degree,
                    repetition=repetition,
                )
            )
            value_columns.append(magnitude_values[:, index])
            if self.include_phase:
                feature_sequence.append(
                    indexed_object_intensity_zernike_feature_name(
                        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
                        degree=degree,
                        repetition=repetition,
                    )
                )
                value_columns.append(phase_values[:, index])

        object_labels = np.tile(object_ids, len(feature_sequence)).astype(
            np.int32,
            copy=False,
        )
        feature_names = np.repeat(
            np.asarray(feature_sequence, dtype=object),
            int(object_ids.size),
        )
        result_values = np.concatenate(value_columns).astype(np.float64, copy=False)

        columns: dict[str, Sequence[object]] = {
            MeasurementRowAxisField.OBJECT_LABEL.value: object_labels,
            MeasurementRowAxisField.FEATURE_NAME.value: feature_names,
            MeasurementRowValueField.RESULT_VALUE.value: result_values,
        }
        if self.slice_index is not None:
            columns[MeasurementRowAxisField.SLICE_INDEX.value] = np.full(
                row_count,
                int(self.slice_index),
                dtype=np.int32,
            )
        if self.row_identity is not None:
            columns[MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value] = np.full(
                row_count,
                self.row_identity.value,
                dtype=object,
            )
        self._columns = MappingProxyType(columns)

    @property
    def columns(self) -> Mapping[str, Sequence[object]]:
        return self._columns


_ZERNIKE_LABEL_GEOMETRY_CACHE: OrderedDict[
    tuple[str, tuple[int, ...], bytes, str, tuple[int, ...], bytes],
    _ZernikeLabelGeometry,
] = OrderedDict()
_ZERNIKE_LABEL_GEOMETRY_CACHE_MAX_ENTRIES = 16


class ShapeZernikeBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shape Zernike moment backends keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> ShapeZernikeMoments:
        """Return Zernike indexes and dense-label moment values."""

    @abstractmethod
    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        image_mask: np.ndarray | None = None,
        max_order: int,
    ) -> IntensityZernikeMoments:
        """Return intensity-weighted Zernike magnitudes and phases."""

    def intensity_zernike_moments_batch(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[ZernikeMomentIndexes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """Return intensity Zernikes for images sharing one label geometry."""
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        zernike_numbers: ZernikeMomentIndexes | None = None
        for image in images:
            image_zernike_numbers, magnitudes, phases = self.intensity_zernike_moments(
                image,
                labels,
                measured_labels,
                image_mask=None,
                max_order=max_order,
            )
            if zernike_numbers is None:
                zernike_numbers = image_zernike_numbers
            elif zernike_numbers != image_zernike_numbers:
                raise ValueError("Batched intensity Zernike indexes changed across images.")
            rows.append((magnitudes, phases))
        return zernike_numbers or (), tuple(rows)


class LegacyFastNumpyShapeZernikeBackendStrategy(ShapeZernikeBackendStrategy):
    """Default shape-Zernike backend using shared label geometry."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = False

    @staticmethod
    def zernike_label_geometry(
        labels: np.ndarray,
        object_ids: np.ndarray,
    ) -> _ZernikeLabelGeometry:
        """Return exact cached label geometry shared by shape/intensity Zernikes."""
        total_started_at = time.perf_counter()
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        object_ids_array = np.ascontiguousarray(object_ids, dtype=np.int32)
        key_started_at = time.perf_counter()
        key = (*_array_content_key(labels_array), *_array_content_key(object_ids_array))
        runtime_profiler.log(
            "zernike_geometry_key",
            time.perf_counter() - key_started_at,
            objects=object_ids_array.size,
        )
        entry = _ZERNIKE_LABEL_GEOMETRY_CACHE.get(key)
        if entry is not None:
            _ZERNIKE_LABEL_GEOMETRY_CACHE.move_to_end(key)
            runtime_profiler.log(
                "zernike_geometry_cache_hit",
                time.perf_counter() - total_started_at,
                objects=object_ids_array.size,
            )
            return entry

        circle_started_at = time.perf_counter()
        centers, radii = minimum_enclosing_circle_from_labels(
            labels_array,
            object_ids_array,
        )
        runtime_profiler.log(
            "zernike_geometry_min_enclosing_circle",
            time.perf_counter() - circle_started_at,
            objects=object_ids_array.size,
        )
        compact_started_at = time.perf_counter()
        y_coords, x_coords = np.nonzero(labels_array > 0)
        label_to_row = np.zeros(int(labels_array.max(initial=0)) + 1, dtype=np.int32)
        valid_object_ids = object_ids_array[
            (object_ids_array > 0) & (object_ids_array < label_to_row.size)
        ]
        label_to_row[valid_object_ids] = np.arange(
            1,
            valid_object_ids.size + 1,
            dtype=np.int32,
        )
        label_values = label_to_row[labels_array[y_coords, x_coords]]
        raw_label_values = np.ascontiguousarray(
            labels_array[y_coords, x_coords],
            dtype=np.int32,
        )
        runtime_profiler.log(
            "zernike_geometry_compact_pixels",
            time.perf_counter() - compact_started_at,
            pixels=label_values.size,
        )
        geometry = _ZernikeLabelGeometry(
            centers=np.ascontiguousarray(centers, dtype=np.float64),
            radii=np.ascontiguousarray(radii, dtype=np.float64),
            y_coords=np.ascontiguousarray(y_coords, dtype=np.float64),
            x_coords=np.ascontiguousarray(x_coords, dtype=np.float64),
            label_values=np.ascontiguousarray(label_values, dtype=np.int32),
            raw_label_values=raw_label_values,
        )
        _ZERNIKE_LABEL_GEOMETRY_CACHE[key] = geometry
        _ZERNIKE_LABEL_GEOMETRY_CACHE.move_to_end(key)
        while len(_ZERNIKE_LABEL_GEOMETRY_CACHE) > _ZERNIKE_LABEL_GEOMETRY_CACHE_MAX_ENTRIES:
            _ZERNIKE_LABEL_GEOMETRY_CACHE.popitem(last=False)
        runtime_profiler.log(
            "zernike_geometry_total",
            time.perf_counter() - total_started_at,
            objects=object_ids_array.size,
            pixels=label_values.size,
        )
        return geometry

    @staticmethod
    def zernike_radial_terms(
        zernike_numbers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return radial-polynomial coefficients in Numba-friendly dense arrays."""
        numbers = np.asarray(zernike_numbers, dtype=np.int64)
        max_terms = 1
        for n_value, m_value in numbers:
            max_terms = max(max_terms, (int(n_value) - abs(int(m_value))) // 2 + 1)

        coefficients = np.zeros((numbers.shape[0], max_terms), dtype=np.float64)
        exponents = np.zeros((numbers.shape[0], max_terms), dtype=np.int64)
        term_counts = np.zeros(numbers.shape[0], dtype=np.int64)
        for zernike_index, (n_value, m_value) in enumerate(numbers):
            n = int(n_value)
            m = abs(int(m_value))
            term_count = (n - m) // 2 + 1
            term_counts[zernike_index] = term_count
            for s in range(term_count):
                coefficients[zernike_index, s] = (
                    (-1.0 if s % 2 else 1.0)
                    * float(math.factorial(n - s))
                    / (
                        float(math.factorial(s))
                        * float(math.factorial((n + m) // 2 - s))
                        * float(math.factorial((n - m) // 2 - s))
                    )
                )
                exponents[zernike_index, s] = n - 2 * s
        return (
            np.ascontiguousarray(coefficients),
            np.ascontiguousarray(exponents),
            np.ascontiguousarray(term_counts),
        )

    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> ShapeZernikeMoments:
        labels_array = np.asarray(labels, dtype=np.int32)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = np.asarray(
            centrosome.zernike.get_zernike_indexes(int(max_order) + 1),
            dtype=np.int32,
        )
        zernike_numbers = tuple(
            (int(n), int(m))
            for n, m in zernike_numbers_array
        )
        if measured_label_ids.size == 0:
            return zernike_numbers, np.zeros(
                (0, len(zernike_numbers)),
                dtype=float,
            )
        if labels_array.size == 0 or int(labels_array.max()) <= 0:
            return zernike_numbers, np.zeros(
                (measured_label_ids.size, len(zernike_numbers)),
                dtype=float,
            )
        if not zernike_numbers:
            return zernike_numbers, np.zeros((measured_label_ids.size, 0), dtype=float)
        centers, radii = _shape_zernike_minimum_enclosing_circles(
            labels_array,
            measured_label_ids,
        )
        score_started_at = time.perf_counter()
        y_coords, x_coords = np.nonzero(labels_array > 0)
        label_to_row = np.zeros(int(labels_array.max(initial=0)) + 1, dtype=np.int32)
        valid_measured_positions = np.nonzero(
            (measured_label_ids > 0) & (measured_label_ids < label_to_row.size)
        )[0]
        label_to_row[measured_label_ids[valid_measured_positions]] = (
            valid_measured_positions.astype(np.int32) + 1
        )
        label_values = label_to_row[labels_array[y_coords, x_coords]]
        valid_pixels = label_values > 0
        coefficients, exponents, term_counts = self.zernike_radial_terms(
            zernike_numbers_array
        )
        score_context = (
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(zernike_numbers_array, dtype=np.int64),
            coefficients,
            exponents,
            term_counts,
            int(measured_label_ids.size),
        )
        values = _score_zernike_moments_direct_numba(
            np.ascontiguousarray(label_values[valid_pixels], dtype=np.int32),
            np.ascontiguousarray(y_coords[valid_pixels], dtype=np.int64),
            np.ascontiguousarray(x_coords[valid_pixels], dtype=np.int64),
            score_context,
            np.ascontiguousarray(np.pi * np.square(radii), dtype=np.float64),
        )
        runtime_profiler.log(
            "zernike_shape_score",
            time.perf_counter() - score_started_at,
            objects=int(measured_label_ids.size),
            orders=len(zernike_numbers),
        )
        return zernike_numbers, np.asarray(values, dtype=np.float64)

    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        image_mask: np.ndarray | None = None,
        max_order: int,
    ) -> IntensityZernikeMoments:
        image_array = np.asarray(image, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int32)
        image_mask_array = _intensity_zernike_image_mask(image_mask, image_array)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = _zernike_indexes_array(int(max_order))
        zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
        if measured_label_ids.size == 0 or zernike_numbers_array.size == 0:
            return (
                zernike_numbers,
                np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
                np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
            )

        geometry = self.zernike_label_geometry(
            labels_array,
            measured_label_ids,
        )
        if geometry.y_coords.size == 0:
            return (
                zernike_numbers,
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
            )

        centers = geometry.centers
        radii = geometry.radii
        y_coords = geometry.y_coords.astype(np.int64, copy=False)
        x_coords = geometry.x_coords.astype(np.int64, copy=False)
        label_values = geometry.label_values
        in_bounds = (
            (y_coords >= 0)
            & (x_coords >= 0)
            & (y_coords < image_array.shape[0])
            & (x_coords < image_array.shape[1])
        )
        if image_mask_array is not None:
            in_bounds = in_bounds & image_mask_array[y_coords, x_coords]
        valid = (
            (label_values > 0)
            & (label_values <= measured_label_ids.size)
            & np.isfinite(radii[label_values - 1])
            & (radii[label_values - 1] > 0)
            & in_bounds
        )
        y_coords = np.ascontiguousarray(y_coords[valid], dtype=np.int64)
        x_coords = np.ascontiguousarray(x_coords[valid], dtype=np.int64)
        label_values = np.ascontiguousarray(label_values[valid], dtype=np.int32)
        if y_coords.size == 0:
            return (
                zernike_numbers,
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
            )

        coefficients, exponents, term_counts = self.zernike_radial_terms(
            zernike_numbers_array
        )
        score_started_at = time.perf_counter()
        score_context = (
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(zernike_numbers_array, dtype=np.int64),
            coefficients,
            exponents,
            term_counts,
            int(measured_label_ids.size),
        )
        magnitudes, phases = _score_intensity_zernike_moments_direct_numba(
            np.ascontiguousarray(image_array, dtype=np.float64),
            label_values,
            y_coords,
            x_coords,
            score_context,
        )
        runtime_profiler.log(
            "zernike_intensity_score",
            time.perf_counter() - score_started_at,
            objects=int(measured_label_ids.size),
            pixels=int(y_coords.size),
            orders=zernike_numbers_array.shape[0],
        )
        ZernikeIntensityDebugTrace.from_intensity_measurement(
            backend_provider=self.backend_provider,
            image=image_array,
            labels=labels_array,
            max_order=max_order,
            object_ids=measured_label_ids,
            zernike_numbers=zernike_numbers,
            centers=centers,
            radii=radii,
            areas=np.bincount(
                label_values,
                minlength=measured_label_ids.size + 1,
            )[1:].astype(np.float64),
            y_coords=y_coords,
            x_coords=x_coords,
            label_values=label_values,
            pixel_values=image_array[y_coords, x_coords],
            magnitudes=magnitudes,
            phases=phases,
        ).write_if_enabled()
        return zernike_numbers, magnitudes, phases

    def intensity_zernike_moments_batch(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[ZernikeMomentIndexes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        image_arrays = tuple(np.asarray(image, dtype=np.float64) for image in images)
        if not image_arrays:
            zernike_numbers_array = _zernike_indexes_array(int(max_order))
            return tuple((int(n), int(m)) for n, m in zernike_numbers_array), ()

        labels_array = np.asarray(labels, dtype=np.int32)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = _zernike_indexes_array(int(max_order))
        zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
        if measured_label_ids.size == 0 or zernike_numbers_array.size == 0:
            empty_rows = tuple(
                (
                    np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
                    np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
                )
                for _image in image_arrays
            )
            return zernike_numbers, empty_rows

        geometry = self.zernike_label_geometry(labels_array, measured_label_ids)
        if geometry.y_coords.size == 0:
            empty_rows = tuple(
                (
                    np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                    np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                )
                for _image in image_arrays
            )
            return zernike_numbers, empty_rows

        centers = geometry.centers
        radii = geometry.radii
        y_coords = geometry.y_coords.astype(np.int64, copy=False)
        x_coords = geometry.x_coords.astype(np.int64, copy=False)
        label_values = geometry.label_values
        valid = (
            (label_values > 0)
            & (label_values <= measured_label_ids.size)
            & np.isfinite(radii[label_values - 1])
            & (radii[label_values - 1] > 0)
        )
        y_coords = np.ascontiguousarray(y_coords[valid], dtype=np.int64)
        x_coords = np.ascontiguousarray(x_coords[valid], dtype=np.int64)
        label_values = np.ascontiguousarray(label_values[valid], dtype=np.int32)
        if y_coords.size == 0:
            empty_rows = tuple(
                (
                    np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                    np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                )
                for _image in image_arrays
            )
            return zernike_numbers, empty_rows

        coefficients, exponents, term_counts = self.zernike_radial_terms(
            zernike_numbers_array
        )
        score_context = (
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(zernike_numbers_array, dtype=np.int64),
            coefficients,
            exponents,
            term_counts,
            int(measured_label_ids.size),
        )
        image_stack = np.ascontiguousarray(np.stack(image_arrays), dtype=np.float64)
        score_started_at = time.perf_counter()
        magnitudes, phases = _score_intensity_zernike_moments_batch_numba(
            image_stack,
            label_values,
            y_coords,
            x_coords,
            score_context,
        )
        runtime_profiler.log(
            "zernike_intensity_batch_score",
            time.perf_counter() - score_started_at,
            images=len(image_arrays),
            objects=int(measured_label_ids.size),
            pixels=int(y_coords.size),
            orders=zernike_numbers_array.shape[0],
        )
        return zernike_numbers, tuple(
            (magnitudes[image_index], phases[image_index])
            for image_index in range(len(image_arrays))
        )


class NativeNumpyShapeZernikeBackendStrategy(LegacyFastNumpyShapeZernikeBackendStrategy):
    """CellProfiler-source Zernike backend for parity-sensitive measurements."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def intensity_zernike_moments_batch(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[ZernikeMomentIndexes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        return _reference_intensity_zernike_moments_batch_numba(
            images,
            labels,
            measured_labels,
            max_order=max_order,
        )

    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> ShapeZernikeMoments:
        labels_array = np.asarray(labels, dtype=np.int32)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = np.asarray(
            centrosome.zernike.get_zernike_indexes(int(max_order) + 1),
            dtype=np.int32,
        )
        zernike_numbers = tuple(
            (int(n), int(m)) for n, m in zernike_numbers_array
        )
        if measured_label_ids.size == 0:
            return zernike_numbers, np.zeros(
                (0, len(zernike_numbers)),
                dtype=np.float64,
            )
        if labels_array.size == 0 or int(labels_array.max(initial=0)) <= 0:
            return zernike_numbers, np.zeros(
                (measured_label_ids.size, len(zernike_numbers)),
                dtype=np.float64,
            )
        values = centrosome.zernike.zernike(
            zernike_numbers_array,
            labels_array,
            measured_label_ids,
        )
        return zernike_numbers, np.asarray(values, dtype=np.float64)

    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        image_mask: np.ndarray | None = None,
        max_order: int,
    ) -> IntensityZernikeMoments:
        return _reference_intensity_zernike_moments_numba(
            image,
            labels,
            measured_labels,
            image_mask=image_mask,
            max_order=max_order,
            backend_provider=self.backend_provider,
        )


def shape_zernike_moments(
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    max_order: int,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ShapeZernikeMoments:
    """Return shape Zernike moments through the selected backend."""
    return ShapeZernikeBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).shape_zernike_moments(
        labels,
        measured_labels,
        max_order=max_order,
    )


def _shape_zernike_moments_with_geometry(
    zernike_numbers: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Score shape Zernikes with CP-compatible geometry and CP's scorer."""
    reverse_indexes = np.empty((int(np.max(indexes)) + 1,), int)
    reverse_indexes.fill(-1)
    reverse_indexes[indexes] = np.arange(indexes.shape[0], dtype=int)
    mask = reverse_indexes[labels] != -1

    y, x = np.asarray(
        np.mgrid[
            0 : labels.shape[0] - 1 : complex(0, labels.shape[0]),
            0 : labels.shape[1] - 1 : complex(0, labels.shape[1]),
        ],
        dtype=float,
    )
    x_masked = x[mask]
    y_masked = y[mask]
    label_masked = labels[mask]
    row_indexes = reverse_indexes[label_masked]
    y_masked -= centers[row_indexes, 0]
    y_masked /= radii[row_indexes]
    x_masked -= centers[row_indexes, 1]
    x_masked /= radii[row_indexes]

    normalized_x = np.zeros_like(x)
    normalized_y = np.zeros_like(y)
    normalized_x[mask] = x_masked
    normalized_y[mask] = y_masked
    zernike_functions = centrosome.zernike.construct_zernike_polynomials(
        normalized_x,
        normalized_y,
        zernike_numbers,
        mask,
    )
    return np.asarray(
        centrosome.zernike.score_zernike(
            zernike_functions,
            radii,
            labels,
            indexes,
        ),
        dtype=np.float64,
    )


def _shape_zernike_minimum_enclosing_circles(
    labels: np.ndarray,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP 3.9-compatible shape-Zernike enclosing circles."""
    hull, point_count = centrosome.cpmorphology.convex_hull(labels, indexes)
    centers = np.zeros((indexes.shape[0], 2), dtype=np.float64)
    radii = np.zeros(indexes.shape[0], dtype=np.float64)
    if indexes.shape[0] == 0:
        return centers, radii
    point_index = np.zeros((indexes.shape[0],), dtype=int)
    point_index[1:] = np.cumsum(point_count[:-1])
    centers[point_count == 0, :] = np.nan
    if np.all(point_count == 0):
        return centers, radii

    centers[point_count == 1, :] = hull[point_index[point_count == 1], 1:]
    radii[point_count < 2] = 0.0
    centers[point_count == 2, :] = (
        hull[point_index[point_count == 2], 1:]
        + hull[point_index[point_count == 2] + 1, 1:]
    ) / 2.0
    distance = centers[point_count == 2, :] - hull[
        point_index[point_count == 2], 1:
    ]
    radii[point_count == 2] = np.sqrt(distance[:, 0] ** 2 + distance[:, 1] ** 2)

    keep_me = point_count > 2
    s0_idx = point_index.copy()
    s1_idx = s0_idx + 1
    anti_indexes = np.zeros((int(np.max(indexes)) + 1,), dtype=int)
    anti_indexes[indexes] = np.arange(indexes.shape[0], dtype=int)
    anti_indexes_per_point = anti_indexes[hull[:, 0]]
    within_label_indexes = (
        np.arange(hull.shape[0], dtype=int) - point_index[anti_indexes_per_point]
    )

    while np.any(keep_me):
        labels_to_consider = indexes[keep_me]
        anti_indexes_to_consider = np.zeros(
            (int(np.max(labels_to_consider)) + 1,),
            dtype=int,
        )
        anti_indexes_to_consider[labels_to_consider] = np.arange(
            labels_to_consider.shape[0],
            dtype=int,
        )
        keep_me_vertices = np.logical_and(
            keep_me[anti_indexes_per_point],
            within_label_indexes >= 2,
        )
        v = hull[keep_me_vertices, 1:]
        v_labels = hull[keep_me_vertices, 0]
        v_indexes = np.argwhere(keep_me_vertices).flatten().astype(np.int32)
        anti_indexes_to_consider_per_vertex = anti_indexes_to_consider[v_labels]

        s0 = hull[s0_idx[keep_me], 1:]
        s1 = hull[s1_idx[keep_me], 1:]
        s0 = s0[anti_indexes_to_consider_per_vertex]
        s1 = s1[anti_indexes_to_consider_per_vertex]
        s01 = (s0 - s1).astype(float)
        vs0 = (v - s0).astype(float)
        vs1 = (v - s1).astype(float)
        angle_vs1s0 = np.abs(
            np.arccos(
                np.sum(s01 * vs1, axis=1)
                / np.sqrt(np.sum(s01 ** 2, axis=1) * np.sum(vs1 ** 2, axis=1))
            )
        )
        angle_vs0s1 = np.abs(
            np.arccos(
                np.sum((-s01) * vs0, axis=1)
                / np.sqrt(np.sum(s01 ** 2, axis=1) * np.sum(vs0 ** 2, axis=1))
            )
        )
        angle_s0vs1 = np.pi - angle_vs1s0 - angle_vs0s1
        min_angle = _legacy_grouped_minimum(
            angle_s0vs1,
            v_labels,
            labels_to_consider,
        )
        min_position = _legacy_grouped_minimum_position(
            angle_s0vs1,
            v_labels,
            indexes,
        )
        vertex_counts = _grouped_true_count(
            keep_me_vertices,
            hull[:, 0],
            labels_to_consider,
        )

        case_1 = np.logical_or(min_angle >= np.pi / 2.0, vertex_counts == 0)
        if np.any(case_1):
            finish_me = np.zeros((indexes.shape[0],), dtype=bool)
            finish_me[anti_indexes[labels_to_consider[case_1]]] = True
            s0_finish_me = hull[s0_idx[finish_me], 1:].astype(float)
            s1_finish_me = hull[s1_idx[finish_me], 1:].astype(float)
            centers[finish_me] = (s0_finish_me + s1_finish_me) / 2.0
            radii[finish_me] = (
                np.sqrt(np.sum((s0_finish_me - s1_finish_me) ** 2, axis=1))
                / 2.0
            )
            keep_me[finish_me] = False

        case_2 = keep_me.copy()
        case_2[angle_vs1s0[min_position] > np.pi / 2.0] = False
        case_2[angle_vs0s1[min_position] > np.pi / 2.0] = False
        case_2[angle_s0vs1[min_position] > np.pi / 2.0] = False
        if np.any(case_2):
            ss0 = hull[s0_idx[case_2], 1:].astype(float)
            ss1 = hull[s1_idx[case_2], 1:].astype(float)
            vv = v[min_position[case_2]].astype(float)
            center_y, center_x, radius = _circumcircles_from_points(ss0, ss1, vv)
            centers[case_2, 0] = center_y
            centers[case_2, 1] = center_x
            radii[case_2] = radius
            keep_me[case_2] = False

        if np.any(keep_me):
            labels_to_consider = indexes[keep_me]
            indexes_to_consider = anti_indexes[labels_to_consider]
            v_obtuse_indexes = v_indexes[min_position[keep_me]]
            angle_vs0s1_to_consider = angle_vs0s1[min_position[keep_me]]
            s0_is_obtuse = angle_vs0s1_to_consider > np.pi / 2.0

            if np.any(s0_is_obtuse):
                v_obtuse_s0_indexes = v_obtuse_indexes[s0_is_obtuse]
                obtuse_s0_idx = s0_idx[indexes_to_consider[s0_is_obtuse]]
                within_label_indexes[obtuse_s0_idx] = within_label_indexes[
                    v_obtuse_s0_indexes
                ]
                s0_idx[indexes_to_consider[s0_is_obtuse]] = v_obtuse_s0_indexes
                within_label_indexes[v_obtuse_s0_indexes] = 0

            s1_is_obtuse = np.logical_not(s0_is_obtuse)
            if np.any(s1_is_obtuse):
                v_obtuse_s1_indexes = v_obtuse_indexes[s1_is_obtuse]
                obtuse_s1_idx = s1_idx[indexes_to_consider[s1_is_obtuse]]
                within_label_indexes[obtuse_s1_idx] = within_label_indexes[
                    v_obtuse_s1_indexes
                ]
                s1_idx[indexes_to_consider[s1_is_obtuse]] = v_obtuse_s1_indexes
                within_label_indexes[v_obtuse_s1_indexes] = 1
    return centers, radii


def _circumcircles_from_points(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ss0 = first.astype(float)
    ss1 = second.astype(float)
    vv = third.astype(float)
    y_axis = 0
    x_axis = 1
    determinant = 2.0 * (
        ss0[:, x_axis] * (ss1[:, y_axis] - vv[:, y_axis])
        + ss1[:, x_axis] * (vv[:, y_axis] - ss0[:, y_axis])
        + vv[:, x_axis] * (ss0[:, y_axis] - ss1[:, y_axis])
    )
    if np.any(determinant == 0.0):
        raise ValueError("Shape-Zernike circumcircle requires non-collinear hull points.")
    center_x = (
        np.sum(ss0 ** 2, axis=1) * (ss1[:, y_axis] - vv[:, y_axis])
        + np.sum(ss1 ** 2, axis=1) * (vv[:, y_axis] - ss0[:, y_axis])
        + np.sum(vv ** 2, axis=1) * (ss0[:, y_axis] - ss1[:, y_axis])
    ) / determinant
    center_y = (
        np.sum(ss0 ** 2, axis=1) * (vv[:, x_axis] - ss1[:, x_axis])
        + np.sum(ss1 ** 2, axis=1) * (ss0[:, x_axis] - vv[:, x_axis])
        + np.sum(vv ** 2, axis=1) * (ss1[:, x_axis] - ss0[:, x_axis])
    ) / determinant
    centers = np.stack((center_y, center_x), axis=1)
    radius = np.sqrt(np.sum((ss0 - centers) ** 2, axis=1))
    return center_y, center_x, radius


def _legacy_grouped_minimum(
    values: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
) -> np.ndarray:
    """Return grouped minima for the Zernike CP42 enclosing-circle kernel."""
    result = np.zeros(indexes.shape[0], dtype=values.dtype)
    for output_index, label_id in enumerate(indexes):
        label_values = values[labels == label_id]
        if label_values.size:
            result[output_index] = np.min(label_values)
    return result


def _grouped_true_count(
    values: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
) -> np.ndarray:
    """Return grouped boolean counts for the Zernike CP42 enclosing-circle kernel."""
    result = np.zeros(indexes.shape[0], dtype=float)
    for output_index, label_id in enumerate(indexes):
        result[output_index] = float(np.count_nonzero(values[labels == label_id]))
    return result


def _legacy_grouped_minimum_position(
    values: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
) -> np.ndarray:
    """Return SciPy minimum_position using NumPy 1.24 scalar argsort semantics."""
    max_label = int(np.max(labels)) if labels.size else 0
    idxs = np.asarray(indexes, dtype=int).copy()
    found = (idxs >= 0) & (idxs <= max_label)
    idxs[~found] = max_label + 1
    order = _numpy_124_scalar_argsort(values.ravel())
    sorted_labels = labels.ravel()[order]
    sorted_positions = np.arange(values.size, dtype=int)[order]
    min_position = np.zeros(max_label + 2, dtype=int)
    min_position[sorted_labels[::-1]] = sorted_positions[::-1]
    return min_position[idxs]


def _numpy_124_scalar_argsort(values: np.ndarray) -> np.ndarray:
    """Return the NumPy 1.24 scalar quicksort argsort permutation for float data."""
    order = np.arange(values.size, dtype=int)
    if order.size <= 1:
        return order

    stack: list[tuple[int, int, int]] = []
    pl = 0
    pr = int(order.size - 1)
    cdepth = (int(order.size).bit_length() - 1) * 2
    while True:
        if cdepth < 0:
            suborder = order[pl : pr + 1]
            order[pl : pr + 1] = suborder[np.argsort(values[suborder], kind="heapsort")]
            if not stack:
                break
            pl, pr, cdepth = stack.pop()
            continue

        while (pr - pl) > 15:
            pm = pl + ((pr - pl) >> 1)
            if values[order[pm]] < values[order[pl]]:
                order[pm], order[pl] = order[pl], order[pm]
            if values[order[pr]] < values[order[pm]]:
                order[pr], order[pm] = order[pm], order[pr]
            if values[order[pm]] < values[order[pl]]:
                order[pm], order[pl] = order[pl], order[pm]
            pivot = values[order[pm]]
            pi = pl
            pj = pr - 1
            order[pm], order[pj] = order[pj], order[pm]
            while True:
                pi += 1
                while values[order[pi]] < pivot:
                    pi += 1
                pj -= 1
                while pivot < values[order[pj]]:
                    pj -= 1
                if pi >= pj:
                    break
                order[pi], order[pj] = order[pj], order[pi]
            pk = pr - 1
            order[pi], order[pk] = order[pk], order[pi]
            if (pi - pl) < (pr - pi):
                stack.append((pi + 1, pr, cdepth - 1))
                pr = pi - 1
            else:
                stack.append((pl, pi - 1, cdepth - 1))
                pl = pi + 1
            cdepth -= 1

        for pi in range(pl + 1, pr + 1):
            vi = int(order[pi])
            pivot = values[vi]
            pj = pi
            pk = pi - 1
            while pj > pl and pivot < values[order[pk]]:
                order[pj] = order[pk]
                pj -= 1
                pk -= 1
            order[pj] = vi

        if not stack:
            break
        pl, pr, cdepth = stack.pop()
    return order


def intensity_zernike_moments(
    image: np.ndarray,
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    image_mask: np.ndarray | None = None,
    max_order: int,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> IntensityZernikeMoments:
    """Return intensity-weighted Zernike moments through the selected backend."""
    return ShapeZernikeBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).intensity_zernike_moments(
        image,
        labels,
        measured_labels,
        image_mask=image_mask,
        max_order=max_order,
    )


def intensity_zernike_moments_batch(
    images: Sequence[np.ndarray],
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    max_order: int,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[ZernikeMomentIndexes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Return intensity Zernikes for images sharing one label geometry."""
    return ShapeZernikeBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).intensity_zernike_moments_batch(
        images,
        labels,
        measured_labels,
        max_order=max_order,
    )


def _array_content_key(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha1(contiguous.view(np.uint8)).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def _intensity_zernike_image_mask(
    image_mask: np.ndarray | None,
    image: np.ndarray,
) -> np.ndarray | None:
    if image_mask is None:
        return None
    projected = project_image_mask_to_data_domain(image_mask, image)
    if projected is None:
        raise ValueError(
            "Intensity Zernike image mask cannot be projected into image domain; "
            f"got mask {np.shape(image_mask)!r} for image {np.shape(image)!r}."
        )
    return np.asarray(projected, dtype=bool)


def _cellprofiler_reference_enclosing_circles(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-reference enclosing circles without vectorized radius drift."""
    label_ids_array = np.asarray(label_ids, dtype=np.int32)
    centers = np.zeros((label_ids_array.size, 2), dtype=np.float64)
    radii = np.zeros(label_ids_array.size, dtype=np.float64)
    hull, point_counts = centrosome.cpmorphology.convex_hull(labels, label_ids_array)
    hull_offset = 0
    for index, label_id in enumerate(label_ids_array):
        point_count = int(point_counts[index])
        label_hull = hull[hull_offset : hull_offset + point_count]
        hull_offset += point_count
        label_centers, label_radii = centrosome.cpmorphology.minimum_enclosing_circle(
            labels,
            np.asarray((int(label_id),), dtype=np.int32),
            hull_and_point_count=(
                label_hull,
                np.asarray((point_count,), dtype=point_counts.dtype),
            ),
        )
        centers[index] = label_centers[0]
        radii[index] = label_radii[0]
    return centers, radii


@dataclass(frozen=True)
class _ReferenceIntensityZernikeInputs:
    """CP-reference geometry normalized for the Numba Zernike scorer."""

    zernike_numbers: ZernikeMomentIndexes
    zernike_numbers_array: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    y_coords: np.ndarray
    x_coords: np.ndarray
    label_values: np.ndarray
    raw_label_values: np.ndarray


def _reference_intensity_zernike_inputs(
    labels: np.ndarray,
    measured_label_ids: np.ndarray,
    *,
    max_order: int,
) -> _ReferenceIntensityZernikeInputs:
    """Return CP-reference circle geometry in measured-label row order."""
    labels_array = np.asarray(labels, dtype=np.int32)
    measured_ids = np.asarray(measured_label_ids, dtype=np.int32)
    zernike_numbers_array = _zernike_indexes_array(int(max_order))
    zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
    center_rows = np.zeros((measured_ids.size, 2), dtype=np.float64)
    radii = np.zeros(measured_ids.size, dtype=np.float64)
    y_coords, x_coords = np.nonzero(labels_array > 0)
    if (
        labels_array.size == 0
        or int(labels_array.max(initial=0)) <= 0
        or measured_ids.size == 0
        or zernike_numbers_array.size == 0
        or y_coords.size == 0
    ):
        return _ReferenceIntensityZernikeInputs(
            zernike_numbers=zernike_numbers,
            zernike_numbers_array=zernike_numbers_array,
            centers=center_rows,
            radii=radii,
            y_coords=np.zeros(0, dtype=np.int64),
            x_coords=np.zeros(0, dtype=np.int64),
            label_values=np.zeros(0, dtype=np.int32),
            raw_label_values=np.zeros(0, dtype=np.int32),
        )

    max_label = max(
        int(labels_array.max(initial=0)),
        int(measured_ids.max(initial=0)),
    )
    label_to_row = np.zeros(max_label + 1, dtype=np.int32)
    measured_in_bounds = (measured_ids > 0) & (measured_ids <= max_label)
    label_to_row[measured_ids[measured_in_bounds]] = (
        np.nonzero(measured_in_bounds)[0].astype(np.int32) + 1
    )
    present_labels = np.unique(labels_array[labels_array > 0])
    valid_label_ids = np.intersect1d(
        measured_ids[measured_in_bounds],
        present_labels,
        assume_unique=False,
    ).astype(np.int32, copy=False)
    if valid_label_ids.size:
        valid_centers, valid_radii = _shape_zernike_minimum_enclosing_circles(
            labels_array,
            valid_label_ids,
        )
        valid_rows = label_to_row[valid_label_ids] - 1
        center_rows[valid_rows] = valid_centers
        radii[valid_rows] = valid_radii

    raw_label_values = labels_array[y_coords, x_coords].astype(np.int32, copy=False)
    label_values = label_to_row[raw_label_values]
    domain_mask = label_values > 0
    return _ReferenceIntensityZernikeInputs(
        zernike_numbers=zernike_numbers,
        zernike_numbers_array=zernike_numbers_array,
        centers=np.ascontiguousarray(center_rows, dtype=np.float64),
        radii=np.ascontiguousarray(radii, dtype=np.float64),
        y_coords=np.ascontiguousarray(y_coords[domain_mask], dtype=np.int64),
        x_coords=np.ascontiguousarray(x_coords[domain_mask], dtype=np.int64),
        label_values=np.ascontiguousarray(label_values[domain_mask], dtype=np.int32),
        raw_label_values=np.ascontiguousarray(
            raw_label_values[domain_mask],
            dtype=np.int32,
        ),
    )


def _reference_intensity_zernike_moments_numba(
    image: np.ndarray,
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    max_order: int,
    backend_provider: CellProfilerBackendProvider,
) -> IntensityZernikeMoments:
    """Score native CellProfiler intensity Zernikes with a Numba accumulator."""
    image_array = np.asarray(image, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int32)
    image_mask_array = _intensity_zernike_image_mask(image_mask, image_array)
    measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
    inputs = _reference_intensity_zernike_inputs(
        labels_array,
        measured_label_ids,
        max_order=max_order,
    )
    zernike_count = len(inputs.zernike_numbers)
    if measured_label_ids.size == 0 or inputs.zernike_numbers_array.size == 0:
        return (
            inputs.zernike_numbers,
            np.zeros((measured_label_ids.size, zernike_count), dtype=float),
            np.zeros((measured_label_ids.size, zernike_count), dtype=float),
        )
    if labels_array.size == 0 or int(labels_array.max(initial=0)) <= 0:
        return (
            inputs.zernike_numbers,
            np.full((measured_label_ids.size, zernike_count), np.nan),
            np.full((measured_label_ids.size, zernike_count), np.nan),
        )
    if inputs.label_values.size == 0:
        return (
            inputs.zernike_numbers,
            np.full((measured_label_ids.size, zernike_count), np.nan),
            np.full((measured_label_ids.size, zernike_count), np.nan),
        )

    in_bounds = (
        (inputs.y_coords >= 0)
        & (inputs.x_coords >= 0)
        & (inputs.y_coords < image_array.shape[0])
        & (inputs.x_coords < image_array.shape[1])
    )
    if image_mask_array is not None:
        in_bounds = in_bounds & image_mask_array[inputs.y_coords, inputs.x_coords]
    y_coords = np.ascontiguousarray(inputs.y_coords[in_bounds], dtype=np.int64)
    x_coords = np.ascontiguousarray(inputs.x_coords[in_bounds], dtype=np.int64)
    label_values = np.ascontiguousarray(inputs.label_values[in_bounds], dtype=np.int32)
    if label_values.size == 0:
        return (
            inputs.zernike_numbers,
            np.zeros((measured_label_ids.size, zernike_count), dtype=float),
            np.zeros((measured_label_ids.size, zernike_count), dtype=float),
        )

    coefficients, exponents, term_counts = (
        LegacyFastNumpyShapeZernikeBackendStrategy.zernike_radial_terms(
            inputs.zernike_numbers_array
        )
    )
    score_context = (
        inputs.centers,
        inputs.radii,
        np.ascontiguousarray(inputs.zernike_numbers_array, dtype=np.int64),
        coefficients,
        exponents,
        term_counts,
        int(measured_label_ids.size),
    )
    score_started_at = time.perf_counter()
    magnitudes, phases = _score_intensity_zernike_moments_direct_numba(
        np.ascontiguousarray(image_array, dtype=np.float64),
        label_values,
        y_coords,
        x_coords,
        score_context,
    )
    runtime_profiler.log(
        "zernike_intensity_reference_numba_score",
        time.perf_counter() - score_started_at,
        objects=int(measured_label_ids.size),
        pixels=int(label_values.size),
        orders=inputs.zernike_numbers_array.shape[0],
    )
    ZernikeIntensityDebugTrace.from_intensity_measurement(
        backend_provider=backend_provider,
        image=image_array,
        labels=labels_array,
        max_order=max_order,
        object_ids=measured_label_ids,
        zernike_numbers=inputs.zernike_numbers,
        centers=inputs.centers,
        radii=inputs.radii,
        areas=np.bincount(
            label_values,
            minlength=measured_label_ids.size + 1,
        )[1:].astype(np.float64),
        y_coords=y_coords,
        x_coords=x_coords,
        label_values=label_values,
        pixel_values=image_array[y_coords, x_coords],
        magnitudes=magnitudes,
        phases=phases,
    ).write_if_enabled()
    return inputs.zernike_numbers, magnitudes, phases


def _reference_intensity_zernike_moments_batch_numba(
    images: Sequence[np.ndarray],
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    max_order: int,
) -> tuple[ZernikeMomentIndexes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Batch-score native CellProfiler intensity Zernikes with shared geometry."""
    image_arrays = tuple(np.asarray(image, dtype=np.float64) for image in images)
    labels_array = np.asarray(labels, dtype=np.int32)
    measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
    inputs = _reference_intensity_zernike_inputs(
        labels_array,
        measured_label_ids,
        max_order=max_order,
    )
    zernike_count = len(inputs.zernike_numbers)
    if not image_arrays:
        return inputs.zernike_numbers, ()
    if measured_label_ids.size == 0 or inputs.zernike_numbers_array.size == 0:
        empty_rows = tuple(
            (
                np.zeros((measured_label_ids.size, zernike_count), dtype=float),
                np.zeros((measured_label_ids.size, zernike_count), dtype=float),
            )
            for _image in image_arrays
        )
        return inputs.zernike_numbers, empty_rows
    if (
        labels_array.size == 0
        or int(labels_array.max(initial=0)) <= 0
        or inputs.label_values.size == 0
    ):
        empty_rows = tuple(
            (
                np.full((measured_label_ids.size, zernike_count), np.nan),
                np.full((measured_label_ids.size, zernike_count), np.nan),
            )
            for _image in image_arrays
        )
        return inputs.zernike_numbers, empty_rows

    coefficients, exponents, term_counts = (
        LegacyFastNumpyShapeZernikeBackendStrategy.zernike_radial_terms(
            inputs.zernike_numbers_array
        )
    )
    score_context = (
        inputs.centers,
        inputs.radii,
        np.ascontiguousarray(inputs.zernike_numbers_array, dtype=np.int64),
        coefficients,
        exponents,
        term_counts,
        int(measured_label_ids.size),
    )
    image_stack = np.ascontiguousarray(np.stack(image_arrays), dtype=np.float64)
    score_started_at = time.perf_counter()
    magnitudes, phases = _score_intensity_zernike_moments_batch_numba(
        image_stack,
        inputs.label_values,
        inputs.y_coords,
        inputs.x_coords,
        score_context,
    )
    runtime_profiler.log(
        "zernike_intensity_reference_numba_batch_score",
        time.perf_counter() - score_started_at,
        images=len(image_arrays),
        objects=int(measured_label_ids.size),
        pixels=int(inputs.label_values.size),
        orders=inputs.zernike_numbers_array.shape[0],
    )
    return inputs.zernike_numbers, tuple(
        (magnitudes[image_index], phases[image_index])
        for image_index in range(len(image_arrays))
    )


def _zernike_indexes_array(max_order: int) -> np.ndarray:
    indexes: list[tuple[int, int]] = []
    for n_value in range(0, int(max_order) + 1):
        for m_value in range(n_value % 2, n_value + 1, 2):
            indexes.append((n_value, m_value))
    return np.asarray(indexes, dtype=np.int64)


__all__ = public_names_from_objects(
    IntensityZernikeMeasurementRowsRequest,
    LegacyFastNumpyShapeZernikeBackendStrategy,
    NativeNumpyShapeZernikeBackendStrategy,
    ShapeZernikeBackendStrategy,
    ZernikeIntensityDebugArrayTrace,
    ZernikeIntensityDebugTrace,
    intensity_zernike_moments,
    intensity_zernike_moments_batch,
    shape_zernike_moments,
)


@njit(cache=True)
def _zernike_max_order_numba(zernike_numbers: np.ndarray) -> int:
    zernike_count = zernike_numbers.shape[0]
    max_order = 0
    for zernike_index in range(zernike_count):
        n_value = int(zernike_numbers[zernike_index, 0])
        m_value = abs(int(zernike_numbers[zernike_index, 1]))
        if n_value > max_order:
            max_order = n_value
        if m_value > max_order:
            max_order = m_value
    return max_order


@njit(cache=True)
def _prepare_zernike_pixel_basis_numba(
    normalized_y: float,
    normalized_x: float,
    max_order: int,
    rho_powers: np.ndarray,
    cos_by_m: np.ndarray,
    sin_by_m: np.ndarray,
) -> bool:
    rho_squared = normalized_x * normalized_x + normalized_y * normalized_y
    if rho_squared > 1.0:
        return False
    rho = np.sqrt(rho_squared)
    rho_powers[0] = 1.0
    for order in range(1, max_order + 1):
        rho_powers[order] = rho_powers[order - 1] * rho

    cos_by_m[0] = 1.0
    sin_by_m[0] = 0.0
    if max_order > 0:
        if rho > 0.0:
            cos_theta = normalized_y / rho
            sin_theta = normalized_x / rho
        else:
            cos_theta = 1.0
            sin_theta = 0.0
        cos_by_m[1] = cos_theta
        sin_by_m[1] = sin_theta
        for order in range(2, max_order + 1):
            cos_by_m[order] = (
                cos_by_m[order - 1] * cos_theta
                - sin_by_m[order - 1] * sin_theta
            )
            sin_by_m[order] = (
                sin_by_m[order - 1] * cos_theta
                + cos_by_m[order - 1] * sin_theta
            )
    return True


@njit(cache=True)
def _accumulate_zernike_projection_numba(
    object_index: int,
    weight: float,
    zernike_numbers: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    term_counts: np.ndarray,
    rho_powers: np.ndarray,
    cos_by_m: np.ndarray,
    sin_by_m: np.ndarray,
    real_sums: np.ndarray,
    imag_sums: np.ndarray,
) -> None:
    zernike_count = zernike_numbers.shape[0]
    for zernike_index in range(zernike_count):
        radial = 0.0
        for term_index in range(term_counts[zernike_index]):
            radial += (
                coefficients[zernike_index, term_index]
                * rho_powers[exponents[zernike_index, term_index]]
            )
        m = abs(zernike_numbers[zernike_index, 1])
        real_sums[object_index, zernike_index] += weight * radial * cos_by_m[m]
        imag_sums[object_index, zernike_index] += weight * radial * sin_by_m[m]


@njit(cache=True)
def _score_zernike_moments_direct_numba(
    label_values: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    score_context: tuple,
    denominators: np.ndarray,
) -> np.ndarray:
    (
        centers,
        radii,
        zernike_numbers,
        coefficients,
        exponents,
        term_counts,
        object_count,
    ) = score_context
    zernike_count = zernike_numbers.shape[0]
    real_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    imag_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    max_order = _zernike_max_order_numba(zernike_numbers)
    rho_powers = np.empty(max_order + 1, dtype=np.float64)
    cos_by_m = np.empty(max_order + 1, dtype=np.float64)
    sin_by_m = np.empty(max_order + 1, dtype=np.float64)
    for pixel_index in range(label_values.size):
        object_index = label_values[pixel_index] - 1
        if object_index < 0 or object_index >= object_count:
            continue
        radius = radii[object_index]
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        normalized_y = (y_coords[pixel_index] - centers[object_index, 0]) / radius
        normalized_x = (x_coords[pixel_index] - centers[object_index, 1]) / radius
        if _prepare_zernike_pixel_basis_numba(
            normalized_y,
            normalized_x,
            max_order,
            rho_powers,
            cos_by_m,
            sin_by_m,
        ):
            _accumulate_zernike_projection_numba(
                object_index,
                1.0,
                zernike_numbers,
                coefficients,
                exponents,
                term_counts,
                rho_powers,
                cos_by_m,
                sin_by_m,
                real_sums,
                imag_sums,
            )

    output = np.empty((object_count, zernike_count), dtype=np.float64)
    for object_index in range(object_count):
        denominator = denominators[object_index]
        if not np.isfinite(denominator) or denominator <= 0.0:
            for zernike_index in range(zernike_count):
                output[object_index, zernike_index] = np.nan
            continue
        for zernike_index in range(zernike_count):
            real_value = real_sums[object_index, zernike_index]
            imag_value = imag_sums[object_index, zernike_index]
            output[object_index, zernike_index] = (
                np.sqrt(real_value * real_value + imag_value * imag_value)
                / denominator
            )
    return output


@njit(cache=True)
def _score_intensity_zernike_moments_direct_numba(
    image: np.ndarray,
    label_values: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    score_context: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    (
        centers,
        radii,
        zernike_numbers,
        coefficients,
        exponents,
        term_counts,
        object_count,
    ) = score_context
    zernike_count = zernike_numbers.shape[0]
    real_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    imag_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    areas = np.zeros(object_count, dtype=np.float64)
    max_order = _zernike_max_order_numba(zernike_numbers)
    rho_powers = np.empty(max_order + 1, dtype=np.float64)
    cos_by_m = np.empty(max_order + 1, dtype=np.float64)
    sin_by_m = np.empty(max_order + 1, dtype=np.float64)
    for pixel_index in range(label_values.size):
        object_index = label_values[pixel_index] - 1
        if object_index < 0 or object_index >= object_count:
            continue
        radius = radii[object_index]
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        y = y_coords[pixel_index]
        x = x_coords[pixel_index]
        areas[object_index] += 1.0
        normalized_y = (y - centers[object_index, 0]) / radius
        normalized_x = (x - centers[object_index, 1]) / radius
        if _prepare_zernike_pixel_basis_numba(
            normalized_y,
            normalized_x,
            max_order,
            rho_powers,
            cos_by_m,
            sin_by_m,
        ):
            _accumulate_zernike_projection_numba(
                object_index,
                image[y, x],
                zernike_numbers,
                coefficients,
                exponents,
                term_counts,
                rho_powers,
                cos_by_m,
                sin_by_m,
                real_sums,
                imag_sums,
            )

    magnitudes = np.empty((object_count, zernike_count), dtype=np.float64)
    phases = np.empty((object_count, zernike_count), dtype=np.float64)
    for object_index in range(object_count):
        area = areas[object_index]
        for zernike_index in range(zernike_count):
            real_value = real_sums[object_index, zernike_index]
            imag_value = imag_sums[object_index, zernike_index]
            if area <= 0.0:
                magnitudes[object_index, zernike_index] = np.nan
                phases[object_index, zernike_index] = np.nan
            else:
                magnitudes[object_index, zernike_index] = (
                    np.sqrt(real_value * real_value + imag_value * imag_value) / area
                )
                phases[object_index, zernike_index] = np.arctan2(
                    real_value,
                    imag_value,
                )

    return magnitudes, phases


@njit(cache=True)
def _score_intensity_zernike_moments_batch_numba(
    images: np.ndarray,
    label_values: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    score_context: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    (
        centers,
        radii,
        zernike_numbers,
        coefficients,
        exponents,
        term_counts,
        object_count,
    ) = score_context
    image_count = images.shape[0]
    zernike_count = zernike_numbers.shape[0]
    real_sums = np.zeros((image_count, object_count, zernike_count), dtype=np.float64)
    imag_sums = np.zeros((image_count, object_count, zernike_count), dtype=np.float64)
    areas = np.zeros(object_count, dtype=np.float64)
    max_order = _zernike_max_order_numba(zernike_numbers)
    rho_powers = np.empty(max_order + 1, dtype=np.float64)
    cos_by_m = np.empty(max_order + 1, dtype=np.float64)
    sin_by_m = np.empty(max_order + 1, dtype=np.float64)
    for pixel_index in range(label_values.size):
        object_index = label_values[pixel_index] - 1
        if object_index < 0 or object_index >= object_count:
            continue
        radius = radii[object_index]
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        y = y_coords[pixel_index]
        x = x_coords[pixel_index]
        areas[object_index] += 1.0
        normalized_y = (y - centers[object_index, 0]) / radius
        normalized_x = (x - centers[object_index, 1]) / radius
        if not _prepare_zernike_pixel_basis_numba(
            normalized_y,
            normalized_x,
            max_order,
            rho_powers,
            cos_by_m,
            sin_by_m,
        ):
            continue
        for zernike_index in range(zernike_count):
            radial = 0.0
            for term_index in range(term_counts[zernike_index]):
                radial += (
                    coefficients[zernike_index, term_index]
                    * rho_powers[exponents[zernike_index, term_index]]
                )
            m = abs(zernike_numbers[zernike_index, 1])
            real_basis = radial * cos_by_m[m]
            imag_basis = radial * sin_by_m[m]
            for image_index in range(image_count):
                weight = images[image_index, y, x]
                real_sums[image_index, object_index, zernike_index] += (
                    weight * real_basis
                )
                imag_sums[image_index, object_index, zernike_index] += (
                    weight * imag_basis
                )

    magnitudes = np.empty((image_count, object_count, zernike_count), dtype=np.float64)
    phases = np.empty((image_count, object_count, zernike_count), dtype=np.float64)
    for image_index in range(image_count):
        for object_index in range(object_count):
            area = areas[object_index]
            for zernike_index in range(zernike_count):
                real_value = real_sums[image_index, object_index, zernike_index]
                imag_value = imag_sums[image_index, object_index, zernike_index]
                if area <= 0.0:
                    magnitudes[image_index, object_index, zernike_index] = np.nan
                    phases[image_index, object_index, zernike_index] = np.nan
                else:
                    magnitudes[image_index, object_index, zernike_index] = (
                        np.sqrt(real_value * real_value + imag_value * imag_value)
                        / area
                    )
                    phases[image_index, object_index, zernike_index] = np.arctan2(
                        real_value,
                        imag_value,
                    )
    return magnitudes, phases
