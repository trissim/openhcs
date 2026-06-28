"""Intensity-distribution backends for CellProfiler-compatible processing."""

from __future__ import annotations

from enum import Enum

from openhcs.core.registry_strategies import enum_member_with_payload
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)

from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageMeasurementInputModule,
    ModuleSettingsSourceModule,
    ObjectMeasurementInputModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
    ObjectMeasurementRowsModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CompactMeasuredObjectMeasurementRowPolicy,
    DenseEmittedObjectMeasurementRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal


class IntensityDistributionCenterChoice(Enum):
    """Nominal CP center choices for radial intensity distribution."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "IntensityDistributionCenterChoice":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    SELF = ("self", "These objects")
    CENTERS_OF_OTHER = (
        "centers_of_other",
        "Centers of other objects",
    )
    EDGES_OF_OTHER = (
        "edges_of_other",
        "Edges of other objects",
    )


class IntensityDistributionZernikeMode(Enum):
    """Nominal CP Zernike output modes for intensity distribution."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "IntensityDistributionZernikeMode":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    NONE = ("none",)
    MAGNITUDES = ("magnitudes", "Magnitudes only")
    MAGNITUDES_AND_PHASE = ("magnitudes_and_phase", "Magnitudes and phase")


def parse_intensity_distribution_zernike_mode(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionZernikeMode, value).value


def parse_intensity_distribution_center_choice(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionCenterChoice, value).value


class MeasureObjectIntensityDistributionObjectMeasurementRowPolicy(
    DenseEmittedObjectMeasurementRowsMixin,
    CompactMeasuredObjectMeasurementRowPolicy,
):
    """Intensity-distribution rows are compact but emitted over a dense domain."""


class MeasureObjectIntensityDistributionModule(
    LabelsObjectInputPolicy,
    PerObjectMeasurementExecutionModule,
    ObjectMeasurementRowsModule,
    MeasureObjectIntensityDistributionObjectMeasurementRowPolicy,
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
):
    module_name = 'MeasureObjectIntensityDistribution'
    function_name = 'measure_object_intensity_distribution'
    validated = True
    confidence = 1.0
    ignored_settings = (
        "Hidden",
        "Select objects to use as centers",
        "Calculate intensity Zernikes?",
        "Maximum Zernike moment",
        "Maximum zernike moment",
        "Object to use as center?",
        "Scale the bins?",
        "Number of bins",
        "Maximum radius",
    )
    zernike_setting = "Calculate intensity Zernikes?"
    zernike_degree_setting = SettingNameFamily(
        "Maximum Zernike moment",
        aliases=("Maximum zernike moment",),
    )
    center_choice_setting = "Object to use as center?"
    scalar_settings: ClassVar[Mapping[str, tuple[str, Callable[[str], Any]]]] = {
        "Scale the bins?": ("wants_scaled", parse_cellprofiler_bool),
        "Number of bins": ("bin_count", parse_cellprofiler_int),
        "Maximum radius": ("maximum_radius", parse_cellprofiler_int),
    }

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        bound = cls._bind_generic_settings(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        zernike_value = optional_setting_value(module, cls.zernike_setting)
        if zernike_value is not None:
            kwargs["wants_zernikes"] = parse_intensity_distribution_zernike_mode(
                zernike_value
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.zernike_setting),
                None,
            )

        zernike_degree = optional_setting_value(module, cls.zernike_degree_setting)
        if zernike_degree is not None:
            kwargs["zernike_degree"] = parse_cellprofiler_int(zernike_degree)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    setting_names(cls.zernike_degree_setting)[0]
                ),
                None,
            )

        for setting_name, (parameter_name, parse) in cls.scalar_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = parse(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        center_choice = optional_setting_value(module, cls.center_choice_setting)
        if center_choice is not None:
            kwargs["center_choice"] = parse_intensity_distribution_center_choice(
                center_choice
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.center_choice_setting),
                None,
            )

        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        return cls.measurement_artifact_contract_from_declared_settings(
            assembler,
            builder,
            module,
        )



from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import logging
import time
from types import MappingProxyType

import numpy as np
import scipy.sparse
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    measurement_image_batch_executor,
    object_label_measurement_execution,
    special_inputs,
    special_outputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    ObjectIntensityDistributionMeasurementFeature,
    indexed_object_intensity_distribution_feature_name,
    dense_object_label_declared_or_extent_id_domain,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelPlaneDomainStackRequest,
    DenseObjectLabelSliceStackRequest,
    ObjectLabelValue,
    image_payload_data,
    object_label_dense_array,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
)
CenterChoice = IntensityDistributionCenterChoice
ZernikeMode = IntensityDistributionZernikeMode
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
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)
from openhcs.processing.backends.cellprofiler.shape import (
    ShapeMeasurementBackendStrategy,
    shape_measurement_backend,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondaryPropagationBackendStrategy,
    secondary_propagation_backend,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    IntensityZernikeMeasurementRowsRequest,
    ObjectIntensityZernikeMeasurementColumnarRows,
    intensity_zernike_moments_batch,
)
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)
_RADIAL_LABEL_GEOMETRY_CACHE_LIMIT = 16
_RADIAL_LABEL_GEOMETRY_CACHE: OrderedDict[
    "RadialLabelGeometryCacheKey",
    "RadialLabelGeometry",
] = OrderedDict()


@dataclass(frozen=True)
class IntensityDistributionProfiler:
    """Bound profiler for object intensity-distribution measurement phases."""

    function_name: str

    def record(self, label: str, started_at: float, **fields: object) -> None:
        runtime_profiler.log(
            label,
            time.perf_counter() - started_at,
            function=self.function_name,
            **fields,
        )

    def record_rows(self, label: str, started_at: float, row_count: int) -> None:
        self.record(label, started_at, rows=row_count)


@dataclass(frozen=True, slots=True)
class RadialDistributionArrays:
    """Dense per-object radial intensity-distribution arrays."""

    fraction_at_distance: np.ndarray
    mean_pixel_fraction: np.ndarray
    radial_cv_by_bin: np.ndarray
    object_has_pixels: np.ndarray
    n_bins: int

    @classmethod
    def empty(cls, *, bin_count: int, wants_scaled: bool) -> "RadialDistributionArrays":
        n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
        return cls(
            fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
            mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
            radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
            object_has_pixels=np.zeros(0, dtype=bool),
            n_bins=n_bins,
        )

    @classmethod
    def from_components(
        cls,
        *,
        fraction_at_distance: np.ndarray,
        mean_pixel_fraction: np.ndarray,
        radial_cv_by_bin: np.ndarray,
        object_has_pixels: np.ndarray,
        n_bins: int,
    ) -> "RadialDistributionArrays":
        return cls(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )


@dataclass(frozen=True, slots=True)
class RadialCenterDistanceFields:
    """CellProfiler center-propagation fields for radial measurements."""

    d_from_center: np.ndarray
    center_labels: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray


@dataclass(frozen=True, slots=True)
class RadialCenterPropagationRequest:
    """Nearest-center propagation for radial intensity-distribution geometry."""

    center_labels: np.ndarray
    colors: np.ndarray
    propagation_backend: SecondaryPropagationBackendStrategy

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Return center distances and propagated center labels by color mask."""
        d_from_center = np.zeros(self.center_labels.shape, dtype=float)
        propagated_center_labels = np.zeros(self.center_labels.shape, dtype=int)
        max_color = int(np.max(self.colors)) if self.colors.size else 0
        seed_labels = np.asarray(self.center_labels, dtype=np.int32)
        for color in range(1, max_color + 1):
            mask = self.colors == color
            seed_mask = mask & (seed_labels > 0)
            if not np.any(seed_mask):
                continue
            propagation = self.propagation_backend.propagate_zero_image_result(
                seed_labels,
                mask,
                1,
            )
            propagated_labels = propagation.labels
            distances = propagation.distances
            d_from_center[mask] = distances[mask]
            propagated_center_labels[mask] = propagated_labels[mask]
        return d_from_center, propagated_center_labels


@dataclass(frozen=True, slots=True)
class RadialLabelGeometryCacheKey:
    """Content identity for radial geometry derived only from object labels."""

    dtype: str
    shape: tuple[int, ...]
    digest: bytes

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "RadialLabelGeometryCacheKey":
        label_array = np.ascontiguousarray(labels, dtype=np.int32)
        digest = hashlib.sha1(label_array.view(np.uint8)).digest()
        return cls(
            dtype=str(label_array.dtype),
            shape=tuple(int(value) for value in label_array.shape),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class RadialLabelGeometry:
    """Label-only radial geometry shared by all intensity images for a label plane."""

    d_to_edge: np.ndarray
    center_fields: RadialCenterDistanceFields


@dataclass(frozen=True, slots=True)
class IntensityDistributionPlaneInputs:
    """2D image/label plane consumed by intensity-distribution measurement."""

    image: np.ndarray
    labels: ObjectLabelValue | np.ndarray

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        label_array = object_label_dense_array(self.labels, dtype=np.int32)
        image_array = np.asarray(self.image)
        if image_array.ndim == 3:
            image_2d = image_array[0]
            labels_2d = label_array[0] if label_array.ndim == 3 else label_array
            return image_2d, labels_2d
        return image_array, label_array


@dataclass(frozen=True, slots=True)
class IntensityDistributionSliceInputs:
    """Runtime-aligned image/label slices for intensity-distribution measurement."""

    image: np.ndarray
    labels: ObjectLabelValue | np.ndarray

    @property
    def image_array(self) -> np.ndarray:
        return np.asarray(image_payload_data(self.image))

    @property
    def slice_count(self) -> int:
        image_array = self.image_array
        return int(image_array.shape[0]) if image_array.ndim == 3 else 1

    def slices(self) -> tuple["IntensityDistributionSliceInput", ...]:
        image_array = self.image_array
        plane_domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            self.labels,
            dtype=np.int32,
            allow_single_plane=image_array.ndim == 2,
            collapse_repeated=image_array.ndim == 2,
        ).stack()
        if image_array.ndim == 2 and plane_domain_stack is not None:
            return tuple(
                IntensityDistributionSliceInput(
                    image=image_array,
                    labels=np.asarray(
                        plane_domain_stack.labels[plane_index],
                        dtype=np.int32,
                    ),
                    slice_index=plane_index,
                    object_domain=plane_domain_stack.object_id_domains[plane_index],
                )
                for plane_index in range(plane_domain_stack.plane_count)
            )
        if (
            image_array.ndim == 3
            and plane_domain_stack is not None
            and image_array.shape[0] == plane_domain_stack.plane_count
        ):
            return tuple(
                IntensityDistributionSliceInput(
                    image=image_array[plane_index],
                    labels=np.asarray(
                        plane_domain_stack.labels[plane_index],
                        dtype=np.int32,
                    ),
                    slice_index=plane_index,
                    object_domain=plane_domain_stack.object_id_domains[plane_index],
                )
                for plane_index in range(plane_domain_stack.plane_count)
            )
        label_stack = DenseObjectLabelSliceStackRequest(
            self.labels,
            slice_count=self.slice_count,
            dtype=np.int32,
        ).stack()
        if label_stack is None:
            image_2d, labels_2d = IntensityDistributionPlaneInputs(
                image_array,
                self.labels,
            ).arrays()
            return (
                IntensityDistributionSliceInput(
                    image=image_2d,
                    labels=labels_2d,
                    slice_index=0,
                    object_domain=intensity_distribution_object_domain(labels_2d),
                ),
            )
        if image_array.ndim == 3:
            return tuple(
                IntensityDistributionSliceInput.from_aligned_arrays(
                    image=image_array[slice_index],
                    labels=label_stack.slice(slice_index),
                    slice_index=slice_index,
                )
                for slice_index in range(self.slice_count)
            )
        return (
            IntensityDistributionSliceInput.from_aligned_arrays(
                image=image_array,
                labels=label_stack.slice(0),
                slice_index=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class IntensityDistributionSliceInput:
    """One aligned 2D image/label slice."""

    image: np.ndarray
    labels: np.ndarray
    slice_index: int
    object_domain: tuple[int, ...]
    row_identity: MeasurementObjectRowIdentity | None = None

    @classmethod
    def from_aligned_arrays(
        cls,
        *,
        image: np.ndarray,
        labels: ObjectLabelValue | np.ndarray,
        slice_index: int,
    ) -> "IntensityDistributionSliceInput":
        return cls(
            image=image,
            labels=np.asarray(object_label_dense_array(labels, dtype=np.int32)),
            slice_index=slice_index,
            object_domain=intensity_distribution_object_domain(labels),
        )


@dataclass(frozen=True, slots=True)
class RadialDistributionMeasureRequest:
    """Complete per-plane radial-distribution measurement request."""

    image: np.ndarray
    labels: np.ndarray
    d_to_edge: np.ndarray
    d_from_center: np.ndarray
    center_labels: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray
    bin_count: int
    wants_scaled: bool
    maximum_radius: int

    def arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return validated dense arrays for radial backend execution."""
        image_array = np.ascontiguousarray(self.image, dtype=np.float64)
        labels_array = np.ascontiguousarray(self.labels, dtype=np.int32)
        d_to_edge_array = np.ascontiguousarray(self.d_to_edge, dtype=np.float64)
        d_from_center_array = np.ascontiguousarray(
            self.d_from_center,
            dtype=np.float64,
        )
        center_labels_array = np.ascontiguousarray(
            self.center_labels,
            dtype=np.int32,
        )
        centers_i_array = np.ascontiguousarray(self.centers_i, dtype=np.float64)
        centers_j_array = np.ascontiguousarray(self.centers_j, dtype=np.float64)

        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler radial intensity distribution currently supports "
                f"2-D NumPy planes, got image {image_array.shape!r} and labels "
                f"{labels_array.shape!r}."
            )
        if labels_array.shape != image_array.shape:
            raise ValueError(
                "Radial distribution labels must match the image shape; got "
                f"labels {labels_array.shape!r} for image {image_array.shape!r}."
            )
        if self.bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {self.bin_count!r}.")

        return (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        )


@dataclass(frozen=True, slots=True)
class RadialDistributionGeometryIndex:
    """Per-label radial bin/wedge index reusable across intensity images."""

    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    object_indices: np.ndarray
    bin_indices: np.ndarray
    radial_indices: np.ndarray
    number_at_distance: np.ndarray
    radial_counts: np.ndarray
    object_count: int
    bin_count: int
    n_bins: int


@dataclass(frozen=True, slots=True)
class IntensityDistributionMeasurementRequest:
    """Executable intensity-distribution request for one aligned slice."""

    slice_input: IntensityDistributionSliceInput
    bin_count: int
    wants_scaled: bool
    maximum_radius: int
    wants_zernikes: ZernikeMode
    zernike_degree: int
    radial_backend: "RadialDistributionBackendStrategy"
    zernike_backend_provider: BackendProviderInput
    profiler: IntensityDistributionProfiler

    def rows(self) -> ColumnarRows:
        labels_2d = self.slice_input.labels
        object_ids = self.slice_input.object_domain
        if not object_ids:
            return ObjectIntensityDistributionMeasurementColumnarRows.empty()

        phase_started_at = time.perf_counter()
        radial_arrays = self.radial_backend.measure_self_centered(
            self.slice_input.image,
            labels_2d,
            bin_count=self.bin_count,
            wants_scaled=self.wants_scaled,
            maximum_radius=self.maximum_radius,
        )
        self.profiler.record(
            "idist_radial_backend",
            phase_started_at,
            nobjects=len(object_ids),
            bins=radial_arrays.n_bins,
        )
        phase_started_at = time.perf_counter()
        measurements = ObjectIntensityDistributionMeasurementColumnarRows(
            radial_arrays=radial_arrays,
            object_ids=object_ids,
            bin_count=self.bin_count,
            slice_index=self.slice_input.slice_index,
            row_identity=self.slice_input.row_identity,
        )
        self.profiler.record_rows(
            "idist_radial_rows",
            phase_started_at,
            len(measurements),
        )

        if self.wants_zernikes != ZernikeMode.NONE:
            phase_started_at = time.perf_counter()
            zernike_measurements = IntensityZernikeMeasurementRowsRequest(
                image=self.slice_input.image,
                labels=labels_2d,
                max_order=self.zernike_degree,
                object_ids=object_ids,
                include_phase=(
                    self.wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE
                ),
                slice_index=self.slice_input.slice_index,
                row_identity=self.slice_input.row_identity,
                backend_provider=self.zernike_backend_provider,
            ).rows()
            measurements = ConcatenatedColumnarRows(
                (
                    measurements,
                    zernike_measurements,
                )
            )
            self.profiler.record_rows(
                "idist_zernike_rows",
                phase_started_at,
                len(measurements),
            )

        return measurements


@dataclass(slots=True)
class ObjectIntensityDistributionMeasurementColumnarRows(ObjectMeasurementColumnarRows):
    """Columnar radial intensity-distribution rows."""

    radial_arrays: RadialDistributionArrays
    object_ids: tuple[int, ...]
    bin_count: int
    slice_index: int | None = None
    row_identity: MeasurementObjectRowIdentity | None = None
    _columns: Mapping[str, np.ndarray] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls) -> "ObjectIntensityDistributionMeasurementColumnarRows":
        return cls(
            radial_arrays=RadialDistributionArrays.empty(
                bin_count=0,
                wants_scaled=True,
            ),
            object_ids=(),
            bin_count=0,
        )

    def __post_init__(self) -> None:
        object_ids = np.asarray(tuple(int(object_id) for object_id in self.object_ids))
        if object_ids.size == 0:
            self._columns = MappingProxyType({})
            return

        row_count = int(object_ids.size) * int(self.radial_arrays.n_bins) * 3
        object_labels = np.empty(row_count, dtype=np.int32)
        feature_names = np.empty(row_count, dtype=object)
        result_values = np.empty(row_count, dtype=np.float64)
        object_has_pixels_by_index = self.radial_arrays.object_has_pixels
        fraction_at_distance = self.radial_arrays.fraction_at_distance
        mean_pixel_fraction = self.radial_arrays.mean_pixel_fraction
        radial_cv_by_bin = self.radial_arrays.radial_cv_by_bin

        row_index = 0
        for bin_idx in range(self.radial_arrays.n_bins):
            bin_index = bin_idx + 1
            fraction_at_distance_feature = (
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.FRACTION_AT_DISTANCE,
                    bin_index=bin_index,
                    bin_count=self.bin_count,
                )
            )
            mean_fraction_feature = indexed_object_intensity_distribution_feature_name(
                ObjectIntensityDistributionMeasurementFeature.MEAN_FRACTION,
                bin_index=bin_index,
                bin_count=self.bin_count,
            )
            radial_cv_feature = indexed_object_intensity_distribution_feature_name(
                ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                bin_index=bin_index,
                bin_count=self.bin_count,
            )
            radial_cv = radial_cv_by_bin[bin_idx]
            for object_label in object_ids:
                object_row = DeclaredRadialDistributionObjectRow(
                    int(object_label),
                    object_has_pixels_by_index.size,
                )
                obj_idx = object_row.array_index
                object_has_pixels = (
                    obj_idx is not None and bool(object_has_pixels_by_index[obj_idx])
                )
                object_labels[row_index : row_index + 3] = int(object_label)
                feature_names[row_index] = fraction_at_distance_feature
                feature_names[row_index + 1] = mean_fraction_feature
                feature_names[row_index + 2] = radial_cv_feature
                result_values[row_index] = (
                    float(fraction_at_distance[obj_idx, bin_idx])
                    if object_has_pixels and obj_idx is not None
                    else np.nan
                )
                result_values[row_index + 1] = (
                    float(mean_pixel_fraction[obj_idx, bin_idx])
                    if object_has_pixels and obj_idx is not None
                    else np.nan
                )
                result_values[row_index + 2] = (
                    CellProfilerRadialCVExportValue(radial_cv[obj_idx]).as_float()
                    if obj_idx is not None
                    else 0.0
                )
                row_index += 3

        columns: dict[str, np.ndarray] = {
            "object_label": object_labels,
            "feature_name": feature_names,
            "result_value": result_values,
        }
        if self.slice_index is not None:
            columns["slice_index"] = np.full(
                row_count,
                int(self.slice_index),
                dtype=np.int32,
            )
        if self.row_identity is not None:
            columns[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = np.full(
                row_count,
                self.row_identity.value,
                dtype=object,
            )
        self._columns = MappingProxyType(columns)

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns


@dataclass(frozen=True, slots=True)
class DeclaredRadialDistributionObjectRow:
    """Array row for a declared object ID, including absent measured objects."""

    object_label: int
    measured_object_count: int

    @property
    def array_index(self) -> int | None:
        index = int(self.object_label) - 1
        if index < 0 or index >= int(self.measured_object_count):
            return None
        return index


@dataclass(frozen=True, slots=True)
class CellProfilerRadialCVExportValue:
    """CellProfiler export value for undefined radial coefficient of variation."""

    value: float

    def as_float(self) -> float:
        raw_value = float(self.value)
        if not np.isfinite(raw_value):
            return 0.0
        return raw_value


class RadialDistributionBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Radial-distribution operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True
    center_propagation_backend_provider = CellProfilerBackendProvider.NUMBA
    shape_geometry_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )

    def center_propagation_backend(self) -> SecondaryPropagationBackendStrategy:
        """Return the propagation backend used for center-distance geometry."""
        return secondary_propagation_backend(
            backend_provider=self.center_propagation_backend_provider,
        )

    def shape_geometry_backend(self) -> ShapeMeasurementBackendStrategy:
        """Return the shape backend used for label-only radial geometry."""
        return shape_measurement_backend(
            backend_provider=self.shape_geometry_backend_provider,
        )

    @abstractmethod
    def measure(
        self,
        request: RadialDistributionMeasureRequest,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays for a normalized request."""

    def measure_from_centers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays while computing center distances."""
        labels_array = np.asarray(labels, dtype=np.int32)
        d_to_edge_array = np.asarray(d_to_edge, dtype=np.float64)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count,
                wants_scaled=wants_scaled,
            )

        center_fields = self.center_distance_fields(
            labels_array,
            centers_i,
            centers_j,
        )

        return self.measure(
            RadialDistributionMeasureRequest(
                image=image,
                labels=labels_array,
                d_to_edge=d_to_edge_array,
                d_from_center=center_fields.d_from_center,
                center_labels=center_fields.center_labels,
                centers_i=center_fields.centers_i,
                centers_j=center_fields.centers_j,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
            )
        )

    def measure_self_centered(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays using each object's own center."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count,
                wants_scaled=wants_scaled,
            )
        geometry = self.label_geometry(labels_array)
        return self.measure_self_centered_with_geometry(
            image,
            labels_array,
            geometry,
            bin_count=bin_count,
            wants_scaled=wants_scaled,
            maximum_radius=maximum_radius,
        )

    def measure_self_centered_with_geometry(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays using precomputed label geometry."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count,
                wants_scaled=wants_scaled,
            )
        return self.measure(
            RadialDistributionMeasureRequest(
                image=image,
                labels=labels_array,
                d_to_edge=geometry.d_to_edge,
                d_from_center=geometry.center_fields.d_from_center,
                center_labels=geometry.center_fields.center_labels,
                centers_i=geometry.center_fields.centers_i,
                centers_j=geometry.center_fields.centers_j,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
            )
        )

    def measure_batch_self_centered_with_geometry(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> tuple[RadialDistributionArrays, ...]:
        """Return radial arrays for same-label images using shared geometry."""
        return tuple(
            self.measure_self_centered_with_geometry(
                image,
                labels,
                geometry,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
            )
            for image in images
        )

    def center_distance_fields(
        self,
        labels: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
    ) -> RadialCenterDistanceFields:
        """Return CP-compatible propagated center labels and distances."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        object_labels = np.arange(1, object_count + 1, dtype=np.int32)
        center_labels = np.zeros(labels_array.shape, dtype=int)
        centers_i_int = np.asarray(centers_i, dtype=int)
        centers_j_int = np.asarray(centers_j, dtype=int)
        valid_center_bounds = (
            (centers_i_int >= 0)
            & (centers_i_int < labels_array.shape[0])
            & (centers_j_int >= 0)
            & (centers_j_int < labels_array.shape[1])
        )
        sampled_center_labels = np.zeros(object_count, dtype=np.int32)
        sampled_center_labels[valid_center_bounds] = labels_array[
            centers_i_int[valid_center_bounds],
            centers_j_int[valid_center_bounds],
        ]
        valid_centers = valid_center_bounds & (sampled_center_labels == object_labels)
        if np.any(valid_centers):
            center_labels[
                centers_i_int[valid_centers],
                centers_j_int[valid_centers],
            ] = labels_array[
                centers_i_int[valid_centers],
                centers_j_int[valid_centers],
            ]
        shape_backend = self.shape_geometry_backend()
        phase_started_at = time.perf_counter()
        colors = shape_backend.color_labels(labels_array)
        runtime_profiler.log(
            "idist_center_color_labels",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )
        phase_started_at = time.perf_counter()
        d_from_center, propagated_center_labels = RadialCenterPropagationRequest(
            center_labels=center_labels,
            colors=colors,
            propagation_backend=self.center_propagation_backend(),
        ).fields()
        runtime_profiler.log(
            "idist_center_propagate",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )

        return RadialCenterDistanceFields(
            d_from_center=d_from_center,
            center_labels=propagated_center_labels,
            centers_i=np.asarray(centers_i, dtype=np.float64),
            centers_j=np.asarray(centers_j, dtype=np.float64),
        )

    def label_geometry(self, labels: np.ndarray) -> RadialLabelGeometry:
        """Return CP-compatible radial geometry derived only from object labels."""
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        cache_key = RadialLabelGeometryCacheKey.from_labels(labels_array)
        cached = _RADIAL_LABEL_GEOMETRY_CACHE.get(cache_key)
        if cached is not None:
            _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
            runtime_profiler.log(
                "idist_label_geometry_cache_hit",
                0.0,
                objects=int(labels_array.max()) if labels_array.size else 0,
            )
            return cached

        object_count = int(labels_array.max()) if labels_array.size else 0
        shape_backend = self.shape_geometry_backend()
        phase_started_at = time.perf_counter()
        d_to_edge = shape_backend.distance_to_edge(labels_array)
        runtime_profiler.log(
            "idist_distance_to_edge",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        phase_started_at = time.perf_counter()
        centers_i, centers_j = shape_backend.maximum_position_of_labels(
            d_to_edge,
            labels_array,
            np.arange(1, object_count + 1, dtype=np.int32),
        )
        runtime_profiler.log(
            "idist_maximum_position",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        geometry = RadialLabelGeometry(
            d_to_edge=d_to_edge,
            center_fields=self.center_distance_fields(
                labels_array,
                centers_i,
                centers_j,
            ),
        )
        _RADIAL_LABEL_GEOMETRY_CACHE[cache_key] = geometry
        _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
        while len(_RADIAL_LABEL_GEOMETRY_CACHE) > _RADIAL_LABEL_GEOMETRY_CACHE_LIMIT:
            _RADIAL_LABEL_GEOMETRY_CACHE.popitem(last=False)
        return geometry


class NativeNumpyRadialDistributionBackendStrategy(
    RadialDistributionBackendStrategy
):
    """CellProfiler-native NumPy radial-distribution backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def measure(
        self,
        request: RadialDistributionMeasureRequest,
    ) -> RadialDistributionArrays:
        (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        ) = request.arrays()
        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = (
            int(request.bin_count)
            if request.wants_scaled
            else int(request.bin_count) + 1
        )
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=request.bin_count,
                wants_scaled=request.wants_scaled,
            )

        good_mask = center_labels_array > 0
        normalized_distance = np.zeros(labels_array.shape, dtype=float)
        if request.wants_scaled:
            total_distance = d_from_center_array + d_to_edge_array
            normalized_distance[good_mask] = d_from_center_array[good_mask] / (
                total_distance[good_mask] + 0.001
            )
        else:
            normalized_distance[good_mask] = (
                d_from_center_array[good_mask] / request.maximum_radius
            )

        good_labels = labels_array[good_mask]
        bin_indexes = (normalized_distance * int(request.bin_count)).astype(int)
        bin_indexes[bin_indexes > int(request.bin_count)] = int(request.bin_count)
        labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

        histogram = scipy.sparse.coo_matrix(
            (image_array[good_mask], labels_and_bins),
            (object_count, int(request.bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(histogram, 1)
        fraction_at_distance = histogram / np.dstack(
            [sum_by_object] * (int(request.bin_count) + 1)
        )[0]

        ngood_pixels = int(np.sum(good_mask))
        number_at_distance = scipy.sparse.coo_matrix(
            (np.ones(ngood_pixels), labels_and_bins),
            (object_count, int(request.bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(number_at_distance, 1)
        fraction_at_bin = number_at_distance / np.dstack(
            [sum_by_object] * (int(request.bin_count) + 1)
        )[0]
        mean_pixel_fraction = fraction_at_distance / (
            fraction_at_bin + np.finfo(float).eps
        )

        object_has_pixels = sum_by_object > 0
        radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=float)
        row_index, column_index = np.mgrid[0 : labels_array.shape[0], 0 : labels_array.shape[1]]
        i_center = np.zeros(labels_array.shape, dtype=float)
        j_center = np.zeros(labels_array.shape, dtype=float)
        i_center[good_mask] = centers_i_array[center_labels_array[good_mask] - 1]
        j_center[good_mask] = centers_j_array[center_labels_array[good_mask] - 1]
        imask = row_index[good_mask] > i_center[good_mask]
        jmask = column_index[good_mask] > j_center[good_mask]
        absmask = np.abs(row_index[good_mask] - i_center[good_mask]) > np.abs(
            column_index[good_mask] - j_center[good_mask]
        )
        radial_index = (
            imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
        )

        for bin_index in range(n_bins):
            bin_mask = good_mask & (bin_indexes == bin_index)
            bin_pixels = int(np.sum(bin_mask))
            bin_labels = labels_array[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin_index]
            labels_and_radii = (bin_labels - 1, bin_radial_index)
            radial_values = scipy.sparse.coo_matrix(
                (image_array[bin_mask], labels_and_radii),
                (object_count, 8),
            ).toarray()
            pixel_count = scipy.sparse.coo_matrix(
                (np.ones(bin_pixels), labels_and_radii),
                (object_count, 8),
            ).toarray()
            mask = pixel_count == 0
            radial_means = np.ma.masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask, 1) == 0] = 0
            radial_cv_by_bin[bin_index] = np.asarray(radial_cv.filled(0), dtype=float)

        return RadialDistributionArrays.from_components(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )


class NumbaNumpyRadialDistributionBackendStrategy(
    RadialDistributionBackendStrategy
):
    """Numba-accelerated NumPy radial-distribution backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:6, 2:6] = 1
        image = np.zeros((8, 8), dtype=np.float32)
        d_to_edge = np.ones((8, 8), dtype=np.float64)
        centers_i = np.array([3.5], dtype=np.float64)
        centers_j = np.array([3.5], dtype=np.float64)
        self.measure_from_centers(
            image,
            labels,
            d_to_edge,
            centers_i,
            centers_j,
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )
        geometry = self.label_geometry(labels)
        self.measure_batch_self_centered_with_geometry(
            (image, image),
            labels,
            geometry,
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )

    def measure(
        self,
        request: RadialDistributionMeasureRequest,
    ) -> RadialDistributionArrays:
        (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        ) = request.arrays()
        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = (
            int(request.bin_count)
            if request.wants_scaled
            else int(request.bin_count) + 1
        )
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=request.bin_count,
                wants_scaled=request.wants_scaled,
            )

        (
            fraction_at_distance,
            mean_pixel_fraction,
            radial_cv_by_bin,
            object_has_pixels,
        ) = _measure_radial_distribution_numba(
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
            int(request.bin_count),
            bool(request.wants_scaled),
            int(request.maximum_radius),
            object_count,
        )
        return RadialDistributionArrays.from_components(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )

    def measure_batch_self_centered_with_geometry(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> tuple[RadialDistributionArrays, ...]:
        """Reuse radial bin/wedge geometry across same-label image planes."""
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            empty = RadialDistributionArrays.empty(
                bin_count=bin_count,
                wants_scaled=wants_scaled,
            )
            return tuple(empty for _image in images)
        index = RadialDistributionGeometryIndex(
            *_radial_distribution_geometry_index_numba(
                labels_array,
                np.ascontiguousarray(geometry.d_to_edge, dtype=np.float64),
                np.ascontiguousarray(
                    geometry.center_fields.d_from_center,
                    dtype=np.float64,
                ),
                np.ascontiguousarray(
                    geometry.center_fields.center_labels,
                    dtype=np.int32,
                ),
                np.ascontiguousarray(
                    geometry.center_fields.centers_i,
                    dtype=np.float64,
                ),
                np.ascontiguousarray(
                    geometry.center_fields.centers_j,
                    dtype=np.float64,
                ),
                int(bin_count),
                bool(wants_scaled),
                int(maximum_radius),
                object_count,
            )
        )
        outputs: list[RadialDistributionArrays] = []
        for image in images:
            (
                fraction_at_distance,
                mean_pixel_fraction,
                radial_cv_by_bin,
                object_has_pixels,
            ) = _measure_radial_distribution_from_geometry_index_numba(
                np.ascontiguousarray(image, dtype=np.float64),
                index.pixel_rows,
                index.pixel_cols,
                index.object_indices,
                index.bin_indices,
                index.radial_indices,
                index.number_at_distance,
                index.radial_counts,
                index.object_count,
                index.bin_count,
                index.n_bins,
            )
            outputs.append(
                RadialDistributionArrays.from_components(
                    fraction_at_distance=fraction_at_distance,
                    mean_pixel_fraction=mean_pixel_fraction,
                    radial_cv_by_bin=radial_cv_by_bin,
                    object_has_pixels=object_has_pixels,
                    n_bins=index.n_bins,
                )
            )
        return tuple(outputs)


def radial_distribution_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> RadialDistributionBackendStrategy:
    """Return the selected radial-distribution backend."""
    return RadialDistributionBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


__all__ = public_names_from_objects(
    NativeNumpyRadialDistributionBackendStrategy,
    NumbaNumpyRadialDistributionBackendStrategy,
    RadialDistributionArrays,
    RadialDistributionBackendStrategy,
    RadialLabelGeometry,
    RadialLabelGeometryCacheKey,
    radial_distribution_backend,
)


def _radial_distribution_geometry_index_numba(
    labels: np.ndarray,
    d_to_edge: np.ndarray,
    d_from_center: np.ndarray,
    center_labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    object_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    height, width = labels.shape
    n_bins = bin_count if wants_scaled else bin_count + 1
    valid_count = 0
    number_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    radial_counts = np.zeros((n_bins, object_count, 8), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius
            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0
            object_index = label_id - 1
            number_at_distance[object_index, bin_index] += 1.0
            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
                radial_counts[bin_index, object_index, radial_index] += 1.0
            valid_count += 1

    pixel_rows = np.empty(valid_count, dtype=np.int64)
    pixel_cols = np.empty(valid_count, dtype=np.int64)
    object_indices = np.empty(valid_count, dtype=np.int64)
    bin_indices = np.empty(valid_count, dtype=np.int64)
    radial_indices = np.empty(valid_count, dtype=np.int64)

    out_index = 0
    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius
            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0
            radial_index = -1
            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4

            pixel_rows[out_index] = y
            pixel_cols[out_index] = x
            object_indices[out_index] = label_id - 1
            bin_indices[out_index] = bin_index
            radial_indices[out_index] = radial_index
            out_index += 1

    return (
        pixel_rows,
        pixel_cols,
        object_indices,
        bin_indices,
        radial_indices,
        number_at_distance,
        radial_counts,
        object_count,
        bin_count,
        n_bins,
    )


@njit(cache=True)
def _measure_radial_distribution_from_geometry_index_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    object_indices: np.ndarray,
    bin_indices: np.ndarray,
    radial_indices: np.ndarray,
    number_at_distance: np.ndarray,
    radial_counts: np.ndarray,
    object_count: int,
    bin_count: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histogram = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    radial_values = np.zeros((n_bins, object_count, 8), dtype=np.float64)

    for pixel_index in range(pixel_rows.size):
        object_index = object_indices[pixel_index]
        bin_index = bin_indices[pixel_index]
        pixel_value = image[pixel_rows[pixel_index], pixel_cols[pixel_index]]
        histogram[object_index, bin_index] += pixel_value
        radial_index = radial_indices[pixel_index]
        if radial_index >= 0:
            radial_values[bin_index, object_index, radial_index] += pixel_value

    fraction_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    fraction_at_bin = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    object_has_pixels = np.zeros(object_count, dtype=np.bool_)
    eps = np.finfo(np.float64).eps

    for object_index in range(object_count):
        intensity_sum = 0.0
        pixel_count = 0.0
        for bin_index in range(bin_count + 1):
            intensity_sum += histogram[object_index, bin_index]
            pixel_count += number_at_distance[object_index, bin_index]
        if intensity_sum == 0.0:
            intensity_sum = 1.0
        if pixel_count > 0.0:
            object_has_pixels[object_index] = True
        else:
            pixel_count = 1.0
        for bin_index in range(bin_count + 1):
            fraction_at_distance[object_index, bin_index] = (
                histogram[object_index, bin_index] / intensity_sum
            )
            fraction_at_bin[object_index, bin_index] = (
                number_at_distance[object_index, bin_index] / pixel_count
            )

    mean_pixel_fraction = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    for object_index in range(object_count):
        for bin_index in range(bin_count + 1):
            mean_pixel_fraction[object_index, bin_index] = (
                fraction_at_distance[object_index, bin_index]
                / (fraction_at_bin[object_index, bin_index] + eps)
            )

    radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=np.float64)
    for bin_index in range(n_bins):
        for object_index in range(object_count):
            populated_wedges = 0
            wedge_sum = 0.0
            wedge_sum_sq = 0.0
            for radial_index in range(8):
                count = radial_counts[bin_index, object_index, radial_index]
                if count <= 0.0:
                    continue
                radial_mean = (
                    radial_values[bin_index, object_index, radial_index] / count
                )
                populated_wedges += 1
                wedge_sum += radial_mean
                wedge_sum_sq += radial_mean * radial_mean
            if populated_wedges == 0:
                continue
            mean = wedge_sum / populated_wedges
            variance = wedge_sum_sq / populated_wedges - mean * mean
            if variance < 0.0:
                variance = 0.0
            radial_cv_by_bin[bin_index, object_index] = np.sqrt(variance) / (
                mean + eps
            )

    return (
        fraction_at_distance,
        mean_pixel_fraction,
        radial_cv_by_bin,
        object_has_pixels,
    )


@njit(cache=True)
def _measure_radial_distribution_numba(
    image: np.ndarray,
    labels: np.ndarray,
    d_to_edge: np.ndarray,
    d_from_center: np.ndarray,
    center_labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    object_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    histogram = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    number_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    n_bins = bin_count if wants_scaled else bin_count + 1
    radial_values = np.zeros((n_bins, object_count, 8), dtype=np.float64)
    radial_counts = np.zeros((n_bins, object_count, 8), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            object_index = label_id - 1
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius

            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0

            pixel_value = image[y, x]
            histogram[object_index, bin_index] += pixel_value
            number_at_distance[object_index, bin_index] += 1.0

            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
                radial_values[bin_index, object_index, radial_index] += pixel_value
                radial_counts[bin_index, object_index, radial_index] += 1.0

    fraction_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    fraction_at_bin = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    object_has_pixels = np.zeros(object_count, dtype=np.bool_)
    eps = np.finfo(np.float64).eps

    for object_index in range(object_count):
        intensity_sum = 0.0
        pixel_count = 0.0
        for bin_index in range(bin_count + 1):
            intensity_sum += histogram[object_index, bin_index]
            pixel_count += number_at_distance[object_index, bin_index]
        if intensity_sum == 0.0:
            intensity_sum = 1.0
        if pixel_count > 0.0:
            object_has_pixels[object_index] = True
        else:
            pixel_count = 1.0
        for bin_index in range(bin_count + 1):
            fraction_at_distance[object_index, bin_index] = (
                histogram[object_index, bin_index] / intensity_sum
            )
            fraction_at_bin[object_index, bin_index] = (
                number_at_distance[object_index, bin_index] / pixel_count
            )

    mean_pixel_fraction = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    for object_index in range(object_count):
        for bin_index in range(bin_count + 1):
            mean_pixel_fraction[object_index, bin_index] = (
                fraction_at_distance[object_index, bin_index]
                / (fraction_at_bin[object_index, bin_index] + eps)
            )

    radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=np.float64)
    for bin_index in range(n_bins):
        for object_index in range(object_count):
            populated_wedges = 0
            wedge_sum = 0.0
            wedge_sum_sq = 0.0
            for radial_index in range(8):
                count = radial_counts[bin_index, object_index, radial_index]
                if count <= 0.0:
                    continue
                radial_mean = (
                    radial_values[bin_index, object_index, radial_index] / count
                )
                populated_wedges += 1
                wedge_sum += radial_mean
                wedge_sum_sq += radial_mean * radial_mean
            if populated_wedges == 0:
                continue
            mean = wedge_sum / populated_wedges
            variance = wedge_sum_sq / populated_wedges - mean * mean
            if variance < 0.0:
                variance = 0.0
            radial_cv_by_bin[bin_index, object_index] = np.sqrt(variance) / (
                mean + eps
            )

    return (
        fraction_at_distance,
        mean_pixel_fraction,
        radial_cv_by_bin,
        object_has_pixels,
    )


@numpy
@special_inputs("labels")
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
@special_outputs(
    (
        "radial_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "object_label",
                "feature_name",
                "result_value",
            ],
            analysis_type="radial_distribution",
        ),
    )
)
def measure_object_intensity_distribution(
    image: np.ndarray,
    labels: ObjectLabelValue,
    bin_count: int = 4,
    wants_scaled: bool = True,
    maximum_radius: int = 100,
    wants_zernikes: ZernikeMode = ZernikeMode.NONE,
    zernike_degree: int = 9,
    center_choice: CenterChoice = CenterChoice.SELF,
    radial_distribution_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    ),
    zernike_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    ),
) -> tuple[np.ndarray, ColumnarRows]:
    """Measure CellProfiler-compatible object intensity distribution rows."""
    total_started_at = time.perf_counter()
    profiler = IntensityDistributionProfiler(
        function_name="measure_object_intensity_distribution"
    )
    del center_choice

    wants_zernikes = coerce_cellprofiler_enum(ZernikeMode, wants_zernikes)
    radial_backend = radial_distribution_backend(
        backend_provider=radial_distribution_backend_provider,
    )
    measurement_batches: list[ColumnarRows] = []
    for slice_input in IntensityDistributionSliceInputs(image, labels).slices():
        measurement_batches.append(
            IntensityDistributionMeasurementRequest(
                slice_input=slice_input,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
                wants_zernikes=wants_zernikes,
                zernike_degree=zernike_degree,
                radial_backend=radial_backend,
                zernike_backend_provider=zernike_backend_provider,
                profiler=profiler,
            ).rows()
        )

    measurements = (
        measurement_batches[0]
        if len(measurement_batches) == 1
        else ConcatenatedColumnarRows(tuple(measurement_batches))
    )
    profiler.record_rows(
        "idist_total",
        total_started_at,
        len(measurements),
    )
    return image, measurements


def intensity_distribution_object_domain(labels: object) -> tuple[int, ...]:
    """Return the object domain for CP intensity-distribution rows."""
    return dense_object_label_declared_or_extent_id_domain(labels)


def _prepare_measure_object_intensity_distribution() -> None:
    """Compile radial-distribution and intensity-Zernike kernels before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity_distribution.__wrapped__(
        image,
        labels,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
        wants_zernikes=ZernikeMode.MAGNITUDES_AND_PHASE,
        zernike_degree=9,
    )


measure_object_intensity_distribution.__openhcs_prepare__ = (
    _prepare_measure_object_intensity_distribution
)


@dataclass(frozen=True, slots=True)
class IntensityDistributionBatchInvocation:
    """One batch invocation projected into the intensity-distribution slice domain."""

    request: RuntimeBatchInvocationRequest
    slice_input: IntensityDistributionSliceInput

    @classmethod
    def from_request(
        cls,
        request: RuntimeBatchInvocationRequest,
    ) -> "IntensityDistributionBatchInvocation | None":
        labels = request.kwargs.get("labels")
        if labels is None:
            return None
        slice_inputs = IntensityDistributionSliceInputs(
            np.asarray(image_payload_data(request.image)),
            labels,
        ).slices()
        if len(slice_inputs) != 1:
            return None
        return cls(request=request, slice_input=slice_inputs[0])


@dataclass(frozen=True, slots=True)
class IntensityDistributionBatchSettings:
    """Shared scalar settings for a serial measurement-image batch."""

    wants_zernikes: ZernikeMode
    zernike_degree: int
    zernike_backend_provider: BackendProviderInput
    radial_distribution_backend_provider: BackendProviderInput

    @classmethod
    def from_requests(
        cls,
        requests: tuple[RuntimeBatchInvocationRequest, ...],
    ) -> "IntensityDistributionBatchSettings | None":
        first = cls.from_kwargs(requests[0].kwargs)
        for request in requests[1:]:
            if cls.from_kwargs(request.kwargs) != first:
                return None
        return first

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, object],
    ) -> "IntensityDistributionBatchSettings":
        return cls(
            wants_zernikes=coerce_cellprofiler_enum(
                ZernikeMode,
                kwargs.get("wants_zernikes", ZernikeMode.NONE),
            ),
            zernike_degree=int(kwargs.get("zernike_degree", 9)),
            zernike_backend_provider=kwargs.get(
                "zernike_backend_provider",
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
            ),
            radial_distribution_backend_provider=kwargs.get(
                "radial_distribution_backend_provider",
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
            ),
        )


@dataclass(frozen=True, slots=True)
class IntensityDistributionRadialSettings:
    """Per-invocation radial settings for intensity distribution rows."""

    bin_count: int
    wants_scaled: bool
    maximum_radius: int

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, object],
    ) -> "IntensityDistributionRadialSettings":
        return cls(
            bin_count=int(kwargs.get("bin_count", 4)),
            wants_scaled=bool(kwargs.get("wants_scaled", True)),
            maximum_radius=int(kwargs.get("maximum_radius", 100)),
        )


def measure_object_intensity_distribution_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest],
        object,
    ],
) -> list[object]:
    """Serially share label-derived Zernike work across measurement images."""
    if len(requests) <= 1:
        return [execute_request(func, request) for request in requests]

    batch_invocations = tuple(
        IntensityDistributionBatchInvocation.from_request(request)
        for request in requests
    )
    if any(batch_invocation is None for batch_invocation in batch_invocations):
        return [execute_request(func, request) for request in requests]
    concrete_invocations = tuple(
        batch_invocation
        for batch_invocation in batch_invocations
        if batch_invocation is not None
    )
    slice_inputs = tuple(
        batch_invocation.slice_input for batch_invocation in concrete_invocations
    )
    first_slice = slice_inputs[0]
    first_labels = first_slice.labels
    first_object_domain = first_slice.object_domain
    if any(
        slice_input.labels.shape != first_labels.shape
        or not np.array_equal(slice_input.labels, first_labels)
        or slice_input.object_domain != first_object_domain
        for slice_input in slice_inputs[1:]
    ):
        return [execute_request(func, request) for request in requests]

    batch_settings = IntensityDistributionBatchSettings.from_requests(requests)
    if batch_settings is None:
        return [execute_request(func, request) for request in requests]

    zernike_rows_by_request: tuple[ColumnarRows | None, ...]
    if batch_settings.wants_zernikes == ZernikeMode.NONE or not first_object_domain:
        zernike_rows_by_request = tuple(None for _request in requests)
    else:
        zernike_indexes, zernike_results = intensity_zernike_moments_batch(
            tuple(slice_input.image for slice_input in slice_inputs),
            first_labels,
            np.asarray(first_object_domain, dtype=np.int32),
            max_order=batch_settings.zernike_degree,
            backend_provider=batch_settings.zernike_backend_provider,
        )
        zernike_rows_by_request = tuple(
            ObjectIntensityZernikeMeasurementColumnarRows(
                object_ids=first_object_domain,
                zernike_indexes=zernike_indexes,
                magnitudes=magnitudes,
                phases=phases,
                include_phase=(
                    batch_settings.wants_zernikes
                    == ZernikeMode.MAGNITUDES_AND_PHASE
                ),
                slice_index=slice_input.slice_index,
                row_identity=slice_input.row_identity,
            )
            for slice_input, (magnitudes, phases) in zip(
                slice_inputs,
                zernike_results,
                strict=True,
            )
        )

    radial_backend = radial_distribution_backend(
        backend_provider=batch_settings.radial_distribution_backend_provider,
    )
    radial_geometry = radial_backend.label_geometry(first_labels)
    radial_settings_by_request = tuple(
        IntensityDistributionRadialSettings.from_kwargs(request.kwargs)
        for request in requests
    )
    first_radial_settings = radial_settings_by_request[0]
    if all(
        radial_settings == first_radial_settings
        for radial_settings in radial_settings_by_request[1:]
    ):
        radial_arrays_by_request = radial_backend.measure_batch_self_centered_with_geometry(
            tuple(slice_input.image for slice_input in slice_inputs),
            first_labels,
            radial_geometry,
            bin_count=first_radial_settings.bin_count,
            wants_scaled=first_radial_settings.wants_scaled,
            maximum_radius=first_radial_settings.maximum_radius,
        )
    else:
        radial_arrays_by_request = tuple(
            radial_backend.measure_self_centered_with_geometry(
                slice_input.image,
                first_labels,
                radial_geometry,
                bin_count=radial_settings.bin_count,
                wants_scaled=radial_settings.wants_scaled,
                maximum_radius=radial_settings.maximum_radius,
            )
            for slice_input, radial_settings in zip(
                slice_inputs,
                radial_settings_by_request,
                strict=True,
            )
        )
    outputs: list[object] = []
    for request, slice_input, radial_settings, radial_arrays, zernike_rows in zip(
        requests,
        slice_inputs,
        radial_settings_by_request,
        radial_arrays_by_request,
        zernike_rows_by_request,
        strict=True,
    ):
        measurements: ColumnarRows = ObjectIntensityDistributionMeasurementColumnarRows(
            radial_arrays=radial_arrays,
            object_ids=first_object_domain,
            bin_count=radial_settings.bin_count,
            slice_index=slice_input.slice_index,
            row_identity=slice_input.row_identity,
        )
        if zernike_rows is not None:
            measurements = ConcatenatedColumnarRows((measurements, zernike_rows))
        outputs.append((request.image, measurements))
    return outputs


measurement_image_batch_executor(measure_object_intensity_distribution_batch)(
    measure_object_intensity_distribution
)
