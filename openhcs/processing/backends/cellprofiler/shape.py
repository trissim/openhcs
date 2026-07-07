"""Shape-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
)
from openhcs.core.runtime_semantics import RuntimeMeasurementFeature
from openhcs.core.runtime_equivalence import (
    AngularShapeDescriptorFeatureSemantics,
    ShapeDescriptorFeatureContext,
)
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.interop.cellprofiler.module_declarations import (
    BinderSettingsSourceModule,
    CellProfilerModule,
    CurrentObjectFeatureVectorAuthority,
    MeasuredObjectAnchorFeature,
    ModuleSettingsSourceModule,
    ObjectMeasurementInputModule,
    ObjectLocationFeature,
    ProcessingContract,
    ScopedMeasurementModule,
    ShapeDescriptorFeature,
    StructuringElementSettingsModule,
    ObjectMeasurementRowsModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    DenseColumnarObjectMeasurementRowsMixin,
    FeatureAnchoredCompactObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentPayloadMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
)
from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
)
from openhcs.processing.backends.cellprofiler.zernike import ShapeZernikeFeatureAuthority


class MeasureObjectSizeShapeObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin,
    FeatureAnchoredCompactObjectMeasurementRowPolicy,
):
    """Object shape rows are object-qualified, not image-source-qualified."""

    def retains_unmeasured_compact_row(
        self,
        row_mapping: "MeasurementRowMapping",
        *,
        object_id_field: str,
        axis_fields: "Sequence[str]",
    ) -> bool:
        """Preserve CP-emitted zero-only shape rows without measuring them."""
        del object_id_field, axis_fields
        return any(
            (
                self.measurement_value_is_present(row_mapping.get(feature.value))
                for feature in type(self).retained_unmeasured_compact_features
            )
        )

    def table_source_image_name(
        self,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        source_image_name: str | None,
    ) -> str | None:
        del measurement_images, source_image_name
        return None


class MeasureObjectSizeShapeModule(
    LabelsObjectInputPolicy,
    TableMeasurementRecordRowsMixin,
    CurrentPayloadMeasurementRecordMixin,
    PerObjectMeasurementExecutionModule,
    ObjectMeasurementInputModule,
    ObjectMeasurementRowsModule,
    MeasureObjectSizeShapeObjectMeasurementRowPolicy,
    MeasuredObjectAnchorFeature,
    ShapeZernikeFeatureAuthority,
    CurrentObjectFeatureVectorAuthority,
):
    module_name = "MeasureObjectSizeShape"
    function_name = "measure_object_size_shape"
    validated = True
    confidence = 1.0
    ignored_settings = ("Select objects to measure", "Select object sets to measure")
    measurement_category_prefixes = (("area", "shape"), ("location",))

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureObjectSizeShape."""

        AREA = (
            "Area",
            (),
            (MeasuredObjectAnchorFeature, ShapeDescriptorFeature),
        )
        PERIMETER = ("Perimeter", (), (ShapeDescriptorFeature,))
        VOLUME = ("Volume", (), (ShapeDescriptorFeature,))
        SURFACE_AREA = ("SurfaceArea", (), (ShapeDescriptorFeature,))
        ECCENTRICITY = ("Eccentricity", (), (ShapeDescriptorFeature,))
        SOLIDITY = ("Solidity", (), (ShapeDescriptorFeature,))
        CONVEX_AREA = ("ConvexArea", (), (ShapeDescriptorFeature,))
        EXTENT = ("Extent", (), (ShapeDescriptorFeature,))
        CENTER_X = (
            "Center_X",
            (),
            (MeasuredObjectAnchorFeature, ShapeDescriptorFeature, ObjectLocationFeature),
        )
        CENTER_Y = (
            "Center_Y",
            (),
            (MeasuredObjectAnchorFeature, ShapeDescriptorFeature, ObjectLocationFeature),
        )
        CENTER_Z = ("Center_Z", (), (ShapeDescriptorFeature, ObjectLocationFeature))
        BOUNDING_BOX_AREA = ("BoundingBoxArea", (), (ShapeDescriptorFeature,))
        BOUNDING_BOX_VOLUME = ("BoundingBoxVolume", (), (ShapeDescriptorFeature,))
        BOUNDING_BOX_MINIMUM_X = (
            "BoundingBoxMinimum_X",
            (),
            (ShapeDescriptorFeature,),
        )
        BOUNDING_BOX_MAXIMUM_X = (
            "BoundingBoxMaximum_X",
            (),
            (ShapeDescriptorFeature,),
        )
        BOUNDING_BOX_MINIMUM_Y = (
            "BoundingBoxMinimum_Y",
            (),
            (ShapeDescriptorFeature,),
        )
        BOUNDING_BOX_MAXIMUM_Y = (
            "BoundingBoxMaximum_Y",
            (),
            (ShapeDescriptorFeature,),
        )
        BOUNDING_BOX_MINIMUM_Z = (
            "BoundingBoxMinimum_Z",
            (),
            (ShapeDescriptorFeature,),
        )
        BOUNDING_BOX_MAXIMUM_Z = (
            "BoundingBoxMaximum_Z",
            (),
            (ShapeDescriptorFeature,),
        )
        EULER_NUMBER = ("EulerNumber", (), (ShapeDescriptorFeature,))
        FORM_FACTOR = ("FormFactor", (), (ShapeDescriptorFeature,))
        MAJOR_AXIS_LENGTH = ("MajorAxisLength", (), (ShapeDescriptorFeature,))
        MINOR_AXIS_LENGTH = ("MinorAxisLength", (), (ShapeDescriptorFeature,))
        ORIENTATION = ("Orientation", (), (ShapeDescriptorFeature,))
        COMPACTNESS = ("Compactness", (), (ShapeDescriptorFeature,))
        MAXIMUM_RADIUS = ("MaximumRadius", (), (ShapeDescriptorFeature,))
        MEDIAN_RADIUS = ("MedianRadius", (), (ShapeDescriptorFeature,))
        MEAN_RADIUS = ("MeanRadius", (), (ShapeDescriptorFeature,))
        MIN_FERET_DIAMETER = ("MinFeretDiameter", (), (ShapeDescriptorFeature,))
        MAX_FERET_DIAMETER = ("MaxFeretDiameter", (), (ShapeDescriptorFeature,))
        EQUIVALENT_DIAMETER = ("EquivalentDiameter", (), (ShapeDescriptorFeature,))
        SPATIAL_MOMENT = ("SpatialMoment", (), (ShapeDescriptorFeature,))
        CENTRAL_MOMENT = ("CentralMoment", (), (ShapeDescriptorFeature,))
        NORMALIZED_MOMENT = ("NormalizedMoment", (), (ShapeDescriptorFeature,))
        HU_MOMENT = ("HuMoment", (), (ShapeDescriptorFeature,))
        INERTIA_TENSOR = ("InertiaTensor", (), (ShapeDescriptorFeature,))
        INERTIA_TENSOR_EIGENVALUES = (
            "InertiaTensorEigenvalues",
            (),
            (ShapeDescriptorFeature,),
        )
        def indexed_name(self, *indices: int) -> str:
            if not indices:
                return self.value
            return "_".join((self.value, *(str(int(index)) for index in indices)))

    zernike_max_order = 9
    standard_2d_features = (
        MeasurementFeature.AREA,
        MeasurementFeature.PERIMETER,
        MeasurementFeature.MAJOR_AXIS_LENGTH,
        MeasurementFeature.MINOR_AXIS_LENGTH,
        MeasurementFeature.ECCENTRICITY,
        MeasurementFeature.ORIENTATION,
        MeasurementFeature.CENTER_X,
        MeasurementFeature.CENTER_Y,
        MeasurementFeature.BOUNDING_BOX_AREA,
        MeasurementFeature.BOUNDING_BOX_MINIMUM_X,
        MeasurementFeature.BOUNDING_BOX_MAXIMUM_X,
        MeasurementFeature.BOUNDING_BOX_MINIMUM_Y,
        MeasurementFeature.BOUNDING_BOX_MAXIMUM_Y,
        MeasurementFeature.FORM_FACTOR,
        MeasurementFeature.EXTENT,
        MeasurementFeature.SOLIDITY,
        MeasurementFeature.COMPACTNESS,
        MeasurementFeature.EULER_NUMBER,
        MeasurementFeature.MAXIMUM_RADIUS,
        MeasurementFeature.MEAN_RADIUS,
        MeasurementFeature.MEDIAN_RADIUS,
        MeasurementFeature.CONVEX_AREA,
        MeasurementFeature.MIN_FERET_DIAMETER,
        MeasurementFeature.MAX_FERET_DIAMETER,
        MeasurementFeature.EQUIVALENT_DIAMETER,
    )
    standard_3d_features = (
        MeasurementFeature.VOLUME,
        MeasurementFeature.SURFACE_AREA,
        MeasurementFeature.MAJOR_AXIS_LENGTH,
        MeasurementFeature.MINOR_AXIS_LENGTH,
        MeasurementFeature.CENTER_X,
        MeasurementFeature.CENTER_Y,
        MeasurementFeature.CENTER_Z,
        MeasurementFeature.BOUNDING_BOX_VOLUME,
        MeasurementFeature.BOUNDING_BOX_MINIMUM_X,
        MeasurementFeature.BOUNDING_BOX_MAXIMUM_X,
        MeasurementFeature.BOUNDING_BOX_MINIMUM_Y,
        MeasurementFeature.BOUNDING_BOX_MAXIMUM_Y,
        MeasurementFeature.BOUNDING_BOX_MINIMUM_Z,
        MeasurementFeature.BOUNDING_BOX_MAXIMUM_Z,
        MeasurementFeature.EXTENT,
        MeasurementFeature.EULER_NUMBER,
        MeasurementFeature.EQUIVALENT_DIAMETER,
    )
    advanced_2d_feature_specs = (
        (MeasurementFeature.SPATIAL_MOMENT, range(3), range(4)),
        (MeasurementFeature.CENTRAL_MOMENT, range(3), range(4)),
        (MeasurementFeature.NORMALIZED_MOMENT, range(4), range(4)),
        (MeasurementFeature.HU_MOMENT, range(7), None),
        (MeasurementFeature.INERTIA_TENSOR, range(2), range(2)),
        (MeasurementFeature.INERTIA_TENSOR_EIGENVALUES, range(2), None),
    )
    measurement_feature_part_aliases = {
        tuple(MeasurementFeature.AREA.feature_family().split("_")): (
            tuple(MeasurementFeature.VOLUME.feature_family().split("_")),
        ),
        tuple(MeasurementFeature.BOUNDING_BOX_AREA.feature_family().split("_")): (
            tuple(MeasurementFeature.BOUNDING_BOX_VOLUME.feature_family().split("_")),
        ),
        tuple(MeasurementFeature.PERIMETER.feature_family().split("_")): (
            tuple(MeasurementFeature.SURFACE_AREA.feature_family().split("_")),
        ),
    }
    measured_object_features = (
        MeasurementFeature.CENTER_X,
        MeasurementFeature.CENTER_Y,
    )
    retained_unmeasured_compact_features = (
        MeasurementFeature.MIN_FERET_DIAMETER,
        MeasurementFeature.MAX_FERET_DIAMETER,
        MeasurementFeature.MAXIMUM_RADIUS,
        MeasurementFeature.MEAN_RADIUS,
        MeasurementFeature.MEDIAN_RADIUS,
    )
    @classmethod
    def indexed_feature_names(
        cls,
        specs: tuple[tuple[MeasurementFeature, range, range | None], ...],
    ) -> tuple[str, ...]:
        fields: list[str] = []
        for feature, rows, columns in specs:
            if columns is None:
                fields.extend(feature.indexed_name(row) for row in rows)
            else:
                fields.extend(
                    feature.indexed_name(row, column)
                    for row in rows
                    for column in columns
                )
        return tuple(fields)

    @classmethod
    def measurement_field_names(
        cls,
        *,
        dimensions: int = 2,
        calculate_advanced: bool = True,
        calculate_zernikes: bool = True,
        object_id_field: str = "object_label",
        slice_index_field: str = "slice_index",
    ) -> tuple[str, ...]:
        if dimensions not in (2, 3):
            raise ValueError(
                f"Object shape measurements support 2D/3D, got {dimensions}D."
            )
        fields: list[str] = [slice_index_field, object_id_field]
        if dimensions == 2:
            fields.extend(feature.value for feature in cls.standard_2d_features)
            fields.append(cls.MeasurementFeature.CENTER_Z.value)
            if calculate_advanced:
                fields.extend(cls.indexed_feature_names(cls.advanced_2d_feature_specs))
            if calculate_zernikes:
                fields.extend(
                    cls.shape_zernike_feature_names(max_order=cls.zernike_max_order)
                )
        else:
            fields.extend(feature.value for feature in cls.standard_3d_features)
            if calculate_advanced:
                fields.append(cls.MeasurementFeature.SOLIDITY.value)
        return tuple(dict.fromkeys(fields))

    @classmethod
    def current_object_feature_vector(
        cls,
        feature_name: str,
        label_array: np.ndarray,
    ) -> CurrentObjectShapeFeatureVectorResult:
        """Derive live AreaShape vectors for current object-label planes."""
        if label_array.ndim not in (2, 3):
            return CurrentObjectShapeFeatureVectorResult.unavailable(
                CurrentObjectShapeFeatureVectorStatus.UNSUPPORTED_LABEL_DIMENSION
            )
        shape_feature = cls.current_shape_feature(feature_name)
        if shape_feature is None:
            return CurrentObjectShapeFeatureVectorResult.unavailable(
                CurrentObjectShapeFeatureVectorStatus.UNKNOWN_SHAPE_FEATURE
            )
        if label_array.ndim == 3:
            return CurrentObjectShapeFeatureVectorResult.available(
                current_object_label_measurement_vector(
                    tuple(
                        cls.current_shape_plane_values(
                            shape_feature,
                            label_array[slice_index],
                        )
                        for slice_index in range(label_array.shape[0])
                    )
                )
            )
        return CurrentObjectShapeFeatureVectorResult.available(
            current_object_label_measurement_vector(
                (cls.current_shape_plane_values(shape_feature, label_array),)
            )
        )

    @classmethod
    def current_shape_feature(
        cls,
        feature_name: str,
    ) -> "MeasureObjectSizeShapeModule.MeasurementFeature | None":
        field_candidates = frozenset(
            MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).field_candidates
        )
        for feature in cls.MeasurementFeature:
            normalized = normalize_runtime_identifier(feature.value)
            if normalized in field_candidates:
                return feature
        return None

    @classmethod
    def current_shape_plane_values(
        cls,
        shape_feature: "MeasureObjectSizeShapeModule.MeasurementFeature",
        label_plane: np.ndarray,
    ) -> np.ndarray:
        feature_arrays, measured_labels = measure_object_size_shape_feature_arrays(
            label_plane.astype(np.int32, copy=False),
            calculate_advanced=True,
            calculate_zernikes=False,
        )
        values = feature_arrays.get(shape_feature.value)
        if values is None:
            raise ValueError(
                f"Current object-label shape vector does not include "
                f"{shape_feature.value!r}."
            )
        return dense_object_label_values(
            label_plane,
            measured_labels=measured_labels,
            values=values,
        )

    @classmethod
    def measurement_all_field_names(
        cls,
        *,
        calculate_advanced: bool = True,
        calculate_zernikes: bool = True,
        object_id_field: str = "object_label",
        slice_index_field: str = "slice_index",
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *cls.measurement_field_names(
                        dimensions=2,
                        calculate_advanced=calculate_advanced,
                        calculate_zernikes=calculate_zernikes,
                        object_id_field=object_id_field,
                        slice_index_field=slice_index_field,
                    ),
                    *cls.measurement_field_names(
                        dimensions=3,
                        calculate_advanced=calculate_advanced,
                        calculate_zernikes=calculate_zernikes,
                        object_id_field=object_id_field,
                        slice_index_field=slice_index_field,
                    ),
                )
            )
        )

    zernike_backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    regionprops_backend_provider = AnalysisBackendProvider.NUMBA
    setting_bindings = (
        SettingToKeywordBinding(
            "Calculate the Zernike features?",
            "calculate_zernikes",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Calculate the advanced features?",
            "calculate_advanced",
            parse_cellprofiler_bool,
        ),
    )


class MeasureObjectSizeShapeOrientationFeatureSemantics(
    AngularShapeDescriptorFeatureSemantics
):
    """Orientation comparison semantics owned by MeasureObjectSizeShape."""

    strategy_key = "measure_object_size_shape_orientation"

    def matches(self, context: ShapeDescriptorFeatureContext) -> bool:
        return (
            context.feature.feature_name
            == MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION.feature_family()
        )


from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
import time
from types import MappingProxyType
from typing import ClassVar
import numpy as np
import scipy.ndimage
import skimage.measure
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution,
    special_outputs,
)
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    MeasurementRowAxisField,
    ObjectLabelDomainScope,
    ObjectLabelRepresentation,
    ObjectFeatureMissingValue,
    ObjectFeatureValueTable,
    dense_object_label_id_domain,
    dense_object_label_measurement_row_domain,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CurrentObjectShapeFeatureVectorResult,
    CurrentObjectShapeFeatureVectorStatus,
    current_object_label_measurement_vector,
    dense_object_label_values,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelSliceStackRequest,
    ObjectLabelDataPlaneStackContract,
    ObjectLabelPayload,
    ObjectLabelPlaneStackContract,
    ObjectLabelSet,
    ObjectLabelValue,
    SparseIJVLabelRows,
    object_label_dense_array,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.label_geometry import (
    feret_diameters_from_labels,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MorphologyBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.distance_propagation_numba import (
    _edt_1d_numba,
)
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.cellprofiler.zernike import shape_zernike_moments

logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)
ShapeFeatureArrays = tuple[dict[str, np.ndarray], np.ndarray]


class ShapeObjectFeatureValueTable(ObjectFeatureValueTable):
    """Object shape feature rows with CellProfiler-compatible missing values."""

    table_label = "shape"
    feature_array_domains = (
        ShapeZernikeFeatureAuthority.shape_zernike_feature_array_domains(
            max_order=MeasureObjectSizeShapeModule.zernike_max_order,
        )
    )
    feature_missing_values = {
        feature.value: ObjectFeatureMissingValue.ZERO
        for feature in (
            MeasureObjectSizeShapeModule.MeasurementFeature.MIN_FERET_DIAMETER,
            MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER,
            MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS,
            MeasureObjectSizeShapeModule.MeasurementFeature.MEAN_RADIUS,
            MeasureObjectSizeShapeModule.MeasurementFeature.MEDIAN_RADIUS,
        )
    }

    def complete_row(self, row: dict[str, float | int]) -> None:
        row.setdefault(MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z.value, 0.0)


class ShapeObjectMeasurementRows(ColumnarRows):
    """Dense AreaShape rows that already span their declared object domain."""

    __slots__ = ("_columns", "_rows")

    def __init__(
        self,
        columns: Mapping[str, tuple[object, ...]],
        rows: tuple[Mapping[str, object], ...],
    ) -> None:
        self._columns = columns
        self._rows = rows

    @classmethod
    def from_rows(cls, rows: list[dict[str, object]]) -> "ShapeObjectMeasurementRows":
        row_tuple = tuple((MappingProxyType(dict(row)) for row in rows))
        if not row_tuple:
            return cls(MappingProxyType({}), ())
        field_names = tuple(row_tuple[0])
        if any((tuple(row) != field_names for row in row_tuple[1:])):
            raise ValueError("AreaShape columnar rows require homogeneous fields.")
        columns = {
            field_name: tuple((row[field_name] for row in row_tuple))
            for field_name in field_names
        }
        return cls(MappingProxyType(columns), row_tuple)

    @property
    def columns(self) -> Mapping[str, tuple[object, ...]]:
        return self._columns

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self._rows[index]

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return self._rows


@dataclass(frozen=True, slots=True)
class SurfaceArea3DRegions:
    """Object labels and bounded 3-D regions for surface-area measurement."""

    label_ids: np.ndarray
    bounds_zyxzyx: np.ndarray

    def __post_init__(self) -> None:
        label_ids = np.ascontiguousarray(self.label_ids, dtype=np.int64)
        bounds = np.ascontiguousarray(self.bounds_zyxzyx, dtype=np.int64)
        if bounds.shape != (label_ids.size, 6):
            raise ValueError(
                f"3-D surface-area bounds must have shape ({label_ids.size}, 6), got {bounds.shape!r}."
            )
        object.__setattr__(self, "label_ids", label_ids)
        object.__setattr__(self, "bounds_zyxzyx", bounds)

    @classmethod
    def from_regionprops_table(
        cls, props: Mapping[str, np.ndarray], labels_shape: tuple[int, int, int]
    ) -> "SurfaceArea3DRegions":
        label_ids = np.asarray(props["label"], dtype=np.int64)
        if label_ids.size == 0:
            return cls(label_ids, np.zeros((0, 6), dtype=np.int64))
        bounds = np.stack(
            (
                props["bbox-0"],
                props["bbox-1"],
                props["bbox-2"],
                props["bbox-3"],
                props["bbox-4"],
                props["bbox-5"],
            ),
            axis=1,
        )
        return cls(label_ids, _expanded_surface_area_bounds(bounds, labels_shape))

    @classmethod
    def from_label_array(
        cls, labels: np.ndarray, label_ids: np.ndarray
    ) -> "SurfaceArea3DRegions":
        label_id_array = np.asarray(label_ids, dtype=np.int64)
        if label_id_array.size == 0:
            return cls(label_id_array, np.zeros((0, 6), dtype=np.int64))
        label_array = np.asarray(labels)
        bounds = np.zeros((label_id_array.size, 6), dtype=np.int64)
        for index, label_id in enumerate(label_id_array):
            positions = np.argwhere(label_array == int(label_id))
            if positions.size == 0:
                continue
            minimum = positions.min(axis=0)
            maximum = positions.max(axis=0) + 1
            bounds[index, :3] = minimum
            bounds[index, 3:] = maximum
        return cls(
            label_id_array, _expanded_surface_area_bounds(bounds, label_array.shape)
        )


@dataclass(frozen=True, slots=True)
class ObjectSizeShapeFeatureArrayOwner(ABC, metaclass=AutoRegisterMeta):
    """Shared AreaShape feature-array invocation policy for backend owners."""

    __registry_key__ = "owner_key"
    __skip_if_no_key__ = True
    owner_key = None
    calculate_advanced: bool
    calculate_zernikes: bool
    shape_backend_provider: BackendProviderInput
    zernike_backend_provider: BackendProviderInput
    regionprops_backend_provider: AnalysisBackendProvider
    feature_source_voxel_spacing: SourceVoxelSpacing = field(
        default_factory=SourceVoxelSpacing,
        kw_only=True,
    )

    def feature_arrays_for_labels(self, labels: np.ndarray) -> ShapeFeatureArrays:
        return measure_object_size_shape_feature_arrays(
            labels,
            calculate_advanced=self.calculate_advanced,
            calculate_zernikes=self.calculate_zernikes,
            shape_backend_provider=self.shape_backend_provider,
            zernike_backend_provider=self.zernike_backend_provider,
            regionprops_backend_provider=self.regionprops_backend_provider,
            source_voxel_spacing=self.feature_source_voxel_spacing,
        )

    def apply_label_source_coordinate_offset(
        self,
        labels: object,
        feature_values: dict[str, np.ndarray],
        *,
        local_offset_yx: tuple[int, int] = (0, 0),
    ) -> None:
        """Project AreaShape coordinate arrays through object-label source metadata."""
        if not isinstance(labels, ObjectLabelValue):
            return
        labels.apply_source_spatial_coordinate_offset(
            feature_values,
            x_fields=ShapeCoordinateFeatureFields.x_fields(),
            y_fields=ShapeCoordinateFeatureFields.y_fields(),
            local_offset_yx=local_offset_yx,
        )


@dataclass(frozen=True, slots=True)
class ObjectSizeShapeFeatureMeasurement(ObjectSizeShapeFeatureArrayOwner):
    """Backend-owned CellProfiler AreaShape feature-array measurement."""

    owner_key = "feature_measurement"
    labels: np.ndarray

    def feature_arrays(self) -> ShapeFeatureArrays:
        """Return feature arrays and measured label ids for 2-D or 3-D labels."""
        label_array = np.asarray(self.labels, dtype=np.int32)
        if label_array.ndim == 2:
            return self._feature_arrays_2d(label_array)
        if label_array.ndim == 3:
            return self._feature_arrays_3d(label_array)
        raise ValueError(f"Object labels must be 2D or 3D, got {label_array.ndim}D.")

    def _feature_arrays_2d(self, labels: np.ndarray) -> ShapeFeatureArrays:
        total_started_at = time.perf_counter()
        phase_started_at = time.perf_counter()
        shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
            backend_provider=self.shape_backend_provider
        )
        runtime_profiler.log(
            "moss_backend_resolution",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
        )
        phase_started_at = time.perf_counter()
        fast_region_props = LabelRegionPropertiesBackendStrategy.for_memory_type(
            backend_provider=self.regionprops_backend_provider
        ).measure_2d(
            labels,
            include_advanced=self.calculate_advanced,
        )
        runtime_profiler.log(
            "moss_region_properties",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(fast_region_props.label.size),
        )
        phase_started_at = time.perf_counter()
        props = fast_region_props.as_regionprops_table_subset(
            include_advanced=self.calculate_advanced
        )
        runtime_profiler.log(
            "moss_regionprops_table_subset",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            fields=len(props),
        )
        phase_started_at = time.perf_counter()
        convex_area, solidity = _convex_area_and_solidity_from_labels(
            labels, fast_region_props
        )
        runtime_profiler.log(
            "moss_convex_area_solidity",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(fast_region_props.label.size),
        )
        props["convex_area"] = convex_area
        props["solidity"] = solidity
        measured_labels = np.asarray(props["label"])
        nobjects = len(measured_labels)
        if nobjects == 0:
            return ({}, measured_labels)
        perimeter = np.asarray(props["perimeter"], dtype=float)
        area = np.asarray(props["area"], dtype=float)
        phase_started_at = time.perf_counter()
        max_radius, mean_radius, median_radius = (
            shape_backend.radius_features_from_labels(labels, measured_labels)
        )
        runtime_profiler.log(
            "moss_radius_features",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=nobjects,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            form_factor = 4.0 * np.pi * area / perimeter**2
        with np.errstate(divide="ignore", invalid="ignore"):
            compactness = 1.0 / form_factor
        phase_started_at = time.perf_counter()
        min_feret_diameter, max_feret_diameter = shape_backend.feret_diameters(
            labels, measured_labels
        )
        runtime_profiler.log(
            "moss_feret_diameters",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(measured_labels.size),
        )
        center_x = np.asarray(props["centroid-1"], dtype=float)
        center_y = np.asarray(props["centroid-0"], dtype=float)
        features = {
            _shape_feature(MeasureObjectSizeShapeModule.MeasurementFeature.AREA): area,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.PERIMETER
            ): perimeter,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MAJOR_AXIS_LENGTH
            ): props["major_axis_length"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MINOR_AXIS_LENGTH
            ): props["minor_axis_length"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.ECCENTRICITY
            ): props["eccentricity"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.ORIENTATION
            ): _cellprofiler_orientation_degrees(props),
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X
            ): center_x,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y
            ): center_y,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_AREA
            ): props["bbox_area"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_X
            ): props["bbox-1"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_X
            ): props["bbox-3"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_Y
            ): props["bbox-0"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_Y
            ): props["bbox-2"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.FORM_FACTOR
            ): form_factor,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EXTENT
            ): props["extent"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.SOLIDITY
            ): props["solidity"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.COMPACTNESS
            ): compactness,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EULER_NUMBER
            ): props["euler_number"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MAXIMUM_RADIUS
            ): max_radius,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MEAN_RADIUS
            ): mean_radius,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MEDIAN_RADIUS
            ): median_radius,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CONVEX_AREA
            ): props["convex_area"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MIN_FERET_DIAMETER
            ): min_feret_diameter,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MAX_FERET_DIAMETER
            ): max_feret_diameter,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EQUIVALENT_DIAMETER
            ): props["equivalent_diameter"],
        }
        if self.calculate_advanced:
            phase_started_at = time.perf_counter()
            features.update(_advanced_2d_features(props))
            runtime_profiler.log(
                "moss_advanced_features",
                time.perf_counter() - phase_started_at,
                function="measure_object_size_shape",
            )
        if self.calculate_zernikes:
            phase_started_at = time.perf_counter()
            features.update(
                _zernike_features(
                    labels,
                    measured_labels,
                    backend_provider=self.zernike_backend_provider,
                )
            )
            runtime_profiler.log(
                "moss_zernike_features",
                time.perf_counter() - phase_started_at,
                function="measure_object_size_shape",
                objects=nobjects,
            )
        runtime_profiler.log(
            "moss_features_2d_total",
            time.perf_counter() - total_started_at,
            function="measure_object_size_shape",
            objects=nobjects,
        )
        return (features, measured_labels)

    def _feature_arrays_3d(self, labels: np.ndarray) -> ShapeFeatureArrays:
        total_started_at = time.perf_counter()
        capture_array_fixture(
            "measure_object_size_shape_3d_input",
            labels=labels,
            calculate_advanced=np.asarray(self.calculate_advanced),
            calculate_zernikes=np.asarray(self.calculate_zernikes),
        )
        phase_started_at = time.perf_counter()
        shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
            backend_provider=self.shape_backend_provider
        )
        runtime_profiler.log(
            "moss_backend_resolution_3d",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
        )
        phase_started_at = time.perf_counter()
        props = skimage.measure.regionprops_table(
            labels, properties=_desired_region_properties(3, self.calculate_advanced)
        )
        runtime_profiler.log(
            "moss_regionprops_table_3d",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
        )
        surface_regions = SurfaceArea3DRegions.from_regionprops_table(
            props, labels.shape
        )
        measured_labels = surface_regions.label_ids
        phase_started_at = time.perf_counter()
        major_axis_length, minor_axis_length = _cellprofiler_3d_axis_lengths(props)
        runtime_profiler.log(
            "moss_axis_lengths_3d",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(measured_labels.size),
        )
        phase_started_at = time.perf_counter()
        surface_areas = shape_backend.surface_areas_3d(
            labels,
            surface_regions,
            spacing=self.feature_source_voxel_spacing.spacing_for_ndim(labels.ndim),
        )
        runtime_profiler.log(
            "moss_surface_areas_3d",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(measured_labels.size),
        )
        features = {
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.VOLUME
            ): props["area"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.SURFACE_AREA
            ): surface_areas,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MAJOR_AXIS_LENGTH
            ): major_axis_length,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.MINOR_AXIS_LENGTH
            ): minor_axis_length,
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X
            ): props["centroid-2"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y
            ): props["centroid-1"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z
            ): props["centroid-0"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_VOLUME
            ): props["bbox_area"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_X
            ): props["bbox-2"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_X
            ): props["bbox-5"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_Y
            ): props["bbox-1"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_Y
            ): props["bbox-4"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_Z
            ): props["bbox-0"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_Z
            ): props["bbox-3"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EXTENT
            ): props["extent"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EULER_NUMBER
            ): props["euler_number"],
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.EQUIVALENT_DIAMETER
            ): props["equivalent_diameter"],
        }
        if self.calculate_advanced:
            features[
                _shape_feature(MeasureObjectSizeShapeModule.MeasurementFeature.SOLIDITY)
            ] = props["solidity"]
        runtime_profiler.log(
            "moss_features_3d_total",
            time.perf_counter() - total_started_at,
            function="measure_object_size_shape",
            objects=int(measured_labels.size),
        )
        return (features, measured_labels)


def measure_object_size_shape_feature_arrays(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    zernike_backend_provider: BackendProviderInput = (
        MeasureObjectSizeShapeModule.zernike_backend_provider
    ),
    regionprops_backend_provider: AnalysisBackendProvider = (
        MeasureObjectSizeShapeModule.regionprops_backend_provider
    ),
    source_voxel_spacing: SourceVoxelSpacing = SourceVoxelSpacing(),
) -> ShapeFeatureArrays:
    """Return CellProfiler AreaShape feature arrays for dense labels."""
    return ObjectSizeShapeFeatureMeasurement(
        labels=np.asarray(labels, dtype=np.int32),
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
        regionprops_backend_provider=regionprops_backend_provider,
        feature_source_voxel_spacing=source_voxel_spacing,
    ).feature_arrays()


@dataclass(frozen=True, slots=True)
class ObjectSizeShapeMeasurementRowsRequest(
    ObjectSizeShapeFeatureArrayOwner,
):
    """Backend-owned AreaShape row request for dense and sparse label payloads."""

    owner_key = "object_size_shape_rows"
    labels: np.ndarray | ObjectLabelValue

    def rows(self) -> list[dict[str, object]]:
        plane_count = _measurement_plane_count(self.labels)
        if (
            isinstance(self.labels, (ObjectLabelPayload, ObjectLabelSet))
            and (
                isinstance(self.labels, ObjectLabelPayload)
                or self.labels.representation is ObjectLabelRepresentation.DENSE_LABELS
            )
            and (plane_count is not None)
            and (plane_count > 1)
        ):
            return DensePlaneStackObjectSizeShapeMeasurement(
                labels=self.labels,
                plane_count=plane_count,
                calculate_advanced=self.calculate_advanced,
                calculate_zernikes=self.calculate_zernikes,
                shape_backend_provider=self.shape_backend_provider,
                zernike_backend_provider=self.zernike_backend_provider,
                regionprops_backend_provider=self.regionprops_backend_provider,
                feature_source_voxel_spacing=self.feature_source_voxel_spacing,
            ).rows()
        if (
            isinstance(self.labels, ObjectLabelSet)
            and self.labels.representation is ObjectLabelRepresentation.SPARSE_IJV
        ):
            return SparseIJVObjectSizeShapeMeasurement(
                labels=self.labels,
                calculate_advanced=self.calculate_advanced,
                calculate_zernikes=self.calculate_zernikes,
                shape_backend_provider=self.shape_backend_provider,
                zernike_backend_provider=self.zernike_backend_provider,
                regionprops_backend_provider=self.regionprops_backend_provider,
                feature_source_voxel_spacing=self.feature_source_voxel_spacing,
            ).rows()
        label_array = object_label_dense_array(self.labels, dtype=np.int32)
        if not np.any(label_array > 0):
            return []
        feature_values, measured_labels = self.feature_arrays_for_labels(label_array)
        self.apply_label_source_coordinate_offset(self.labels, feature_values)
        return ShapeObjectFeatureValueTable.from_feature_arrays(
            feature_values,
            measured_labels,
            object_domain=dense_object_label_measurement_row_domain(
                self.labels, label_array
            ),
        ).rows()


def _measurement_plane_count(labels: ObjectLabelValue) -> int | None:
    """Return a per-plane measurement count for plane-scoped stacked labels."""
    if not isinstance(labels, (ObjectLabelPayload, ObjectLabelSet)):
        return None
    if labels.domain.scope is not ObjectLabelDomainScope.PLANE:
        return None
    data_count = ObjectLabelDataPlaneStackContract.plane_count(labels.labels)
    if data_count is None or data_count <= 1:
        return None
    return ObjectLabelPlaneStackContract.plane_count(labels)


@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
@special_outputs(
    (
        "measurements",
        csv_materializer(
            fields=list(MeasureObjectSizeShapeModule.measurement_all_field_names())
        ),
    )
)
def measure_object_size_shape(
    image: np.ndarray,
    labels: np.ndarray | ObjectLabelSet,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    shape_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    zernike_backend_provider: BackendProviderInput = (
        MeasureObjectSizeShapeModule.zernike_backend_provider
    ),
    regionprops_backend_provider: AnalysisBackendProvider = (
        MeasureObjectSizeShapeModule.regionprops_backend_provider
    ),
    slice_index: int | None = None,
) -> tuple[np.ndarray, ShapeObjectMeasurementRows]:
    """Measure CellProfiler AreaShape rows for labeled objects."""
    total_started_at = time.perf_counter()
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if (
        np.asarray(image).ndim == 2
        and slice_index is not None
        and (label_array.ndim == 3)
        and (label_array.shape[-2:] == np.asarray(image).shape)
    ):
        label_stack = DenseObjectLabelSliceStackRequest(
            labels, slice_count=int(label_array.shape[0]), dtype=np.int32
        ).stack()
        if label_stack is not None:
            projected_index = (
                slice_index
                if slice_index < label_stack.labels.shape[0]
                else 0 if label_stack.labels.shape[0] == 1 else None
            )
            if projected_index is not None:
                labels = label_stack.slice(projected_index)
    rows = ObjectSizeShapeMeasurementRowsRequest(
        labels=labels,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
        regionprops_backend_provider=regionprops_backend_provider,
        feature_source_voxel_spacing=(
            labels.parent_image_source_voxel_spacing
            if isinstance(labels, ObjectLabelValue)
            else SourceVoxelSpacing()
        ),
    ).rows()
    measurement_rows = ShapeObjectMeasurementRows.from_rows(rows)
    runtime_profiler.log(
        "moss_total",
        time.perf_counter() - total_started_at,
        function="measure_object_size_shape",
        objects=len(measurement_rows),
    )
    return (image, measurement_rows)


def prepare_measure_object_size_shape() -> None:
    """Compile AreaShape paths before benchmark execution."""
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_object_size_shape.__wrapped__(image, labels)
    image_3d = np.linspace(0.0, 1.0, 8 * 16 * 16, dtype=np.float32).reshape((8, 16, 16))
    labels_3d = np.zeros(image_3d.shape, dtype=np.int32)
    labels_3d[1:4, 3:9, 3:9] = 1
    labels_3d[4:7, 7:14, 7:14] = 2
    measure_object_size_shape.__wrapped__(image_3d, labels_3d)


measure_object_size_shape.__openhcs_prepare__ = prepare_measure_object_size_shape


@dataclass(frozen=True, slots=True)
class DensePlaneStackObjectSizeShapeMeasurement(
    ObjectSizeShapeFeatureArrayOwner,
):
    """Per-plane size/shape measurement for plane-scoped object domains."""

    owner_key = "dense_plane_stack"
    labels: ObjectLabelValue
    plane_count: int

    def rows(self) -> list[dict[str, object]]:
        label_stack, plane_count = self.measurement_label_stack()
        rows: list[dict[str, object]] = []
        for slice_index in range(plane_count):
            rows.extend(self.slice_rows(label_stack[slice_index], slice_index))
        return rows

    def measurement_label_stack(self) -> tuple[np.ndarray, int]:
        label_stack = object_label_dense_array(self.labels, dtype=np.int32)
        if (
            label_stack.ndim == 4
            and label_stack.shape[0] == self.plane_count
            and (label_stack.shape[1] == self.plane_count)
        ):
            diagonal = tuple(
                (label_stack[index, index] for index in range(self.plane_count))
            )
            if all((np.array_equal(diagonal[0], plane) for plane in diagonal[1:])):
                return (np.ascontiguousarray(diagonal[0][np.newaxis, ...]), 1)
            return (np.ascontiguousarray(np.stack(diagonal, axis=0)), self.plane_count)
        if label_stack.ndim == 4 and label_stack.shape[0] == self.plane_count:
            return (label_stack, self.plane_count)
        if label_stack.ndim != 3 or label_stack.shape[0] != self.plane_count:
            raise ValueError(
                "Dense plane-scoped object labels must have shape "
                "(plane, y, x) or (plane, z, y, x), got "
                f"{label_stack.shape!r} for {self.plane_count} semantic planes."
            )
        return (label_stack, self.plane_count)

    def slice_rows(
        self, labels_nd: np.ndarray, slice_index: int
    ) -> list[dict[str, object]]:
        feature_values, measured_labels = self.feature_arrays_for_labels(labels_nd)
        self.apply_label_source_coordinate_offset(self.labels, feature_values)
        slice_domain = self.labels.object_label_domain().project_planes((slice_index,))
        rows = ShapeObjectFeatureValueTable.from_feature_arrays(
            feature_values,
            measured_labels,
            object_domain=dense_object_label_id_domain(
                labels_nd,
                declared_object_count=slice_domain.declared_object_count,
                declared_object_ids=slice_domain.declared_object_ids,
            ),
        ).rows()
        for row in rows:
            row[MeasurementRowAxisField.SLICE_INDEX.value] = int(slice_index)
        return rows


@dataclass(frozen=True, slots=True)
class SparseIJVObjectSizeShapeMeasurement(
    ObjectSizeShapeFeatureArrayOwner,
):
    """AreaShape rows for sparse IJV object-label payloads."""

    owner_key = "sparse_ijv"
    labels: ObjectLabelSet

    def rows(self) -> list[dict[str, object]]:
        raw_labels = self.labels.labels
        sparse_rows = (
            raw_labels
            if isinstance(raw_labels, SparseIJVLabelRows)
            else SparseIJVLabelRows.from_yx_label(raw_labels)
        )
        if sparse_rows.as_array().size == 0:
            return []
        if sparse_rows.has_slice_index:
            return self.slice_stack_rows(sparse_rows)
        return self.plane_rows(
            np.asarray(sparse_rows.as_yx_label_array(), dtype=np.int32)
        )

    def slice_stack_rows(
        self, sparse_rows: SparseIJVLabelRows
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        slice_indices = sparse_rows.slice_indices()
        slice_count = max(slice_indices) + 1 if slice_indices else 0
        for slice_index in slice_indices:
            slice_ijv = np.asarray(
                sparse_rows.slice(slice_index).as_array(), dtype=np.int32
            )
            slice_domain = self.labels.domain.project_slice(
                int(slice_index), slice_count
            )
            for row in self.plane_rows(slice_ijv, domain=slice_domain):
                row[MeasurementRowAxisField.SLICE_INDEX.value] = int(slice_index)
                rows.append(row)
        return rows

    def plane_rows(
        self,
        ijv: np.ndarray,
        *,
        domain: ObjectLabelDomain | None = None,
    ) -> list[dict[str, object]]:
        object_ids = self.object_ids(ijv, domain=domain)
        rows: list[dict[str, object]] = []
        for object_id in object_ids:
            rows.append(self.object_row(ijv, int(object_id)))
        return rows

    def object_ids(
        self,
        ijv: np.ndarray,
        *,
        domain: ObjectLabelDomain | None = None,
    ) -> np.ndarray:
        domain = self.labels.domain if domain is None else domain
        if domain.declared_object_ids:
            return np.asarray(domain.declared_object_ids, dtype=np.int32)
        if domain.declared_object_count is not None:
            return np.arange(1, domain.declared_object_count + 1, dtype=np.int32)
        return np.unique(ijv[:, 2]).astype(np.int32, copy=False)

    def object_row(self, ijv: np.ndarray, object_id: int) -> dict[str, object]:
        object_pixels = ijv[ijv[:, 2] == object_id]
        if object_pixels.size == 0:
            return self.empty_row(object_id)
        pixel_y = object_pixels[:, 0]
        pixel_x = object_pixels[:, 1]
        min_y = int(pixel_y.min())
        min_x = int(pixel_x.min())
        max_y = int(pixel_y.max()) + 1
        max_x = int(pixel_x.max()) + 1
        local = np.zeros((max_y - min_y, max_x - min_x), dtype=np.int32)
        local[pixel_y - min_y, pixel_x - min_x] = object_id
        feature_values, measured_labels = self.feature_arrays_for_labels(local)
        if len(measured_labels) == 0:
            return self.empty_row(object_id)
        self.apply_label_source_coordinate_offset(
            self.labels, feature_values, local_offset_yx=(min_y, min_x)
        )
        return ShapeObjectFeatureValueTable.from_feature_arrays(
            feature_values,
            np.asarray([object_id], dtype=np.int32),
            object_domain=(object_id,),
        ).rows()[0]

    def empty_row(self, object_id: int) -> dict[str, object]:
        axis_fields = {
            MeasurementRowAxisField.SLICE_INDEX.value,
            MeasurementRowAxisField.OBJECT_LABEL.value,
        }
        return ShapeObjectFeatureValueTable.from_feature_arrays(
            {
                field: np.asarray([], dtype=float)
                for field in MeasureObjectSizeShapeModule.measurement_field_names()
                if field not in axis_fields
            },
            (),
            object_domain=(object_id,),
        ).rows()[0]


@dataclass(frozen=True, slots=True)
class ShapeCoordinateFeatureFields:
    """AreaShape feature fields whose values live in object-label XY coordinates."""

    @staticmethod
    def x_fields() -> tuple[str, ...]:
        return (
            _shape_feature(MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X),
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_X
            ),
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_X
            ),
        )

    @staticmethod
    def y_fields() -> tuple[str, ...]:
        return (
            _shape_feature(MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y),
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MINIMUM_Y
            ),
            _shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_MAXIMUM_Y
            ),
        )


class ShapeMeasurementBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Shape-measurement operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def form_factor_values(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> np.ndarray:
        """Return CP-compatible AreaShape_FormFactor values."""

    @abstractmethod
    def radius_features(
        self, object_images: np.ndarray, object_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii."""

    @abstractmethod
    def radius_features_from_labels(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii from dense labels."""

    @abstractmethod
    def feret_diameters(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum Feret diameters."""

    @abstractmethod
    def minimum_enclosing_circle(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return object center coordinates and radii."""

    def surface_areas_3d(
        self,
        labels: np.ndarray,
        regions: SurfaceArea3DRegions,
        *,
        spacing: tuple[float, ...] | None = None,
    ) -> np.ndarray:
        """Return Lewiner marching-cubes surface areas for 3-D dense labels."""
        return _surface_areas_3d_from_regions(labels, regions, spacing=spacing)

    @abstractmethod
    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        """Return per-pixel distance-to-edge for labeled objects."""

    @abstractmethod
    def maximum_position_of_labels(
        self, image: np.ndarray, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return maximum-value positions for each label."""

    @abstractmethod
    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return non-touching label color classes."""

    @abstractmethod
    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate labels through a mask and return labels plus distances."""

    @abstractmethod
    def zernike_indexes(self, max_order: int) -> np.ndarray:
        """Return Zernike index pairs up to ``max_order``."""

    @abstractmethod
    def construct_zernike_polynomials(
        self, x: np.ndarray, y: np.ndarray, zernike_indexes: np.ndarray
    ) -> np.ndarray:
        """Return Zernike polynomial values at normalized coordinates."""


class NumbaShapeMeasurementMixin(ABC):
    """Shared Numba-backed shape leaves reused by concrete backend policies."""

    def prepare_numba_shape_leaves(self) -> None:
        labels = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]], dtype=np.int32)
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        label_ids = np.array([1, 2], dtype=np.int32)
        object_images = np.stack((labels == 1, labels == 2), axis=0)
        self.form_factor_values(labels, label_ids)
        self.radius_features(object_images, 2)
        self.radius_features_from_labels(labels, label_ids)
        self.feret_diameters(labels, label_ids)
        self.distance_to_edge(labels)
        self.maximum_position_of_labels(image, labels, label_ids)
        self.color_labels(labels)

    def form_factor_values(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> np.ndarray:
        return _form_factor_values_from_labels(
            np.asarray(labels, dtype=np.int32), np.asarray(label_ids, dtype=np.int32)
        )

    def radius_features(
        self, object_images: np.ndarray, object_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_radius = np.zeros(object_count, dtype=np.float64)
        mean_radius = np.zeros(object_count, dtype=np.float64)
        median_radius = np.zeros(object_count, dtype=np.float64)
        for index, object_image in enumerate(object_images):
            max_value, mean_value, median_value = _object_radius_features_numba(
                np.asarray(object_image, dtype=np.bool_)
            )
            max_radius[index] = max_value
            mean_radius[index] = mean_value
            median_radius[index] = median_value
        return (max_radius, mean_radius, median_radius)

    def radius_features_from_labels(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _radius_features_from_labels_numba(
            np.asarray(labels, dtype=np.int32), np.asarray(label_ids, dtype=np.int32)
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            return _distance_to_edge_planewise(self, label_array)
        return _distance_to_label_edge_numba(np.ascontiguousarray(label_array))

    def maximum_position_of_labels(
        self, image: np.ndarray, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _maximum_position_of_labels_scipy_select(
            np.asarray(image),
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        return _color_labels_numpy(np.asarray(labels, dtype=np.int32))

    def feret_diameters(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return feret_diameters_from_labels(labels, label_ids)

    def zernike_indexes(self, max_order: int) -> np.ndarray:
        return _zernike_indexes_numpy(int(max_order))


class LegacyFastNumpyShapeMeasurementBackendStrategy(
    NumbaShapeMeasurementMixin, ShapeMeasurementBackendStrategy
):
    """Default NumPy shape backend with native leaves and explicit gaps."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.LEGACY_FAST
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = True

    def prepare_backend(self) -> None:
        self.prepare_numba_shape_leaves()

    def minimum_enclosing_circle(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Default shape measurements do not provide minimum enclosing circles. Use the dedicated Zernike backend."
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Default shape measurements do not own propagation. Use SecondaryPropagationBackendStrategy."
        )

    def construct_zernike_polynomials(
        self, x: np.ndarray, y: np.ndarray, zernike_indexes: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError(
            "Default shape measurements do not own Zernike polynomial construction. Use the Zernike backend family."
        )


class NumbaNumpyShapeMeasurementBackendStrategy(
    NumbaShapeMeasurementMixin, ShapeMeasurementBackendStrategy
):
    """Pure Numba shape backend. Unsupported leaves fail explicitly."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        self.prepare_numba_shape_leaves()
        label_ids = np.array([1, 2], dtype=np.int32)
        labels_3d = np.zeros((3, 3, 3), dtype=np.int32)
        labels_3d[0:2, 0:2, 0:2] = 1
        labels_3d[1:3, 1:3, 1:3] = 2
        regions_3d = SurfaceArea3DRegions.from_label_array(labels_3d, label_ids)
        self.surface_areas_3d(labels_3d, regions_3d)
        self.zernike_indexes(2)

    def minimum_enclosing_circle(
        self, labels: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba minimum enclosing circle is not implemented yet. Use the Zernike backend family."
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba propagation is not implemented yet. Use SecondaryPropagationBackendStrategy."
        )

    def construct_zernike_polynomials(
        self, x: np.ndarray, y: np.ndarray, zernike_indexes: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba Zernike polynomial construction is not implemented in the shape backend. Use the zernike backend family instead."
        )


def _distance_to_edge_planewise(
    backend: ShapeMeasurementBackendStrategy, labels: np.ndarray
) -> np.ndarray:
    if labels.ndim < 2:
        raise ValueError("Distance-to-edge requires at least two dimensions.")
    distances = np.empty(labels.shape, dtype=np.float64)
    plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
    source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
    target_planes = distances.reshape((plane_count, *labels.shape[-2:]))
    for plane_index in range(plane_count):
        target_planes[plane_index] = backend.distance_to_edge(
            source_planes[plane_index]
        )
    return distances


def shape_measurement_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ShapeMeasurementBackendStrategy:
    """Return the selected shape-measurement backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


def form_factor_values(
    labels: np.ndarray,
    label_ids: np.ndarray,
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Return CP-compatible AreaShape_FormFactor values through a backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    ).form_factor_values(labels, label_ids)


def _convex_area_and_solidity_from_labels(
    labels: np.ndarray, region_props: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact skimage-compatible convex area and solidity per label."""
    morphology_backend = MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)
    object_count = int(region_props.label.size)
    convex_area = np.zeros(object_count, dtype=float)
    solidity = np.ones(object_count, dtype=float)
    for index, label_id in enumerate(region_props.label):
        min_y = int(region_props.bbox_min_y[index])
        min_x = int(region_props.bbox_min_x[index])
        max_y = int(region_props.bbox_max_y[index])
        max_x = int(region_props.bbox_max_x[index])
        crop = labels[min_y:max_y, min_x:max_x] == int(label_id)
        hull = morphology_backend.convex_hull_image(crop)
        hull_area = float(np.count_nonzero(hull))
        convex_area[index] = hull_area
        solidity[index] = (
            float(region_props.area[index]) / hull_area if hull_area > 0.0 else np.nan
        )
    return (convex_area, solidity)


def _desired_region_properties(dimensions: int, calculate_advanced: bool) -> list[str]:
    if dimensions == 2:
        properties = [
            "label",
            "image",
            "area",
            "perimeter",
            "bbox",
            "bbox_area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "centroid",
            "equivalent_diameter",
            "extent",
            "eccentricity",
            "convex_area",
            "solidity",
            "euler_number",
        ]
        if calculate_advanced:
            properties.extend(
                [
                    "inertia_tensor",
                    "inertia_tensor_eigvals",
                    "moments",
                    "moments_central",
                    "moments_hu",
                    "moments_normalized",
                ]
            )
        return properties
    properties = [
        "label",
        "image",
        "area",
        "centroid",
        "bbox",
        "bbox_area",
        "inertia_tensor_eigvals",
        "extent",
        "equivalent_diameter",
        "euler_number",
    ]
    if calculate_advanced:
        properties.append("solidity")
    return properties


def _shape_feature(feature: MeasureObjectSizeShapeModule.MeasurementFeature) -> str:
    return feature.value


def _indexed_shape_feature(
    feature: MeasureObjectSizeShapeModule.MeasurementFeature, *indices: int
) -> str:
    return feature.indexed_name(*indices)


def _advanced_2d_features(props: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for row in range(3):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    MeasureObjectSizeShapeModule.MeasurementFeature.SPATIAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments-{row}-{column}"]
            features[
                _indexed_shape_feature(
                    MeasureObjectSizeShapeModule.MeasurementFeature.CENTRAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_central-{row}-{column}"]
    for row in range(4):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    MeasureObjectSizeShapeModule.MeasurementFeature.NORMALIZED_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_normalized-{row}-{column}"]
    for index in range(7):
        features[
            _indexed_shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.HU_MOMENT, index
            )
        ] = props[f"moments_hu-{index}"]
    for row in range(2):
        for column in range(2):
            features[
                _indexed_shape_feature(
                    MeasureObjectSizeShapeModule.MeasurementFeature.INERTIA_TENSOR,
                    row,
                    column,
                )
            ] = props[f"inertia_tensor-{row}-{column}"]
    for index in range(2):
        features[
            _indexed_shape_feature(
                MeasureObjectSizeShapeModule.MeasurementFeature.INERTIA_TENSOR_EIGENVALUES,
                index,
            )
        ] = props[f"inertia_tensor_eigvals-{index}"]
    return features


def _cellprofiler_orientation_degrees(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(props["orientation"], dtype=float) * (180 / np.pi)


def _cellprofiler_3d_axis_lengths(
    props: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return CellProfiler-compatible 3-D AreaShape axis lengths."""
    return (
        4.0 * np.sqrt(np.maximum(props["inertia_tensor_eigvals-0"], 0.0)),
        4.0 * np.sqrt(np.maximum(props["inertia_tensor_eigvals-2"], 0.0)),
    )


def _zernike_features(
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    backend_provider: BackendProviderInput,
) -> dict[str, np.ndarray]:
    zernike_numbers, zernike_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=MeasureObjectSizeShapeModule.zernike_max_order,
        backend_provider=backend_provider,
    )
    return {
        MeasureObjectSizeShapeModule.shape_zernike_feature_name(
            degree=int(n), repetition=int(m)
        ): values
        for (n, m), values in zip(zernike_numbers, zernike_values.transpose())
    }


def _surface_area(volume: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    if not np.any(volume):
        return 0.0
    if spacing is None:
        spacing = (1.0,) * volume.ndim
    try:
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume, method="lewiner", spacing=spacing, level=0
        )
    except ValueError:
        return 0.0
    return float(skimage.measure.mesh_surface_area(verts, faces))


def _expanded_surface_area_bounds(
    bounds_zyxzyx: np.ndarray, labels_shape: tuple[int, ...]
) -> np.ndarray:
    bounds = np.ascontiguousarray(bounds_zyxzyx, dtype=np.int64)
    if bounds.shape[1:] != (6,):
        raise ValueError(
            f"3-D surface-area bounds must have six columns (z0, y0, x0, z1, y1, x1), got {bounds.shape!r}."
        )
    if len(labels_shape) != 3:
        raise ValueError(
            f"3-D surface-area labels require a 3-D shape, got {labels_shape!r}."
        )
    expanded = bounds.copy()
    shape = np.asarray(labels_shape, dtype=np.int64)
    expanded[:, :3] = np.maximum(expanded[:, :3] - 1, 0)
    expanded[:, 3:] = np.minimum(expanded[:, 3:] + 1, shape)
    return np.ascontiguousarray(expanded, dtype=np.int64)


def _surface_areas_3d_from_labels(
    labels: np.ndarray,
    label_ids: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    return _surface_areas_3d_from_regions(
        labels,
        SurfaceArea3DRegions.from_label_array(labels, label_ids),
        spacing=spacing,
    )


def _surface_areas_3d_from_regions(
    labels: np.ndarray,
    regions: SurfaceArea3DRegions,
    *,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    if regions.label_ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    label_array = np.asarray(labels)
    if label_array.ndim != 3:
        raise ValueError("3-D surface-area measurement requires a 3-D label array.")
    surface_areas = np.zeros(regions.label_ids.size, dtype=np.float64)
    for object_index, label_id in enumerate(regions.label_ids):
        z0, y0, x0, z1, y1, x1 = (
            int(value) for value in regions.bounds_zyxzyx[object_index]
        )
        volume = label_array[z0:z1, y0:y1, x0:x1] == int(label_id)
        surface_areas[object_index] = _surface_area(volume, spacing=spacing)
    return surface_areas


def _zernike_indexes_numpy(max_order: int) -> np.ndarray:
    indexes: list[tuple[int, int]] = []
    for n_value in range(max_order + 1):
        for m_value in range(n_value + 1):
            if (n_value - m_value) % 2 == 0:
                indexes.append((n_value, m_value))
    return np.asarray(indexes, dtype=np.int64)


def _form_factor_values_from_labels(
    labels: np.ndarray, label_ids: np.ndarray
) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int32)
    label_id_array = np.asarray(label_ids, dtype=np.int32)
    if label_id_array.size == 0:
        return np.zeros(0, dtype=np.float64)
    if label_array.ndim != 2:
        raise ValueError(
            f"Form-factor values require 2-D labels, got {label_array.ndim}D."
        )
    properties = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array, include_advanced=False
    )
    max_label = int(max(label_id_array.max(initial=0), properties.label.max(initial=0)))
    areas_by_label = np.zeros(max_label + 1, dtype=np.float64)
    perimeters_by_label = np.zeros(max_label + 1, dtype=np.float64)
    if properties.label.size:
        areas_by_label[properties.label.astype(np.int32, copy=False)] = properties.area
        perimeters_by_label[properties.label.astype(np.int32, copy=False)] = (
            properties.perimeter
        )
    valid = (label_id_array > 0) & (label_id_array <= max_label)
    areas = np.zeros(label_id_array.size, dtype=np.float64)
    perimeters = np.zeros(label_id_array.size, dtype=np.float64)
    areas[valid] = areas_by_label[label_id_array[valid]]
    perimeters[valid] = perimeters_by_label[label_id_array[valid]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 4.0 * np.pi * areas / perimeters**2


def _first_scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


@njit(cache=True)
def _object_radius_features_numba(mask: np.ndarray) -> tuple[float, float, float]:
    height = mask.shape[0] + 2
    width = mask.shape[1] + 2
    inf = 1e20
    row_distances = np.empty((height, width), dtype=np.float64)
    distances_sq = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        source = np.empty(width, dtype=np.float64)
        for x in range(width):
            source[x] = 0.0
            if 0 < y < height - 1 and 0 < x < width - 1:
                if mask[y - 1, x - 1]:
                    source[x] = inf
        row_output = np.empty(width, dtype=np.float64)
        row_arg = np.empty(width, dtype=np.int64)
        _edt_1d_numba(source, row_output, row_arg)
        for x in range(width):
            row_distances[y, x] = row_output[x]
    for x in range(width):
        source = np.empty(height, dtype=np.float64)
        for y in range(height):
            source[y] = row_distances[y, x]
        column_output = np.empty(height, dtype=np.float64)
        column_arg = np.empty(height, dtype=np.int64)
        _edt_1d_numba(source, column_output, column_arg)
        for y in range(height):
            distances_sq[y, x] = column_output[y]
    count = 0
    total = 0.0
    maximum = 0.0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y - 1, x - 1]:
                value = np.sqrt(distances_sq[y, x])
                total += value
                if value > maximum:
                    maximum = value
                count += 1
    if count == 0:
        return (0.0, 0.0, 0.0)
    values = np.empty(count, dtype=np.float64)
    index = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y - 1, x - 1]:
                values[index] = np.sqrt(distances_sq[y, x])
                index += 1
    values.sort()
    middle = count // 2
    if count % 2 == 1:
        median = values[middle]
    else:
        median = 0.5 * (values[middle - 1] + values[middle])
    return (maximum, total / count, median)


@njit(cache=True)
def _radius_features_from_distance_image_numba(
    labels: np.ndarray, distances: np.ndarray, label_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for i in range(object_count):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    counts_by_label = np.zeros(max_label + 1, dtype=np.int64)
    sums_by_label = np.zeros(max_label + 1, dtype=np.float64)
    max_by_label = np.zeros(max_label + 1, dtype=np.float64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                value = distances[row, col]
                counts_by_label[label] += 1
                sums_by_label[label] += value
                if value > max_by_label[label]:
                    max_by_label[label] = value
    offsets = np.zeros(max_label + 2, dtype=np.int64)
    for label in range(max_label + 1):
        offsets[label + 1] = offsets[label] + counts_by_label[label]
    cursor = offsets.copy()
    ordered = np.empty(offsets[max_label + 1], dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                index = cursor[label]
                ordered[index] = distances[row, col]
                cursor[label] = index + 1
    max_radius = np.zeros(object_count, dtype=np.float64)
    mean_radius = np.zeros(object_count, dtype=np.float64)
    median_radius = np.zeros(object_count, dtype=np.float64)
    for i in range(object_count):
        label = int(label_ids[i])
        if label <= 0 or label > max_label:
            continue
        count = counts_by_label[label]
        if count <= 0:
            continue
        start = offsets[label]
        values = ordered[start : start + count].copy()
        values.sort()
        max_radius[i] = max_by_label[label]
        mean_radius[i] = sums_by_label[label] / count
        middle = count // 2
        if count % 2 == 1:
            median_radius[i] = values[middle]
        else:
            median_radius[i] = 0.5 * (values[middle - 1] + values[middle])
    return (max_radius, mean_radius, median_radius)


@njit(cache=True)
def _radius_features_from_labels_numba(
    labels: np.ndarray, label_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for i in range(object_count):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    height, width = labels.shape
    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label > max_label:
                continue
            counts[label] += 1
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y + 1 > max_y[label]:
                max_y[label] = y + 1
            if x + 1 > max_x[label]:
                max_x[label] = x + 1
    max_radius = np.zeros(object_count, dtype=np.float64)
    mean_radius = np.zeros(object_count, dtype=np.float64)
    median_radius = np.zeros(object_count, dtype=np.float64)
    inf = 1e20
    for object_index in range(object_count):
        label = int(label_ids[object_index])
        if label <= 0 or label > max_label or counts[label] <= 0:
            continue
        crop_height = max_y[label] - min_y[label] + 2
        crop_width = max_x[label] - min_x[label] + 2
        row_distances = np.empty((crop_height, crop_width), dtype=np.float64)
        distances_sq = np.empty((crop_height, crop_width), dtype=np.float64)
        for yy in range(crop_height):
            source = np.empty(crop_width, dtype=np.float64)
            source_y = min_y[label] + yy - 1
            for xx in range(crop_width):
                source_x = min_x[label] + xx - 1
                if (
                    source_y >= 0
                    and source_y < height
                    and (source_x >= 0)
                    and (source_x < width)
                    and (labels[source_y, source_x] == label)
                ):
                    source[xx] = inf
                else:
                    source[xx] = 0.0
            row_output = np.empty(crop_width, dtype=np.float64)
            row_arg = np.empty(crop_width, dtype=np.int64)
            _edt_1d_numba(source, row_output, row_arg)
            for xx in range(crop_width):
                row_distances[yy, xx] = row_output[xx]
        for xx in range(crop_width):
            source = np.empty(crop_height, dtype=np.float64)
            for yy in range(crop_height):
                source[yy] = row_distances[yy, xx]
            column_output = np.empty(crop_height, dtype=np.float64)
            column_arg = np.empty(crop_height, dtype=np.int64)
            _edt_1d_numba(source, column_output, column_arg)
            for yy in range(crop_height):
                distances_sq[yy, xx] = column_output[yy]
        object_pixel_count = counts[label]
        values = np.empty(object_pixel_count, dtype=np.float64)
        value_index = 0
        total = 0.0
        maximum = 0.0
        for yy in range(1, crop_height - 1):
            source_y = min_y[label] + yy - 1
            for xx in range(1, crop_width - 1):
                source_x = min_x[label] + xx - 1
                if labels[source_y, source_x] != label:
                    continue
                value = np.sqrt(distances_sq[yy, xx])
                values[value_index] = value
                value_index += 1
                total += value
                if value > maximum:
                    maximum = value
        values.sort()
        middle = object_pixel_count // 2
        max_radius[object_index] = maximum
        mean_radius[object_index] = total / object_pixel_count
        if object_pixel_count % 2 == 1:
            median_radius[object_index] = values[middle]
        else:
            median_radius[object_index] = 0.5 * (values[middle - 1] + values[middle])
    return (max_radius, mean_radius, median_radius)


@njit(cache=True)
def _distance_to_label_edge_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    output = np.zeros((height, width), dtype=np.float64)
    if max_label <= 0:
        return output
    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y + 1 > max_y[label]:
                max_y[label] = y + 1
            if x + 1 > max_x[label]:
                max_x[label] = x + 1
    inf = 1e20
    for label in range(1, max_label + 1):
        if counts[label] <= 0:
            continue
        crop_y0 = min_y[label] - 1
        if crop_y0 < 0:
            crop_y0 = 0
        crop_x0 = min_x[label] - 1
        if crop_x0 < 0:
            crop_x0 = 0
        crop_y1 = max_y[label] + 1
        if crop_y1 > height:
            crop_y1 = height
        crop_x1 = max_x[label] + 1
        if crop_x1 > width:
            crop_x1 = width
        crop_height = crop_y1 - crop_y0
        crop_width = crop_x1 - crop_x0
        has_background = False
        for yy in range(crop_height):
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] != label:
                    has_background = True
                    break
            if has_background:
                break
        if not has_background:
            for yy in range(crop_height):
                source_y = crop_y0 + yy
                y_distance = yy + 1
                for xx in range(crop_width):
                    source_x = crop_x0 + xx
                    output[source_y, source_x] = np.sqrt(
                        y_distance * y_distance + xx * xx
                    )
            continue
        row_distances = np.empty((crop_height, crop_width), dtype=np.float64)
        distances_sq = np.empty((crop_height, crop_width), dtype=np.float64)
        for yy in range(crop_height):
            source = np.empty(crop_width, dtype=np.float64)
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] == label:
                    source[xx] = inf
                else:
                    source[xx] = 0.0
            row_output = np.empty(crop_width, dtype=np.float64)
            row_arg = np.empty(crop_width, dtype=np.int64)
            _edt_1d_numba(source, row_output, row_arg)
            for xx in range(crop_width):
                row_distances[yy, xx] = row_output[xx]
        for xx in range(crop_width):
            source = np.empty(crop_height, dtype=np.float64)
            for yy in range(crop_height):
                source[yy] = row_distances[yy, xx]
            column_output = np.empty(crop_height, dtype=np.float64)
            column_arg = np.empty(crop_height, dtype=np.int64)
            _edt_1d_numba(source, column_output, column_arg)
            for yy in range(crop_height):
                distances_sq[yy, xx] = column_output[yy]
        for yy in range(crop_height):
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] == label:
                    output[source_y, source_x] = np.sqrt(distances_sq[yy, xx])
    return output


@njit(cache=True)
def _maximum_position_of_labels_numba(
    image: np.ndarray, labels: np.ndarray, label_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for index in range(object_count):
        label = int(label_ids[index])
        if label > max_label:
            max_label = label
    best_values = np.full(max_label + 1, -np.inf, dtype=np.float64)
    best_y = np.full(max_label + 1, -1, dtype=np.int64)
    best_x = np.full(max_label + 1, -1, dtype=np.int64)
    seen = np.zeros(max_label + 1, dtype=np.bool_)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label > max_label:
                continue
            value = image[y, x]
            if not seen[label] or value > best_values[label]:
                seen[label] = True
                best_values[label] = value
                best_y[label] = y
                best_x[label] = x
    centers_i = np.zeros(object_count, dtype=np.float64)
    centers_j = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        label = int(label_ids[index])
        if label > 0 and label <= max_label and seen[label]:
            centers_i[index] = float(best_y[label])
            centers_j[index] = float(best_x[label])
    return (centers_i, centers_j)


def _maximum_position_of_labels_scipy_select(
    image: np.ndarray, labels: np.ndarray, label_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum positions using CellProfiler 4.2 labeled tie semantics."""
    image_array = np.asarray(image)
    label_array = np.asarray(labels, dtype=np.int32)
    label_id_array = np.asarray(label_ids, dtype=np.int32)
    if image_array.shape != label_array.shape:
        raise ValueError(
            f"Maximum-position image and labels must have matching shapes; got {image_array.shape!r} and {label_array.shape!r}."
        )
    if image_array.ndim != 2:
        raise ValueError(
            f"Maximum-position labels must be 2D, got {image_array.ndim}D."
        )
    if label_id_array.size == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    max_label = int(np.max(label_array)) if label_array.size else 0
    positions = np.arange(image_array.size, dtype=np.int64)
    order = _numpy124_aquicksort_indices(np.asarray(image_array).ravel())
    sorted_labels = label_array.ravel()[order]
    sorted_positions = positions[order]
    max_positions = np.zeros(max_label + 2, dtype=np.int64)
    valid_sorted = (sorted_labels >= 0) & (sorted_labels <= max_label)
    valid_sorted_labels = sorted_labels[valid_sorted]
    valid_sorted_positions = sorted_positions[valid_sorted]
    max_positions[valid_sorted_labels] = valid_sorted_positions
    safe_label_ids = np.zeros(label_id_array.shape, dtype=np.int64)
    present = (label_id_array >= 0) & (label_id_array <= max_label)
    safe_label_ids[present] = label_id_array[present]
    selected_positions = max_positions[safe_label_ids]
    centers_i = (selected_positions // image_array.shape[1]).astype(np.float64)
    centers_j = (selected_positions % image_array.shape[1]).astype(np.float64)
    return (centers_i, centers_j)


@njit(cache=True)
def _numpy124_msb_numba(value: int) -> int:
    depth_limit = 0
    while value >> 1:
        value >>= 1
        depth_limit += 1
    return depth_limit


@njit(cache=True)
def _numpy124_aheapsort_indices_numba(
    values: np.ndarray, indices: np.ndarray, start: int, count: int
) -> None:
    n = count
    level = n >> 1
    while level > 0:
        temporary = indices[start + level - 1]
        parent = level
        child = level << 1
        while child <= n:
            if (
                child < n
                and values[indices[start + child - 1]] < values[indices[start + child]]
            ):
                child += 1
            if values[temporary] < values[indices[start + child - 1]]:
                indices[start + parent - 1] = indices[start + child - 1]
                parent = child
                child += child
            else:
                break
        indices[start + parent - 1] = temporary
        level -= 1
    while n > 1:
        temporary = indices[start + n - 1]
        indices[start + n - 1] = indices[start]
        n -= 1
        parent = 1
        child = 2
        while child <= n:
            if (
                child < n
                and values[indices[start + child - 1]] < values[indices[start + child]]
            ):
                child += 1
            if values[temporary] < values[indices[start + child - 1]]:
                indices[start + parent - 1] = indices[start + child - 1]
                parent = child
                child += child
            else:
                break
        indices[start + parent - 1] = temporary


@njit(cache=True)
def _numpy124_aquicksort_indices_numba(values: np.ndarray) -> np.ndarray:
    count = values.size
    indices = np.arange(count, dtype=np.int64)
    if count < 2:
        return indices
    stack_left = np.empty(128, dtype=np.int64)
    stack_right = np.empty(128, dtype=np.int64)
    stack_depth = np.empty(128, dtype=np.int64)
    stack_size = 0
    left = 0
    right = count - 1
    current_depth = _numpy124_msb_numba(count) * 2
    while True:
        if current_depth < 0:
            _numpy124_aheapsort_indices_numba(values, indices, left, right - left + 1)
            if stack_size == 0:
                break
            stack_size -= 1
            left = stack_left[stack_size]
            right = stack_right[stack_size]
            current_depth = stack_depth[stack_size]
            continue
        while right - left > 15:
            middle = left + (right - left >> 1)
            if values[indices[middle]] < values[indices[left]]:
                indices[middle], indices[left] = (indices[left], indices[middle])
            if values[indices[right]] < values[indices[middle]]:
                indices[right], indices[middle] = (indices[middle], indices[right])
            if values[indices[middle]] < values[indices[left]]:
                indices[middle], indices[left] = (indices[left], indices[middle])
            pivot_value = values[indices[middle]]
            scan_left = left
            scan_right = right - 1
            indices[middle], indices[scan_right] = (
                indices[scan_right],
                indices[middle],
            )
            while True:
                scan_left += 1
                while values[indices[scan_left]] < pivot_value:
                    scan_left += 1
                scan_right -= 1
                while pivot_value < values[indices[scan_right]]:
                    scan_right -= 1
                if scan_left >= scan_right:
                    break
                indices[scan_left], indices[scan_right] = (
                    indices[scan_right],
                    indices[scan_left],
                )
            pivot_slot = right - 1
            indices[scan_left], indices[pivot_slot] = (
                indices[pivot_slot],
                indices[scan_left],
            )
            if scan_left - left < right - scan_left:
                stack_left[stack_size] = scan_left + 1
                stack_right[stack_size] = right
                stack_size += 1
                right = scan_left - 1
            else:
                stack_left[stack_size] = left
                stack_right[stack_size] = scan_left - 1
                stack_size += 1
                left = scan_left + 1
            current_depth -= 1
            stack_depth[stack_size - 1] = current_depth
        insertion_index = left + 1
        while insertion_index <= right:
            current_index = indices[insertion_index]
            current_value = values[current_index]
            target = insertion_index
            previous = insertion_index - 1
            while target > left and current_value < values[indices[previous]]:
                indices[target] = indices[previous]
                target -= 1
                previous -= 1
            indices[target] = current_index
            insertion_index += 1
        if stack_size == 0:
            break
        stack_size -= 1
        left = stack_left[stack_size]
        right = stack_right[stack_size]
        current_depth = stack_depth[stack_size]
    return indices


def _numpy124_aquicksort_indices(values: np.ndarray) -> np.ndarray:
    return _numpy124_aquicksort_indices_numba(np.asarray(values))


def _color_labels_numpy(labels: np.ndarray) -> np.ndarray:
    """Return CP-compatible non-touching label color classes."""
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.size == 0:
        return np.zeros(label_array.shape, dtype=int)
    if not _has_touching_foreground_labels_numba(np.ascontiguousarray(label_array)):
        return (label_array != 0).astype(int)
    neighbor_counts, neighbor_starts, neighbor_labels = _find_label_neighbors_numpy(
        label_array
    )
    colors_by_label = np.zeros(neighbor_counts.size + 1, dtype=int)
    if neighbor_counts.size == 0:
        return colors_by_label[label_array]
    isolated_labels = neighbor_counts == 0
    if np.all(isolated_labels):
        return (label_array != 0).astype(int)
    colors_by_label[1:][isolated_labels] = 1
    connected_counts = neighbor_counts[~isolated_labels]
    connected_starts = neighbor_starts[~isolated_labels]
    connected_labels = np.flatnonzero(~isolated_labels) + 1
    sort_order = np.lexsort((-connected_counts,))
    connected_counts = connected_counts[sort_order]
    connected_starts = connected_starts[sort_order]
    connected_labels = connected_labels[sort_order]
    for index in range(connected_counts.size):
        start = int(connected_starts[index])
        end = start + int(connected_counts[index])
        neighbor_colors = np.unique(colors_by_label[neighbor_labels[start:end]])
        if neighbor_colors.size == 1 and neighbor_colors[0] == 0:
            colors_by_label[connected_labels[index]] = 1
            continue
        if neighbor_colors[0] == 0:
            neighbor_colors = neighbor_colors[1:]
        expected_colors = np.arange(1, neighbor_colors.size + 1)
        missing_color_positions = expected_colors[neighbor_colors != expected_colors]
        if missing_color_positions.size:
            colors_by_label[connected_labels[index]] = int(missing_color_positions[0])
        else:
            colors_by_label[connected_labels[index]] = int(neighbor_colors.size + 1)
    return colors_by_label[label_array]


@njit(cache=True)
def _has_touching_foreground_labels_numba(labels: np.ndarray) -> bool:
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            if x + 1 < width:
                neighbor = labels[y, x + 1]
                if neighbor > 0 and neighbor != label:
                    return True
            if y + 1 < height:
                neighbor = labels[y + 1, x]
                if neighbor > 0 and neighbor != label:
                    return True
                if x + 1 < width:
                    neighbor = labels[y + 1, x + 1]
                    if neighbor > 0 and neighbor != label:
                        return True
                if x > 0:
                    neighbor = labels[y + 1, x - 1]
                    if neighbor > 0 and neighbor != label:
                        return True
    return False


def _find_label_neighbors_numpy(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-label 8-connected neighboring label lists."""
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.size == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    max_label = int(np.max(label_array))
    padded = np.zeros(np.asarray(label_array.shape) + 2, dtype=np.int32)
    padded[1:-1, 1:-1] = label_array
    adjacent_y, adjacent_x = np.argwhere(_adjacent_label_mask_numpy(padded)).transpose()
    if adjacent_y.size == 0:
        return (
            np.zeros(max_label, dtype=int),
            np.zeros(max_label, dtype=int),
            np.zeros(0, dtype=int),
        )
    repeated_labels = np.hstack([padded[adjacent_y, adjacent_x]] * 8)
    neighbor_values = np.zeros(adjacent_y.size * 8, dtype=int)
    offset = 0
    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        neighbor_values[offset : offset + adjacent_y.size] = padded[
            adjacent_y + dy, adjacent_x + dx
        ]
        offset += adjacent_y.size
    sort_order = np.lexsort((neighbor_values, repeated_labels))
    repeated_labels = repeated_labels[sort_order]
    neighbor_values = neighbor_values[sort_order]
    first_occurrence = np.ones(repeated_labels.size, dtype=bool)
    first_occurrence[1:] = (repeated_labels[1:] != repeated_labels[:-1]) | (
        neighbor_values[1:] != neighbor_values[:-1]
    )
    repeated_labels = repeated_labels[first_occurrence]
    neighbor_values = neighbor_values[first_occurrence]
    keep = (repeated_labels != neighbor_values) & (neighbor_values != 0)
    repeated_labels = repeated_labels[keep]
    neighbor_values = neighbor_values[keep]
    neighbor_counts = np.bincount(repeated_labels, minlength=max_label + 1)[1:].astype(
        int
    )
    neighbor_starts = np.cumsum(neighbor_counts)
    if neighbor_starts.size:
        neighbor_starts[1:] = neighbor_starts[:-1]
        neighbor_starts[0] = 0
    return (neighbor_counts, neighbor_starts, neighbor_values)


def _adjacent_label_mask_numpy(labels: np.ndarray) -> np.ndarray:
    """Return foreground labels touching a different 8-connected label."""
    label_array = labels.astype(np.int32, copy=False)
    high = int(label_array.max()) + 1 if label_array.size else 1
    image_with_high_background = label_array.copy()
    image_with_high_background[label_array == 0] = high
    footprint = np.ones((3, 3), dtype=bool)
    minimum_label = scipy.ndimage.minimum_filter(
        image_with_high_background, footprint=footprint, mode="constant", cval=high
    )
    maximum_label = scipy.ndimage.maximum_filter(
        label_array, footprint=footprint, mode="constant", cval=0
    )
    return (minimum_label != maximum_label) & (label_array > 0)


__all__ = [
    "LegacyFastNumpyShapeMeasurementBackendStrategy",
    "NumbaNumpyShapeMeasurementBackendStrategy",
    "ShapeMeasurementBackendStrategy",
    "form_factor_values",
    "shape_measurement_backend",
]
