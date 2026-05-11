"""Shape-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
import time

import numpy as np
import scipy.ndimage
import skimage.measure
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.runtime_semantics import (
    ObjectShapeMeasurementFeature,
    indexed_measurement_feature_name,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.secondary import _edt_1d_numba
from openhcs.processing.backends.cellprofiler.zernike import shape_zernike_moments

_ZERNIKE_MAX_ORDER = 9
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True, slots=True)
class ObjectSizeShapeFeatureMeasurement:
    """Backend-owned CellProfiler AreaShape feature-array measurement."""

    labels: np.ndarray
    calculate_advanced: bool
    calculate_zernikes: bool
    shape_backend_provider: BackendProviderInput | None = None
    zernike_backend_provider: BackendProviderInput | None = None

    def feature_arrays(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return feature arrays and measured label ids for 2-D or 3-D labels."""
        label_array = np.asarray(self.labels, dtype=np.int32)
        if label_array.ndim == 2:
            return self._feature_arrays_2d(label_array)
        if label_array.ndim == 3:
            return self._feature_arrays_3d(label_array)
        raise ValueError(
            f"Object labels must be 2D or 3D, got {label_array.ndim}D."
        )

    def _feature_arrays_2d(
        self,
        labels: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        total_started_at = time.perf_counter()
        phase_started_at = time.perf_counter()
        shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
            backend_provider=self.shape_backend_provider,
        )
        _log_profile(
            "moss_backend_resolution",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
        )
        phase_started_at = time.perf_counter()
        fast_region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
            labels
        )
        _log_profile(
            "moss_region_properties",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(fast_region_props.label.size),
        )
        phase_started_at = time.perf_counter()
        props = fast_region_props.as_regionprops_table_subset()
        _log_profile(
            "moss_regionprops_table_subset",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            fields=len(props),
        )
        phase_started_at = time.perf_counter()
        convex_area, solidity = _convex_area_and_solidity_from_labels(
            labels,
            fast_region_props,
        )
        _log_profile(
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
            return {}, measured_labels

        perimeter = np.asarray(props["perimeter"], dtype=float)
        area = np.asarray(props["area"], dtype=float)
        phase_started_at = time.perf_counter()
        max_radius, mean_radius, median_radius = shape_backend.radius_features_from_labels(
            labels,
            measured_labels,
        )
        _log_profile(
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
            labels,
            measured_labels,
        )
        _log_profile(
            "moss_feret_diameters",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(measured_labels.size),
        )

        phase_started_at = time.perf_counter()
        dense_center_y, dense_center_x = _dense_label_centers_2d(labels)
        _log_profile(
            "moss_dense_centers",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=int(dense_center_x.size),
        )
        center_x = _compact_values_with_dense_tail(
            np.asarray(props["centroid-1"], dtype=float),
            dense_center_x,
            measured_labels=measured_labels,
        )
        center_y = _compact_values_with_dense_tail(
            np.asarray(props["centroid-0"], dtype=float),
            dense_center_y,
            measured_labels=measured_labels,
        )

        features = {
            _shape_feature(ObjectShapeMeasurementFeature.AREA): area,
            _shape_feature(ObjectShapeMeasurementFeature.PERIMETER): perimeter,
            _shape_feature(ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH): props[
                "major_axis_length"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH): props[
                "minor_axis_length"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.ECCENTRICITY): props[
                "eccentricity"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.ORIENTATION): (
                _cellprofiler_orientation_degrees(props)
            ),
            _shape_feature(ObjectShapeMeasurementFeature.CENTER_X): center_x,
            _shape_feature(ObjectShapeMeasurementFeature.CENTER_Y): center_y,
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_AREA): props[
                "bbox_area"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X): props[
                "bbox-1"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X): props[
                "bbox-3"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y): props[
                "bbox-0"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y): props[
                "bbox-2"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.FORM_FACTOR): form_factor,
            _shape_feature(ObjectShapeMeasurementFeature.EXTENT): props["extent"],
            _shape_feature(ObjectShapeMeasurementFeature.SOLIDITY): props["solidity"],
            _shape_feature(ObjectShapeMeasurementFeature.COMPACTNESS): compactness,
            _shape_feature(ObjectShapeMeasurementFeature.EULER_NUMBER): props[
                "euler_number"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.MAXIMUM_RADIUS): max_radius,
            _shape_feature(ObjectShapeMeasurementFeature.MEAN_RADIUS): mean_radius,
            _shape_feature(ObjectShapeMeasurementFeature.MEDIAN_RADIUS): median_radius,
            _shape_feature(ObjectShapeMeasurementFeature.CONVEX_AREA): props[
                "convex_area"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.MIN_FERET_DIAMETER): (
                min_feret_diameter
            ),
            _shape_feature(ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER): (
                max_feret_diameter
            ),
            _shape_feature(ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER): props[
                "equivalent_diameter"
            ],
        }
        if self.calculate_advanced:
            phase_started_at = time.perf_counter()
            features.update(_advanced_2d_features(props))
            _log_profile(
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
            _log_profile(
                "moss_zernike_features",
                time.perf_counter() - phase_started_at,
                function="measure_object_size_shape",
                objects=nobjects,
            )
        _log_profile(
            "moss_features_2d_total",
            time.perf_counter() - total_started_at,
            function="measure_object_size_shape",
            objects=nobjects,
        )
        return features, measured_labels

    def _feature_arrays_3d(
        self,
        labels: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
            backend_provider=self.shape_backend_provider,
        )
        props = skimage.measure.regionprops_table(
            labels,
            properties=_desired_region_properties(3, self.calculate_advanced),
        )
        measured_labels = np.asarray(props["label"])
        inertia_tensor_eigvals = np.stack(
            (
                props["inertia_tensor_eigvals-0"],
                props["inertia_tensor_eigvals-1"],
                props["inertia_tensor_eigvals-2"],
            ),
            axis=1,
        )
        major_axis_length, minor_axis_length = (
            shape_backend.axis_lengths_3d_from_inertia_eigvals(inertia_tensor_eigvals)
        )
        surface_areas = np.zeros(len(measured_labels), dtype=float)
        for index, label in enumerate(measured_labels):
            volume = labels[
                max(props["bbox-0"][index] - 1, 0) : min(
                    props["bbox-3"][index] + 1,
                    labels.shape[0],
                ),
                max(props["bbox-1"][index] - 1, 0) : min(
                    props["bbox-4"][index] + 1,
                    labels.shape[1],
                ),
                max(props["bbox-2"][index] - 1, 0) : min(
                    props["bbox-5"][index] + 1,
                    labels.shape[2],
                ),
            ]
            surface_areas[index] = _surface_area(volume == label)

        features = {
            _shape_feature(ObjectShapeMeasurementFeature.VOLUME): props["area"],
            _shape_feature(ObjectShapeMeasurementFeature.SURFACE_AREA): surface_areas,
            _shape_feature(ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH): (
                major_axis_length
            ),
            _shape_feature(ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH): (
                minor_axis_length
            ),
            _shape_feature(ObjectShapeMeasurementFeature.CENTER_X): props["centroid-2"],
            _shape_feature(ObjectShapeMeasurementFeature.CENTER_Y): props["centroid-1"],
            _shape_feature(ObjectShapeMeasurementFeature.CENTER_Z): props["centroid-0"],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_VOLUME): props[
                "bbox_area"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X): props[
                "bbox-2"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X): props[
                "bbox-5"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y): props[
                "bbox-1"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y): props[
                "bbox-4"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Z): props[
                "bbox-0"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Z): props[
                "bbox-3"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.EXTENT): props["extent"],
            _shape_feature(ObjectShapeMeasurementFeature.EULER_NUMBER): props[
                "euler_number"
            ],
            _shape_feature(ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER): props[
                "equivalent_diameter"
            ],
        }
        if self.calculate_advanced:
            features[_shape_feature(ObjectShapeMeasurementFeature.SOLIDITY)] = props[
                "solidity"
            ]
        return features, measured_labels


def measure_object_size_shape_feature_arrays(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: BackendProviderInput | None = None,
    zernike_backend_provider: BackendProviderInput | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return CellProfiler AreaShape feature arrays for dense labels."""
    return ObjectSizeShapeFeatureMeasurement(
        labels=np.asarray(labels, dtype=np.int32),
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
    ).feature_arrays()


class ShapeMeasurementBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shape-measurement operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        """Return CP-compatible AreaShape_FormFactor values."""

    @abstractmethod
    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii."""

    @abstractmethod
    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii from dense labels."""

    @abstractmethod
    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum Feret diameters."""

    @abstractmethod
    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return object center coordinates and radii."""

    def axis_lengths_3d_from_inertia_eigvals(
        self,
        inertia_tensor_eigvals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return numerically stable 3-D major and minor axis lengths."""
        eigvals = np.asarray(inertia_tensor_eigvals, dtype=np.float64)
        if eigvals.ndim != 2 or eigvals.shape[1] != 3:
            raise ValueError(
                "3-D axis length calculation requires inertia eigenvalues with "
                "shape (object_count, 3)."
            )
        major_argument = 10.0 * (eigvals[:, 0] + eigvals[:, 1] - eigvals[:, 2])
        minor_argument = 10.0 * (-eigvals[:, 0] + eigvals[:, 1] + eigvals[:, 2])
        return np.sqrt(np.maximum(major_argument, 0.0)), np.sqrt(
            np.maximum(minor_argument, 0.0)
        )

    @abstractmethod
    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        """Return per-pixel distance-to-edge for labeled objects."""

    @abstractmethod
    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
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
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        """Return Zernike polynomial values at normalized coordinates."""


class CentrosomeNumpyShapeMeasurementBackendStrategy(ShapeMeasurementBackendStrategy):
    """Centrosome-backed NumPy shape measurements."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        import centrosome.cpmorphology

        labels_array = np.asarray(labels, dtype=np.int32)
        label_ids_array = np.asarray(label_ids, dtype=np.int32)
        if label_ids_array.size == 0:
            return np.array([], dtype=float)
        areas = np.bincount(
            labels_array.ravel(),
            minlength=int(label_ids_array[-1]) + 1,
        )[label_ids_array]
        perimeters = np.asarray(
            centrosome.cpmorphology.calculate_perimeters(
                labels_array,
                label_ids_array,
            ),
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return 4.0 * np.pi * areas.astype(float) / perimeters**2

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        max_radius = np.zeros(object_count)
        mean_radius = np.zeros(object_count)
        median_radius = np.zeros(object_count)
        for index, object_image in enumerate(object_images):
            mini_image = np.pad(object_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = _first_scalar(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
            )
            mean_radius[index] = _first_scalar(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
            )
            median_radius[index] = _first_scalar(
                centrosome.cpmorphology.median_of_labels(
                    distances,
                    mini_image.astype("int"),
                    [1],
                )
            )
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return radius features via one full-image distance-to-edge pass."""
        distances = self.distance_to_edge(labels)
        return _radius_features_from_distance_image_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(distances, dtype=np.float64),
            np.asarray(label_ids, dtype=np.int32),
        )

    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        chulls, chull_counts = centrosome.cpmorphology.convex_hull(
            labels,
            label_ids,
        )
        return centrosome.cpmorphology.feret_diameter(
            chulls,
            chull_counts,
            label_ids,
        )

    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.minimum_enclosing_circle(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.distance_to_edge(
            np.asarray(labels, dtype=np.int32)
        )

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        centers_i, centers_j = centrosome.cpmorphology.maximum_position_of_labels(
            np.asarray(image),
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )
        return centers_i, centers_j

    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.color_labels(
            np.asarray(labels, dtype=np.int32)
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.propagate

        return centrosome.propagate.propagate(
            np.asarray(image, dtype=np.float64),
            np.asarray(labels, dtype=np.int32),
            np.asarray(mask, dtype=bool),
            regularization_factor,
        )

    def zernike_indexes(self, max_order: int) -> np.ndarray:
        import centrosome.zernike

        return centrosome.zernike.get_zernike_indexes(int(max_order))

    def construct_zernike_polynomials(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        import centrosome.zernike

        return centrosome.zernike.construct_zernike_polynomials(
            x,
            y,
            zernike_indexes,
        )


class LegacyFastNumpyShapeMeasurementBackendStrategy(
    CentrosomeNumpyShapeMeasurementBackendStrategy
):
    """Mixed legacy-fast shape backend with explicit centrosome exact leaves."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = True

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_radius = np.zeros(object_count, dtype=np.float64)
        mean_radius = np.zeros(object_count, dtype=np.float64)
        median_radius = np.zeros(object_count, dtype=np.float64)
        for index, object_image in enumerate(object_images):
            max_value, mean_value, median_value = _object_radius_features_numba(
                np.asarray(object_image, dtype=np.bool_),
            )
            max_radius[index] = max_value
            mean_radius[index] = mean_value
            median_radius[index] = median_value
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _radius_features_from_labels_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            return _distance_to_edge_planewise(self, label_array)
        return _distance_to_label_edge_numba(np.ascontiguousarray(label_array))

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _maximum_position_of_labels_numba(
            np.ascontiguousarray(np.asarray(image, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(labels, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(label_ids, dtype=np.int32)),
        )


class NumbaNumpyShapeMeasurementBackendStrategy(ShapeMeasurementBackendStrategy):
    """Pure Numba shape backend. Unsupported leaves fail explicitly."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba form-factor values are not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_radius = np.zeros(object_count, dtype=np.float64)
        mean_radius = np.zeros(object_count, dtype=np.float64)
        median_radius = np.zeros(object_count, dtype=np.float64)
        for index, object_image in enumerate(object_images):
            max_value, mean_value, median_value = _object_radius_features_numba(
                np.asarray(object_image, dtype=np.bool_),
            )
            max_radius[index] = max_value
            mean_radius[index] = mean_value
            median_radius[index] = median_value
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _radius_features_from_labels_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba Feret diameters are not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba minimum enclosing circle is not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            return _distance_to_edge_planewise(self, label_array)
        return _distance_to_label_edge_numba(np.ascontiguousarray(label_array))

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _maximum_position_of_labels_numba(
            np.ascontiguousarray(np.asarray(image, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(labels, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(label_ids, dtype=np.int32)),
        )

    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba label coloring is not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba propagation is not implemented yet. "
            "Select CENTROSOME explicitly for this leaf."
        )

    def zernike_indexes(self, max_order: int) -> np.ndarray:
        return _zernike_indexes_numpy(int(max_order))

    def construct_zernike_polynomials(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba Zernike polynomial construction is not implemented in "
            "the shape backend. Use the zernike backend family instead."
        )


def _distance_to_edge_planewise(
    backend: ShapeMeasurementBackendStrategy,
    labels: np.ndarray,
) -> np.ndarray:
    if labels.ndim < 2:
        raise ValueError("Distance-to-edge requires at least two dimensions.")
    distances = np.empty(labels.shape, dtype=np.float64)
    plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
    source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
    target_planes = distances.reshape((plane_count, *labels.shape[-2:]))
    for plane_index in range(plane_count):
        target_planes[plane_index] = backend.distance_to_edge(source_planes[plane_index])
    return distances


def shape_measurement_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ShapeMeasurementBackendStrategy:
    """Return the selected shape-measurement backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def form_factor_values(
    labels: np.ndarray,
    label_ids: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> np.ndarray:
    """Return CP-compatible AreaShape_FormFactor values through a backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).form_factor_values(labels, label_ids)


def _convex_area_and_solidity_from_labels(
    labels: np.ndarray,
    region_props: object,
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
    return convex_area, solidity


def _desired_region_properties(
    dimensions: int,
    calculate_advanced: bool,
) -> list[str]:
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


def _shape_feature(feature: ObjectShapeMeasurementFeature) -> str:
    return feature.value


def _indexed_shape_feature(
    feature: ObjectShapeMeasurementFeature,
    *indices: int,
) -> str:
    return indexed_measurement_feature_name(feature, *indices)


def _dense_label_centers_2d(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _dense_label_centers_2d_numba(np.ascontiguousarray(labels, dtype=np.int32))


@njit(cache=True)
def _dense_label_centers_2d_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    counts = np.zeros(max_label + 1, dtype=np.float64)
    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1.0
            y_sums[label] += float(y)
            x_sums[label] += float(x)

    center_y = np.empty(max_label, dtype=np.float64)
    center_x = np.empty(max_label, dtype=np.float64)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count > 0.0:
            center_y[label - 1] = y_sums[label] / count
            center_x[label - 1] = x_sums[label] / count
        else:
            center_y[label - 1] = np.nan
            center_x[label - 1] = np.nan
    return center_y, center_x


def _compact_values_with_dense_tail(
    compact_values: np.ndarray,
    dense_values: np.ndarray,
    *,
    measured_labels: np.ndarray,
) -> np.ndarray:
    compact = np.asarray(compact_values, dtype=float)
    dense = np.asarray(dense_values, dtype=float)
    if dense.shape[0] <= compact.shape[0]:
        return compact
    values = dense.copy()
    for index, object_label in enumerate(np.asarray(measured_labels, dtype=np.int32)):
        value_index = int(object_label) - 1
        if 0 <= value_index < values.shape[0] and index < compact.shape[0]:
            values[value_index] = compact[index]
    return values


def _advanced_2d_features(props: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for row in range(3):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.SPATIAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments-{row}-{column}"]
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.CENTRAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_central-{row}-{column}"]

    for row in range(4):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.NORMALIZED_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_normalized-{row}-{column}"]

    for index in range(7):
        features[
            _indexed_shape_feature(ObjectShapeMeasurementFeature.HU_MOMENT, index)
        ] = props[f"moments_hu-{index}"]

    for row in range(2):
        for column in range(2):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.INERTIA_TENSOR,
                    row,
                    column,
                )
            ] = props[f"inertia_tensor-{row}-{column}"]

    for index in range(2):
        features[
            _indexed_shape_feature(
                ObjectShapeMeasurementFeature.INERTIA_TENSOR_EIGENVALUES,
                index,
            )
        ] = props[f"inertia_tensor_eigvals-{index}"]
    return features


def _cellprofiler_orientation_degrees(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(props["orientation"], dtype=float) * (180 / np.pi)


def _zernike_features(
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None,
) -> dict[str, np.ndarray]:
    zernike_numbers, zernike_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=_ZERNIKE_MAX_ORDER,
        backend_provider=backend_provider,
    )
    return {
        _indexed_shape_feature(ObjectShapeMeasurementFeature.ZERNIKE, int(n), int(m)): (
            values
        )
        for (n, m), values in zip(zernike_numbers, zernike_values.transpose())
    }


def _surface_area(volume: np.ndarray) -> float:
    if not np.any(volume):
        return 0.0
    try:
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume,
            method="lewiner",
            spacing=(1.0,) * volume.ndim,
            level=0,
        )
    except ValueError:
        return 0.0
    return float(skimage.measure.mesh_surface_area(verts, faces))


def _zernike_indexes_numpy(max_order: int) -> np.ndarray:
    indexes: list[tuple[int, int]] = []
    for n_value in range(max_order + 1):
        for m_value in range(n_value + 1):
            if (n_value - m_value) % 2 == 0:
                indexes.append((n_value, m_value))
    return np.asarray(indexes, dtype=np.int64)


def _first_scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


@njit(cache=True)
def _object_radius_features_numba(mask: np.ndarray) -> tuple[float, float, float]:
    height = mask.shape[0] + 2
    width = mask.shape[1] + 2
    inf = 1.0e20

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
        return 0.0, 0.0, 0.0

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
    return maximum, total / count, median


@njit(cache=True)
def _radius_features_from_distance_image_numba(
    labels: np.ndarray,
    distances: np.ndarray,
    label_ids: np.ndarray,
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
    return max_radius, mean_radius, median_radius


@njit(cache=True)
def _radius_features_from_labels_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
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
    inf = 1.0e20
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
                    and source_x >= 0
                    and source_x < width
                    and labels[source_y, source_x] == label
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
            median_radius[object_index] = 0.5 * (
                values[middle - 1] + values[middle]
            )
    return max_radius, mean_radius, median_radius


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

    inf = 1.0e20
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
                        (y_distance * y_distance) + (xx * xx)
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
    image: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
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
            if (not seen[label]) or value > best_values[label]:
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
    return centers_i, centers_j


__all__ = [
    "CentrosomeNumpyShapeMeasurementBackendStrategy",
    "LegacyFastNumpyShapeMeasurementBackendStrategy",
    "NumbaNumpyShapeMeasurementBackendStrategy",
    "ShapeMeasurementBackendStrategy",
    "form_factor_values",
    "shape_measurement_backend",
]
