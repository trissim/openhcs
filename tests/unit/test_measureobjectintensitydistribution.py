import numpy as np

from benchmark.cellprofiler_library.functions import measureobjectintensitydistribution as mid
from openhcs.core.config import DtypeConfig
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    NativeNumpyRadialDistributionBackendStrategy,
    radial_distribution_backend,
)


def test_native_radial_distribution_excludes_pixels_without_valid_center():
    image = np.ones((2, 2), dtype=np.float32)
    labels = np.ones((2, 2), dtype=np.int32)

    radial_arrays = NativeNumpyRadialDistributionBackendStrategy().measure_from_centers(
        image,
        labels,
        np.zeros(labels.shape, dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )

    assert not radial_arrays.object_has_pixels[0]
    assert np.all(np.isnan(radial_arrays.fraction_at_distance[0]))
    assert np.all(np.isnan(radial_arrays.mean_pixel_fraction[0]))
    assert np.all(radial_arrays.radial_cv_by_bin[:, 0] == 0.0)


def test_radial_distribution_marks_missing_dense_label_fraction_fields_nan():
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.array(
        [
            [1, 0, 3],
            [1, 0, 3],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    _result, measurements = mid.measure_object_intensity_distribution(
        image,
        labels,
        bin_count=4,
        dtype_config=DtypeConfig(),
    )

    missing_label_measurements = [
        measurement for measurement in measurements
        if measurement.object_label == 2
    ]

    assert len(missing_label_measurements) == 4
    assert all(np.isnan(measurement.frac_at_d) for measurement in missing_label_measurements)
    assert all(np.isnan(measurement.mean_frac) for measurement in missing_label_measurements)
    assert all(measurement.radial_cv == 0.0 for measurement in missing_label_measurements)


def test_radial_cv_ignores_empty_angular_wedges():
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.ones((3, 3), dtype=np.int32)

    radial_arrays = NativeNumpyRadialDistributionBackendStrategy().measure(
        image,
        labels,
        np.zeros(labels.shape, dtype=np.float64),
        np.zeros(labels.shape, dtype=np.float64),
        np.ones(labels.shape, dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )

    assert radial_arrays.radial_cv_by_bin[0, 0] == 0.0


def test_explicit_numba_radial_provider_remains_available():
    selected = radial_distribution_backend(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    assert selected.backend_provider is CellProfilerBackendProvider.NUMBA
