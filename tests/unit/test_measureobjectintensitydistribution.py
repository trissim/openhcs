import numpy as np

from benchmark.cellprofiler_library.functions import measureobjectintensitydistribution as mid
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_semantics import (
    ObjectIntensityDistributionMeasurementFeature,
    indexed_object_intensity_distribution_feature_name,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    NativeNumpyRadialDistributionBackendStrategy,
    NumbaNumpyRadialDistributionBackendStrategy,
    radial_distribution_backend,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    secondary_propagation_backend,
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


def test_radial_distribution_uses_dense_extent_domain_for_missing_object_rows():
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

    gap_label_measurements = [
        measurement for measurement in measurements
        if measurement.object_label == 2
    ]
    label_three_measurements = [
        measurement for measurement in measurements
        if measurement.object_label == 3
    ]

    assert len(gap_label_measurements) == 12
    assert len(label_three_measurements) == 12
    gap_values_by_feature = {
        measurement.feature_name: measurement.result_value
        for measurement in gap_label_measurements
    }
    values_by_feature = {
        measurement.feature_name: measurement.result_value
        for measurement in label_three_measurements
    }
    for bin_index in range(1, 5):
        assert np.isfinite(
            values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.FRACTION_AT_DISTANCE,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
        )
        assert np.isnan(
            gap_values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.FRACTION_AT_DISTANCE,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
        )
        assert np.isnan(
            gap_values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.MEAN_FRACTION,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
        )
        assert (
            gap_values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
            == 0.0
        )
        assert np.isfinite(
            values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.MEAN_FRACTION,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
        )
        assert np.isfinite(
            values_by_feature[
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                    bin_index=bin_index,
                    bin_count=4,
                )
            ]
        )


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


def test_numba_propagation_result_matches_centrosome_distances():
    image = np.arange(64, dtype=np.float64).reshape((8, 8)) / 64.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2, 2] = 1
    labels[2, 5] = 2
    labels[6, 4] = 3
    mask = np.ones((8, 8), dtype=bool)
    mask[4, 1:6] = False

    reference = secondary_propagation_backend(
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    ).propagate_result(image, labels, mask, 1)
    accelerated = secondary_propagation_backend(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).propagate_result(image, labels, mask, 1)

    np.testing.assert_array_equal(accelerated.labels, reference.labels)
    np.testing.assert_allclose(accelerated.distances, reference.distances)


def test_numba_self_centered_radial_distribution_matches_native_reference():
    image = np.arange(25, dtype=np.float32).reshape((5, 5))
    labels = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    native = NativeNumpyRadialDistributionBackendStrategy().measure_self_centered(
        image,
        labels,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )
    accelerated = NumbaNumpyRadialDistributionBackendStrategy().measure_self_centered(
        image,
        labels,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )

    np.testing.assert_allclose(
        accelerated.fraction_at_distance,
        native.fraction_at_distance,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        accelerated.mean_pixel_fraction,
        native.mean_pixel_fraction,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        accelerated.radial_cv_by_bin,
        native.radial_cv_by_bin,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        accelerated.object_has_pixels,
        native.object_has_pixels,
    )
