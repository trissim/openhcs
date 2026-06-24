import numpy as np

from benchmark.cellprofiler_library.functions import measureobjectintensitydistribution as mid
from openhcs.core.config import DtypeConfig
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.measurement_row_materialization import columnar_row_values
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ObjectIntensityDistributionMeasurementFeature,
    ObjectZernikeDescriptorFeature,
    indexed_object_intensity_distribution_feature_name,
    indexed_object_intensity_zernike_feature_name,
)
from openhcs.core.runtime_values import ObjectLabelPayload
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    NativeNumpyRadialDistributionBackendStrategy,
    NumbaNumpyRadialDistributionBackendStrategy,
    ObjectIntensityDistributionMeasurementColumnarRows,
    RadialDistributionArrays,
    RadialDistributionMeasureRequest,
    intensity_distribution_object_domain,
    measure_object_intensity_distribution,
    radial_distribution_backend,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    secondary_propagation_backend,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    IntensityZernikeMeasurementRowsRequest,
    ObjectIntensityZernikeMeasurementColumnarRows,
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


def test_radial_cv_export_values_zero_undefined_coefficients():
    rows = ObjectIntensityDistributionMeasurementColumnarRows(
        radial_arrays=RadialDistributionArrays(
            fraction_at_distance=np.array([[1.0]], dtype=np.float64),
            mean_pixel_fraction=np.array([[1.0]], dtype=np.float64),
            radial_cv_by_bin=np.array([[np.nan]], dtype=np.float64),
            object_has_pixels=np.array([True]),
            n_bins=1,
        ),
        object_ids=(1,),
        bin_count=1,
    )

    values_by_feature = {
        feature: value
        for feature, value in zip(
            columnar_row_values(rows, "feature_name"),
            columnar_row_values(rows, "result_value"),
            strict=True,
        )
    }

    assert (
        values_by_feature[
            indexed_object_intensity_distribution_feature_name(
                ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                bin_index=1,
                bin_count=1,
            )
        ]
        == 0.0
    )


def test_intensity_distribution_object_domain_uses_declared_payload_domain():
    labels = np.array(
        [
            [1, 0, 3],
            [1, 0, 3],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    assert intensity_distribution_object_domain(labels) == (1, 2, 3)
    assert intensity_distribution_object_domain(
        ObjectLabelPayload(labels=labels, domain=ObjectLabelDomain(declared_object_count=4))
    ) == (1, 2, 3, 4)


def test_radial_cv_ignores_empty_angular_wedges():
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.ones((3, 3), dtype=np.int32)

    radial_arrays = NativeNumpyRadialDistributionBackendStrategy().measure(
        RadialDistributionMeasureRequest(
            image=image,
            labels=labels,
            d_to_edge=np.zeros(labels.shape, dtype=np.float64),
            d_from_center=np.zeros(labels.shape, dtype=np.float64),
            center_labels=np.ones(labels.shape, dtype=np.int32),
            centers_i=np.array([1.0], dtype=np.float64),
            centers_j=np.array([1.0], dtype=np.float64),
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )
    )

    assert radial_arrays.radial_cv_by_bin[0, 0] == 0.0


def test_explicit_numba_radial_provider_remains_available():
    selected = radial_distribution_backend(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    assert selected.backend_provider is CellProfilerBackendProvider.NUMBA


def test_measure_object_intensity_distribution_declares_full_stack_labels():
    assert (
        object_label_measurement_execution_from_callable(
            measure_object_intensity_distribution
        )
        is ObjectLabelMeasurementExecution.FULL_STACK
    )


def test_measure_object_intensity_distribution_preserves_runtime_slice_axis():
    image = np.stack(
        (
            np.ones((4, 4), dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
        )
    )
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 1:3] = 1

    _result, measurements = measure_object_intensity_distribution(
        image,
        labels,
        bin_count=2,
        wants_zernikes="None",
    )

    assert set(columnar_row_values(measurements, "slice_index")) == {0, 1}


def test_measure_object_intensity_distribution_collapses_repeated_2d_plane_domains():
    image = np.ones((4, 4), dtype=np.float32)
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[:, 1:3, 1:3] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(declared_object_id_domains=((1, 2), (1, 2)),
        scope=ObjectLabelDomainScope.PLANE,
    ))

    _result, measurements = measure_object_intensity_distribution(
        image,
        payload,
        bin_count=2,
        wants_zernikes="None",
    )

    assert set(columnar_row_values(measurements, "slice_index")) == {0}
    assert sum(
        1
        for feature in columnar_row_values(measurements, "feature_name")
        if feature
        == indexed_object_intensity_distribution_feature_name(
            ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
            bin_index=1,
            bin_count=2,
        )
    ) == 2


def test_intensity_zernike_uses_compact_rows_for_noncontiguous_domains():
    image = np.ones((5, 5), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 5

    phase_feature = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
        degree=0,
        repetition=0,
    )

    for provider in (
        CellProfilerBackendProvider.CENTROSOME,
        CellProfilerBackendProvider.LEGACY_FAST,
    ):
        rows = IntensityZernikeMeasurementRowsRequest(
            image=image,
            labels=labels,
            max_order=0,
            include_phase=True,
            object_ids=(5,),
            backend_provider=provider,
        ).rows()
        values = [
            value
            for feature, value in zip(
                columnar_row_values(rows, "feature_name"),
                columnar_row_values(rows, "result_value"),
                strict=True,
            )
            if feature == phase_feature
        ]

        np.testing.assert_allclose(values, [np.pi / 2.0])


def test_intensity_zernike_phase_export_values_zero_undefined_phase():
    phase_feature = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
        degree=0,
        repetition=0,
    )
    rows = ObjectIntensityZernikeMeasurementColumnarRows(
        object_ids=(1,),
        zernike_indexes=((0, 0),),
        magnitudes=np.array([[np.nan]], dtype=np.float64),
        phases=np.array([[np.nan]], dtype=np.float64),
        include_phase=True,
    )

    values_by_feature = {
        feature: value
        for feature, value in zip(
            columnar_row_values(rows, "feature_name"),
            columnar_row_values(rows, "result_value"),
            strict=True,
        )
    }

    assert values_by_feature[phase_feature] == 0.0


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
