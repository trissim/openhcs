import numpy as np
import pytest

import openhcs.processing.backends.cellprofiler.intensity_distribution as mid
from openhcs.core.config import DtypeConfig
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode_from_callable,
)
from openhcs.core.measurement_row_materialization import columnar_row_values
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain, ObjectLabelDomainScope
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)
from openhcs.core.runtime_tabular_values import MeasurementObjectRowIdentity
from openhcs.interop.cellprofiler.measurement_dialect import (
    cellprofiler_projected_measurement_feature_name,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    MeasureObjectIntensityDistributionModule,
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
    IntensityZernikeMeasurementFeature,
    IntensityZernikeMeasurementRowsRequest,
    ObjectIntensityZernikeMeasurementColumnarRows,
    ObjectZernikeDescriptorFeature,
    indexed_object_intensity_zernike_feature_name,
)

SOURCE_IMAGE_NAME = "BF_image"


def source_image(image: np.ndarray):
    return ImagePayloadMetadata(source_image_names=(SOURCE_IMAGE_NAME,)).attach_to(
        image
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

    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_count=4),
    )

    _result, measurements = mid.measure_object_intensity_distribution(
        source_image(image),
        label_payload,
        bin_count=4,
        dtype_config=DtypeConfig(),
    )

    assert MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value not in measurements.columns

    gap_label_measurements = [
        measurement for measurement in measurements if measurement.object_label == 2
    ]
    label_three_measurements = [
        measurement for measurement in measurements if measurement.object_label == 3
    ]
    trailing_label_measurements = [
        measurement for measurement in measurements if measurement.object_label == 4
    ]

    assert len(gap_label_measurements) == 12
    assert len(label_three_measurements) == 12
    assert len(trailing_label_measurements) == 12
    rows = tuple(
        zip(
            columnar_row_values(measurements, "object_label"),
            columnar_row_values(measurements, "feature_name"),
            columnar_row_values(measurements, "bin_index"),
            columnar_row_values(measurements, "result_value"),
            strict=True,
        )
    )
    gap_values_by_feature_and_bin = {
        (feature_name, bin_index): value
        for object_label, feature_name, bin_index, value in rows
        if object_label == 2
    }
    values_by_feature_and_bin = {
        (feature_name, bin_index): value
        for object_label, feature_name, bin_index, value in rows
        if object_label == 3
    }
    trailing_values_by_feature_and_bin = {
        (feature_name, bin_index): value
        for object_label, feature_name, bin_index, value in rows
        if object_label == 4
    }
    fraction_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.FRACTION_AT_DISTANCE.source_qualified_name(
        source_image_name=SOURCE_IMAGE_NAME
    )
    mean_fraction_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.MEAN_FRACTION.source_qualified_name(
        source_image_name=SOURCE_IMAGE_NAME
    )
    radial_cv_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.RADIAL_CV.source_qualified_name(
        source_image_name=SOURCE_IMAGE_NAME
    )
    for bin_index in range(1, 5):
        assert np.isfinite(values_by_feature_and_bin[(fraction_feature, bin_index)])
        assert np.isnan(gap_values_by_feature_and_bin[(fraction_feature, bin_index)])
        assert np.isnan(
            gap_values_by_feature_and_bin[(mean_fraction_feature, bin_index)]
        )
        assert gap_values_by_feature_and_bin[(radial_cv_feature, bin_index)] == 0.0
        assert np.isnan(
            trailing_values_by_feature_and_bin[(radial_cv_feature, bin_index)]
        )
        assert np.isfinite(
            values_by_feature_and_bin[(mean_fraction_feature, bin_index)]
        )
        assert np.isfinite(values_by_feature_and_bin[(radial_cv_feature, bin_index)])


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
        source_image_name=SOURCE_IMAGE_NAME,
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
            MeasureObjectIntensityDistributionModule.MeasurementFeature.RADIAL_CV.source_qualified_name(
                source_image_name=SOURCE_IMAGE_NAME,
            )
        ]
        == 0.0
    )


def test_radial_distribution_rows_own_native_feature_identity_and_axes():
    rows = ObjectIntensityDistributionMeasurementColumnarRows(
        radial_arrays=RadialDistributionArrays(
            fraction_at_distance=np.array([[0.25]], dtype=np.float64),
            mean_pixel_fraction=np.array([[1.0]], dtype=np.float64),
            radial_cv_by_bin=np.array([[0.0]], dtype=np.float64),
            object_has_pixels=np.array([True]),
            n_bins=1,
        ),
        object_ids=(1,),
        source_image_name=SOURCE_IMAGE_NAME,
        bin_count=4,
    )

    feature_name = columnar_row_values(rows, "feature_name")[0]
    assert rows.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert feature_name == "RadialDistribution_FracAtD_BF_image"
    assert cellprofiler_projected_measurement_feature_name(
        feature_name,
        (("bin_index", 1), ("bin_count", 4)),
    ) == ("RadialDistribution_FracAtD_BF_image_1of4")
    assert set(columnar_row_values(rows, "source_image_name")) == {SOURCE_IMAGE_NAME}
    assert set(columnar_row_values(rows, "bin_index")) == {1}
    assert set(columnar_row_values(rows, "bin_count")) == {4}


def test_intensity_distribution_module_owns_canonical_source_projection():
    assert (
        MeasureObjectIntensityDistributionModule.source_qualified_measurement_category()
        == "RadialDistribution"
    )
    radial_name = MeasureObjectIntensityDistributionModule.MeasurementFeature.FRACTION_AT_DISTANCE.source_qualified_name(
        source_image_name=SOURCE_IMAGE_NAME
    )
    zernike_name = MeasureObjectIntensityDistributionModule.source_qualified_feature_name(
        IntensityZernikeMeasurementFeature.ZERNIKE_MAGNITUDE.measurement_row_field_name,
        SOURCE_IMAGE_NAME,
    )

    assert radial_name == "RadialDistribution_FracAtD_BF_image"
    assert zernike_name == "RadialDistribution_ZernikeMagnitude_BF_image"
    assert (
        MeasureObjectIntensityDistributionModule.source_qualified_feature_name(
            radial_name,
            SOURCE_IMAGE_NAME,
        )
        == radial_name
    )
    assert (
        MeasureObjectIntensityDistributionModule.source_qualified_feature_name(
            zernike_name,
            SOURCE_IMAGE_NAME,
        )
        == zernike_name
    )


def test_intensity_zernike_rows_own_native_feature_identity_and_axes():
    rows = ObjectIntensityZernikeMeasurementColumnarRows(
        object_ids=(1,),
        zernike_indexes=((2, 0),),
        magnitudes=np.array([[0.5]], dtype=np.float64),
        phases=np.array([[0.0]], dtype=np.float64),
        include_phase=False,
        source_image_name=SOURCE_IMAGE_NAME,
    )

    assert tuple(columnar_row_values(rows, "feature_name")) == (
        "RadialDistribution_ZernikeMagnitude_BF_image_2_0",
    )
    assert tuple(columnar_row_values(rows, "source_image_name")) == (SOURCE_IMAGE_NAME,)
    assert tuple(columnar_row_values(rows, "n")) == (2,)
    assert tuple(columnar_row_values(rows, "m")) == (0,)
    native_feature_name = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE,
        source_image_name=SOURCE_IMAGE_NAME,
        degree=2,
        repetition=0,
    )
    assert native_feature_name == "RadialDistribution_ZernikeMagnitude_BF_image_2_0"
    assert (
        cellprofiler_projected_measurement_feature_name(
            native_feature_name,
            (("n", 2), ("m", 0)),
        )
        == native_feature_name
    )


def test_radial_and_zernike_rows_preserve_exact_zero_row_schemas():
    radial_rows = ObjectIntensityDistributionMeasurementColumnarRows.empty(
        source_image_name=SOURCE_IMAGE_NAME,
        slice_index=0,
    )
    zernike_rows = ObjectIntensityZernikeMeasurementColumnarRows.empty(
        source_image_name=SOURCE_IMAGE_NAME,
        slice_index=0,
    )

    assert tuple(field.name for field in radial_rows.fields) == tuple(
        radial_rows.columns
    )
    assert tuple(field.dtype for field in radial_rows.fields) == (
        int,
        str,
        str,
        int,
        int,
        float,
        int,
    )
    assert tuple(field.name for field in zernike_rows.fields) == tuple(
        zernike_rows.columns
    )
    assert tuple(field.dtype for field in zernike_rows.fields) == (
        int,
        str,
        str,
        int,
        int,
        float,
        int,
    )
    assert radial_rows.row_count() == 0
    assert zernike_rows.row_count() == 0


def test_intensity_distribution_object_domain_uses_declared_payload_domain():
    labels = np.array(
        [
            [1, 0, 3],
            [1, 0, 3],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError, match="explicit object-ID domain"):
        intensity_distribution_object_domain(labels)
    assert intensity_distribution_object_domain(
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_count=4),
        )
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


def test_measure_object_intensity_distribution_declares_slice_aligned_labels():
    assert (
        object_label_input_execution_mode_from_callable(
            measure_object_intensity_distribution
        )
        is ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )


def test_measure_object_intensity_distribution_preserves_runtime_slice_axis():
    image = np.full((4, 4), 2.0, dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:3, 1:3] = 1

    _result, measurements = measure_object_intensity_distribution(
        source_image(image),
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_count=1),
        ),
        bin_count=2,
        wants_zernikes="None",
        slice_index=1,
    )

    assert set(columnar_row_values(measurements, "slice_index")) == {1}


def test_measure_object_intensity_distribution_rejects_unprojected_label_stack():
    image = np.ones((2, 4, 4), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 0, 0] = 3
    labels[1, 1:3, 1:3] = 1

    with pytest.raises(ValueError, match="already projected to one 2-D plane"):
        measure_object_intensity_distribution(
            source_image(image),
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
            bin_count=2,
            wants_zernikes="None",
        )


def test_measure_object_intensity_distribution_rejects_repeated_plane_domains():
    image = np.ones((4, 4), dtype=np.float32)
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[:, 1:3, 1:3] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match="already projected to one 2-D plane"):
        measure_object_intensity_distribution(
            source_image(image),
            payload,
            bin_count=2,
            wants_zernikes="None",
        )


def test_intensity_zernike_uses_compact_rows_for_noncontiguous_domains():
    image = np.ones((5, 5), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 5

    phase_feature = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
        source_image_name=SOURCE_IMAGE_NAME,
        degree=0,
        repetition=0,
    )

    rows = IntensityZernikeMeasurementRowsRequest(
        image=image,
        labels=labels,
        max_order=0,
        include_phase=True,
        source_image_name=SOURCE_IMAGE_NAME,
        object_ids=(5,),
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
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


def test_intensity_zernike_phase_export_preserves_undefined_phase():
    phase_feature = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
        source_image_name=SOURCE_IMAGE_NAME,
        degree=0,
        repetition=0,
    )
    rows = ObjectIntensityZernikeMeasurementColumnarRows(
        object_ids=(1,),
        zernike_indexes=((0, 0),),
        magnitudes=np.array([[np.nan]], dtype=np.float64),
        phases=np.array([[np.nan]], dtype=np.float64),
        include_phase=True,
        source_image_name=SOURCE_IMAGE_NAME,
    )

    values_by_feature = {
        feature: value
        for feature, value in zip(
            columnar_row_values(rows, "feature_name"),
            columnar_row_values(rows, "result_value"),
            strict=True,
        )
    }

    assert np.isnan(values_by_feature[phase_feature])


def test_intensity_zernike_phase_export_zeroes_undefined_phase_within_extent():
    phase_feature = indexed_object_intensity_zernike_feature_name(
        ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
        source_image_name=SOURCE_IMAGE_NAME,
        degree=0,
        repetition=0,
    )
    rows = ObjectIntensityZernikeMeasurementColumnarRows(
        object_ids=(1, 2),
        zernike_indexes=((0, 0),),
        magnitudes=np.array([[np.nan], [np.nan]], dtype=np.float64),
        phases=np.array([[np.nan], [0.0]], dtype=np.float64),
        include_phase=True,
        source_image_name=SOURCE_IMAGE_NAME,
        phase_zero_extent=1,
    )

    values_by_object = {
        object_label: value
        for object_label, feature, value in zip(
            columnar_row_values(rows, "object_label"),
            columnar_row_values(rows, "feature_name"),
            columnar_row_values(rows, "result_value"),
            strict=True,
        )
        if feature == phase_feature
    }

    assert values_by_object[1] == 0.0
    assert np.isnan(values_by_object[2])


def test_numba_propagation_result_matches_native_reference_values():
    image = np.arange(64, dtype=np.float64).reshape((8, 8)) / 64.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2, 2] = 1
    labels[2, 5] = 2
    labels[6, 4] = 3
    mask = np.ones((8, 8), dtype=bool)
    mask[4, 1:6] = False

    accelerated = secondary_propagation_backend(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).propagate_result(image, labels, mask, 1)
    expected_labels = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 0, 0, 0, 0, 0, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
        dtype=np.int32,
    )
    expected_distances = np.array(
        [
            [
                3.5446316437742986,
                3.1478426279923740,
                2.7551993223490370,
                2.9730769398448230,
                3.1478426279923740,
                2.7551993223490370,
                2.9730769398448230,
                3.3094569581569550,
            ],
            [
                2.8767489184090140,
                1.8978426279923740,
                1.5051993223490370,
                1.7230769398448231,
                1.8978426279923740,
                1.5051993223490370,
                1.7230769398448231,
                2.7274618573440854,
            ],
            [
                2.0142242070027950,
                1.0098392895035329,
                0.0,
                1.0098392895035329,
                1.0098392895035329,
                0.0,
                1.0098392895035329,
                2.0142242070027950,
            ],
            [
                2.7274618573440854,
                1.7230769398448231,
                1.5051993223490370,
                1.8978426279923740,
                1.7230769398448231,
                1.5051993223490370,
                1.8978426279923740,
                2.8767489184090140,
            ],
            [
                3.4733559354623790,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                3.4030419503414110,
                3.7647522568978546,
            ],
            [
                4.8964274974160790,
                3.9175212069994396,
                2.9076819174959070,
                1.8978426279923740,
                1.5051993223490370,
                1.7230769398448231,
                2.7329162293483558,
                3.7373011468476180,
            ],
            [
                4.0339027860098610,
                3.0295178685105990,
                2.0196785790070657,
                1.0098392895035329,
                0.0,
                1.0098392895035329,
                2.0196785790070657,
                3.0240634965063280,
            ],
            [
                4.6034256353538440,
                3.5990407178545816,
                2.5892014283510490,
                1.5793621388475159,
                1.2500000000000000,
                1.6712907857775678,
                2.6811300752811010,
                3.6664675947889904,
            ],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(accelerated.labels, expected_labels)
    np.testing.assert_allclose(accelerated.distances, expected_distances)


def test_numba_zero_image_propagation_matches_uniform_image_path():
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[1, 1] = 2
    labels[1, 7] = 1
    labels[7, 4] = 3
    mask = np.ones(labels.shape, dtype=bool)
    mask[4, 1:7] = False
    backend = secondary_propagation_backend(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    reference = backend.propagate_result(
        np.zeros(labels.shape, dtype=np.float64),
        labels,
        mask,
        1,
    )
    accelerated = backend.propagate_zero_image_result(labels, mask, 1)

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


def test_numba_self_centered_radial_distribution_preserves_native_zero_intensity_edges():
    image = np.zeros((5, 5), dtype=np.float32)
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
    native_backend = NativeNumpyRadialDistributionBackendStrategy()
    accelerated_backend = NumbaNumpyRadialDistributionBackendStrategy()
    geometry = native_backend.label_geometry(labels)

    native = native_backend.measure_self_centered_with_geometry(
        image,
        labels,
        geometry,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )
    accelerated = accelerated_backend.measure_batch_self_centered_with_geometry(
        (image,),
        labels,
        geometry,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
    )[0]

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
