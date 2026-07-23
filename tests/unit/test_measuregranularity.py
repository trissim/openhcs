import numpy as np

from openhcs.core.measurement_row_materialization import (
    MeasurementProjectedColumnarRows,
)
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.processing.backends.cellprofiler.granularity import (
    GRANULARITY_IMAGE_SERIES_CACHE,
    GRANULARITY_SPECTRUM_LENGTH,
    GranularityImageSeriesRequest,
    GranularitySpectrumDescriptor,
    GranularitySpectrumDescriptorDeclaration,
    MeasureGranularityModule,
    NativeGranularityReconstructionBackendStrategy,
    NumbaGranularityReconstructionBackendStrategy,
    ObjectGranularityMeasurementRows,
    OpenCVGranularityReconstructionBackendStrategy,
    background_corrected_pixels,
    granularity_grey_erosion,
    granularity_reconstruction_backend,
    granularity_reconstruction_series,
    measure_granularity_objects,
)
from openhcs.core.config import DtypeConfig


def test_measure_granularity_declares_one_indexed_spectrum_feature_authority():
    feature = MeasureGranularityModule.MeasurementFeature.SPECTRUM
    descriptor = GranularitySpectrumDescriptor(GRANULARITY_SPECTRUM_LENGTH)

    assert tuple(MeasureGranularityModule.MeasurementFeature) == (feature,)
    assert MeasureGranularityModule.numbered_measurement_feature_prefix_aliases == {}
    assert MeasureGranularityModule.source_qualified_measurement_feature_types() == (
        MeasureGranularityModule.MeasurementFeature,
    )
    assert feature.indexed_descriptor_declarations() == (
        GranularitySpectrumDescriptorDeclaration,
    )
    assert (
        GranularitySpectrumDescriptorDeclaration.from_measurement_row_field_name("gs16")
        == descriptor
    )
    assert (
        GranularitySpectrumDescriptorDeclaration.from_feature_name("Granularity_16")
        == descriptor
    )
    assert (
        GranularitySpectrumDescriptorDeclaration.source_qualified_feature_name(
            descriptor,
            source_image_name="BF_image",
        )
        == "Granularity_16_BF_image"
    )


def test_measure_granularity_projects_exact_image_feature_identities_at_producer():
    projected = MeasureGranularityModule.prepare_measurement_record_rows(
        MeasurementProjectedColumnarRows(
            {
                "slice_index": (0,),
                "gs1": (1.25,),
                "gs16": (16.25,),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("gs1", float),
                FieldSpec("gs16", float),
            ),
        ),
        source_image_name="BF_image",
    )

    assert tuple(projected.columns) == (
        "slice_index",
        "Granularity_1_BF_image",
        "Granularity_16_BF_image",
    )
    assert projected.column_values("Granularity_1_BF_image") == (1.25,)
    assert projected.column_values("Granularity_16_BF_image") == (16.25,)


def test_measure_granularity_projects_exact_object_feature_identities_at_producer():
    rows = ObjectGranularityMeasurementRows(
        np.asarray((3,), dtype=np.int32),
        np.arange(1.0, 17.0, dtype=np.float64).reshape(1, 16),
    )

    projected = MeasureGranularityModule.prepare_measurement_record_rows(
        rows,
        source_image_name="BF_image",
    )

    assert "gs1" not in projected.columns
    assert "gs16" not in projected.columns
    np.testing.assert_array_equal(
        projected.column_values("Granularity_1_BF_image"),
        np.asarray((1.0,)),
    )
    np.testing.assert_array_equal(
        projected.column_values("Granularity_16_BF_image"),
        np.asarray((16.0,)),
    )


def test_object_granularity_rows_preserve_exact_zero_row_schema():
    rows = ObjectGranularityMeasurementRows(
        np.empty(0, dtype=np.int32),
        np.empty((0, GRANULARITY_SPECTRUM_LENGTH), dtype=np.float64),
    )

    assert tuple(field.name for field in rows.fields) == tuple(rows.columns)
    assert tuple(field.dtype for field in rows.fields[:2]) == (int, int)
    assert all(field.dtype is float for field in rows.fields[2:])
    assert rows.row_count() == 0


def test_measure_granularity_objects_preserves_sparse_label_ids():
    image = np.ones((5, 5), dtype=np.float32)
    labels = np.array(
        [
            [1, 1, 0, 3, 3],
            [1, 1, 0, 3, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _result, measurements = measure_granularity_objects(
        image,
        labels,
        subsample_size=1.0,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=1,
        dtype_config=DtypeConfig(),
    )

    assert [measurement.object_id for measurement in measurements] == [1, 3]


def test_granularity_series_cache_reuses_equal_image_values():
    image = np.arange(36, dtype=np.float64).reshape(6, 6)
    image_copy = image.copy()
    GRANULARITY_IMAGE_SERIES_CACHE.clear()

    first = GranularityImageSeriesRequest(
        image=image,
        subsample_size=1.0,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=2,
        profile_function="test",
    ).series()
    second = GranularityImageSeriesRequest(
        image=image_copy,
        subsample_size=1.0,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=2,
        profile_function="test",
    ).series()

    assert second is first
    assert len(GRANULARITY_IMAGE_SERIES_CACHE) == 1


def test_measure_granularity_objects_uses_order_one_coordinate_sampling_after_subsampling():
    import scipy.ndimage

    image = np.arange(25, dtype=np.float64).reshape(5, 5) / 25.0
    labels = np.array(
        [
            [1, 1, 0, 0, 3],
            [1, 0, 0, 3, 3],
            [0, 0, 3, 3, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    _result, measurements = measure_granularity_objects(
        image,
        labels,
        subsample_size=0.8,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=1,
        dtype_config=DtypeConfig(),
    )

    series = GranularityImageSeriesRequest(
        image=image,
        subsample_size=0.8,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=1,
        profile_function="test",
    ).series()
    object_ids = np.array([measurement.object_id for measurement in measurements])
    current_means = scipy.ndimage.mean(image, labels, object_ids)
    start_means = np.maximum(current_means, np.finfo(float).eps)
    rec = series.reconstructions[0]
    row_scale = float(series.new_shape[0] - 1) / float(labels.shape[0] - 1)
    col_scale = float(series.new_shape[1] - 1) / float(labels.shape[1] - 1)
    ri, rj = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]].astype(np.float64)
    ri *= row_scale
    rj *= col_scale
    rec_full = scipy.ndimage.map_coordinates(rec, (ri, rj), order=1)
    new_means = scipy.ndimage.mean(rec_full, labels, object_ids)
    expected = (current_means - new_means) * 100 / start_means
    actual = np.array([measurement.gs1 for measurement in measurements])

    np.testing.assert_allclose(actual, expected)


def test_background_corrected_pixels_match_reference_operations():
    import scipy.ndimage
    import skimage.morphology

    image = np.arange(99, dtype=np.float64).reshape(9, 11) / 99.0
    pixels, _shape = background_corrected_pixels(
        image,
        subsample_size=1.0,
        background_subsample_size=0.5,
        element_radius=1,
    )

    back_shape = np.asarray(image.shape) * 0.5
    bi, bj = np.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float) / 0.5
    back_pixels = scipy.ndimage.map_coordinates(image, (bi, bj), order=1)
    footprint = skimage.morphology.disk(1, dtype=bool)
    back_pixels = skimage.morphology.erosion(back_pixels, footprint=footprint)
    back_pixels = skimage.morphology.dilation(back_pixels, footprint=footprint)
    ui, uj = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
    ui *= float(back_shape[0] - 1) / float(image.shape[0] - 1)
    uj *= float(back_shape[1] - 1) / float(image.shape[1] - 1)
    expected = image - scipy.ndimage.map_coordinates(back_pixels, (ui, uj), order=1)
    expected[expected < 0] = 0

    np.testing.assert_allclose(pixels, expected)


def test_granularity_reconstruction_default_backend_is_numba():
    assert isinstance(
        granularity_reconstruction_backend(),
        NumbaGranularityReconstructionBackendStrategy,
    )


def test_numba_granularity_reconstruction_matches_native_radius_one():
    import skimage.morphology

    rng = np.random.default_rng(22)
    pixels = rng.random((40, 41), dtype=np.float32)
    footprint = skimage.morphology.disk(1, dtype=np.uint8)
    seed = granularity_grey_erosion(pixels, footprint)

    native = NativeGranularityReconstructionBackendStrategy().reconstruct_radius_one(
        seed,
        pixels,
    )
    accelerated = (
        NumbaGranularityReconstructionBackendStrategy().reconstruct_radius_one(
            seed,
            pixels,
        )
    )

    np.testing.assert_array_equal(accelerated, native)


def test_opencv_granularity_reconstruction_matches_native_radius_one():
    import skimage.morphology

    rng = np.random.default_rng(24)
    pixels = rng.random((40, 41), dtype=np.float32)
    footprint = skimage.morphology.disk(1, dtype=np.uint8)
    seed = granularity_grey_erosion(pixels, footprint)

    native = NativeGranularityReconstructionBackendStrategy().reconstruct_radius_one(
        seed,
        pixels,
    )
    accelerated = (
        OpenCVGranularityReconstructionBackendStrategy().reconstruct_radius_one(
            seed,
            pixels,
        )
    )

    np.testing.assert_array_equal(accelerated, native)


def test_numba_granularity_reconstruction_series_matches_reference():
    import skimage.morphology

    rng = np.random.default_rng(23)
    pixels = rng.random((35, 37), dtype=np.float32)
    footprint = skimage.morphology.disk(1, dtype=bool)
    erosion_footprint = skimage.morphology.disk(1, dtype=np.uint8)
    expected = []
    ero = pixels.copy()
    for _index in range(3):
        ero = granularity_grey_erosion(ero, erosion_footprint)
        expected.append(
            skimage.morphology.reconstruction(
                ero,
                pixels,
                footprint=footprint,
            )
        )

    actual = granularity_reconstruction_series(pixels, 3)

    for actual_image, expected_image in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_image, expected_image)


def test_measure_granularity_objects_matches_reference_operations():
    import scipy.ndimage
    import skimage.morphology

    image = np.array(
        [
            [0.0, 0.1, 0.4, 0.2, 0.0],
            [0.2, 0.7, 0.9, 0.5, 0.1],
            [0.1, 0.6, 0.8, 0.4, 0.2],
            [0.0, 0.2, 0.5, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    _result, actual_measurements = measure_granularity_objects(
        image,
        labels,
        subsample_size=1.0,
        background_subsample_size=1.0,
        element_radius=1,
        spectrum_length=2,
        dtype_config=DtypeConfig(),
    )

    footprint = skimage.morphology.disk(1, dtype=bool)
    pixels = skimage.morphology.erosion(image, footprint=footprint)
    pixels = skimage.morphology.dilation(pixels, footprint=footprint)
    pixels = image - pixels
    pixels[pixels < 0] = 0
    object_ids = np.array([1, 2], dtype=np.int32)
    current_means = scipy.ndimage.mean(image, labels, object_ids)
    start_means = np.maximum(current_means, np.finfo(float).eps)
    ero = pixels.copy()
    expected = []
    for _index in range(2):
        previous_means = current_means.copy()
        ero = skimage.morphology.erosion(ero, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        current_means = scipy.ndimage.mean(rec, labels, object_ids)
        expected.append((previous_means - current_means) * 100 / start_means)

    actual = np.array(
        [[measurement.gs1, measurement.gs2] for measurement in actual_measurements]
    )
    np.testing.assert_allclose(actual, np.asarray(expected).T)


def test_measure_granularity_objects_preserves_negative_first_scale():
    rng = np.random.default_rng(51)
    image = rng.random((40, 40))
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:15, 5:15] = 1
    labels[20:35, 20:35] = 2

    _result, measurements = measure_granularity_objects(
        image,
        labels,
        subsample_size=0.25,
        background_subsample_size=0.25,
        element_radius=10,
        spectrum_length=2,
        dtype_config=DtypeConfig(),
    )

    assert measurements[0].gs1 < 0.0
    assert measurements[0].gs2 >= 0.0
