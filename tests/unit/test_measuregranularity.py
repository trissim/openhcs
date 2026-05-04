import numpy as np

from benchmark.cellprofiler_library.functions.measuregranularity import (
    _background_corrected_pixels,
    _disk_offsets,
    _gray_dilation_offsets_reflect_numba,
    _gray_erosion_offsets_reflect_numba,
    _mean_by_label_from_resampled_numba,
    _reconstruct_dilation_cross_numba,
    measure_granularity_objects,
)
from openhcs.core.config import DtypeConfig


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


def test_resampled_object_means_match_order_one_coordinate_sampling():
    import scipy.ndimage

    rec = np.arange(16, dtype=np.float64).reshape(4, 4)
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
    object_ids = np.array([1, 3], dtype=np.int32)
    row_scale = float(rec.shape[0] - 1) / float(labels.shape[0] - 1)
    col_scale = float(rec.shape[1] - 1) / float(labels.shape[1] - 1)
    ri, rj = np.mgrid[0:labels.shape[0], 0:labels.shape[1]].astype(float)
    ri *= row_scale
    rj *= col_scale
    rec_full = scipy.ndimage.map_coordinates(rec, (ri, rj), order=1)

    expected = np.array(
        [rec_full[labels == object_id].mean() for object_id in object_ids]
    )
    actual = _mean_by_label_from_resampled_numba(
        rec,
        labels,
        object_ids,
        row_scale,
        col_scale,
    )

    np.testing.assert_allclose(actual, expected)


def test_numba_disk_morphology_matches_skimage_reflect_mode():
    import skimage.morphology

    image = np.array(
        [
            [5.0, 4.0, 3.0, 2.0],
            [6.0, 1.0, 7.0, 8.0],
            [9.0, 2.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    footprint = skimage.morphology.disk(1, dtype=bool)
    offsets = _disk_offsets(1)

    np.testing.assert_allclose(
        _gray_erosion_offsets_reflect_numba(image, offsets),
        skimage.morphology.erosion(image, footprint=footprint),
    )
    np.testing.assert_allclose(
        _gray_dilation_offsets_reflect_numba(image, offsets),
        skimage.morphology.dilation(image, footprint=footprint),
    )


def test_numba_reconstruction_matches_skimage_disk_one_connectivity():
    import skimage.morphology

    mask = np.array(
        [
            [0.2, 0.5, 0.4, 0.1, 0.0],
            [0.3, 0.9, 0.8, 0.3, 0.2],
            [0.1, 0.4, 0.7, 0.6, 0.1],
            [0.0, 0.2, 0.5, 0.4, 0.3],
        ],
        dtype=np.float64,
    )
    seed = skimage.morphology.erosion(
        mask,
        footprint=skimage.morphology.disk(1, dtype=bool),
    )

    expected = skimage.morphology.reconstruction(
        seed,
        mask,
        footprint=skimage.morphology.disk(1, dtype=bool),
    )
    actual = _reconstruct_dilation_cross_numba(seed, mask)

    np.testing.assert_allclose(actual, expected)


def test_background_corrected_pixels_match_reference_operations():
    import scipy.ndimage
    import skimage.morphology

    image = np.arange(80, dtype=np.float64).reshape(8, 10) / 80.0
    pixels, _shape = _background_corrected_pixels(
        image,
        subsample_size=1.0,
        background_subsample_size=0.5,
        element_radius=1,
    )

    back_shape = (np.asarray(image.shape) * 0.5).astype(int)
    bi, bj = np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / 0.5
    back_pixels = scipy.ndimage.map_coordinates(image, (bi, bj), order=1)
    footprint = skimage.morphology.disk(1, dtype=bool)
    back_pixels = skimage.morphology.erosion(back_pixels, footprint=footprint)
    back_pixels = skimage.morphology.dilation(back_pixels, footprint=footprint)
    ui, uj = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(float)
    ui *= float(back_shape[0] - 1) / float(image.shape[0] - 1)
    uj *= float(back_shape[1] - 1) / float(image.shape[1] - 1)
    expected = image - scipy.ndimage.map_coordinates(back_pixels, (ui, uj), order=1)
    expected[expected < 0] = 0

    np.testing.assert_allclose(pixels, expected)


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
        [
            [measurement.gs1, measurement.gs2]
            for measurement in actual_measurements
        ]
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
