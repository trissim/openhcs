import importlib
import sys
import types
import numpy as np
import pytest
import skimage.morphology
import skimage.segmentation

from benchmark.cellprofiler_library import (
    canonical_module_name,
    get_contract,
    get_function,
    list_modules,
)
from benchmark.cellprofiler_library.functions.align import AlignShiftMeasurement, align
from benchmark.cellprofiler_library.functions.closing import closing
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    correct_illumination_apply,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    correct_illumination_calculate,
)
from benchmark.cellprofiler_library.functions.crop import crop
from benchmark.cellprofiler_library.functions.measureimageareaoccupied import (
    measure_image_area_occupied,
)
from openhcs.processing.backends.cellprofiler.texture import (
    ObjectTextureCropBackendStrategy,
)
from benchmark.cellprofiler_library.functions.imagemath import image_math
from benchmark.cellprofiler_library.functions.maskimage import mask_image
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    FillHolesOption,
    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
    _declumping_maxima_geometry,
    _fill_after_declump_requested,
    _fill_before_declump_requested,
    _filter_border_objects,
    _declumping_suppression_footprint,
    _manual_declumping_size,
    identify_primary_objects,
)
import benchmark.cellprofiler_library.functions.identifyprimaryobjects as identifyprimaryobjects_module
from benchmark.cellprofiler_library.functions.measurecolocalization import (
    measure_colocalization,
    measure_colocalization_objects,
    _bisection_costes,
    _costes_first_channel_bin_threshold,
    _divide_costes_measurements,
    _object_colocalization_row,
    _thresholded_colocalization_metrics_numba,
)
from benchmark.cellprofiler_library.functions.opening import opening
from benchmark.cellprofiler_library.functions.overlayobjects import overlay_objects
from benchmark.cellprofiler_library.functions.overlayoutlines import overlay_outlines
from benchmark.cellprofiler_library.functions.relateobjects import (
    DistanceMethod,
    relate_objects,
)
from benchmark.cellprofiler_library.functions.resize import resize, resize_volumetric
from benchmark.cellprofiler_library.functions.enhanceedges import enhance_edges
from benchmark.cellprofiler_library.functions.smooth import smooth
from benchmark.cellprofiler_library.functions.thresholding import (
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET,
    CELLPROFILER_LOG_MULTI_OTSU_BINS,
    _threshold_histogram_bin_width,
    _threshold_multiotsu,
    cellprofiler_get_global_threshold,
    cellprofiler_threshold_diagnostics,
    cellprofiler_threshold,
)
from benchmark.cellprofiler_library.functions.threshold import threshold
from benchmark.cellprofiler_library.functions.unmixcolors import unmix_colors
from benchmark.cellprofiler_library.semantic_defaults import (
    CellProfilerSemanticDefaultContract,
)
from benchmark.cellprofiler_semantics.crop import CropShape, RemovalMethod
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MaskedImagePayload,
    image_payload_with_context,
)
from openhcs.core.runtime_values import image_payload_data, image_payload_mask
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.morphology import CellProfilerDeclumpMethod


def test_absorbed_registry_resolves_every_declared_function():
    unresolved_modules = tuple(
        module_name
        for module_name in list_modules()
        if get_contract(module_name) is not None and get_function(module_name) is None
    )

    assert unresolved_modules == ()


def test_active_absorbed_cellprofiler_functions_import_cleanly():
    function_names = (
        "ConvertObjectsToImage",
        "GrayToColor",
        "Opening",
        "OverlayOutlines",
    )

    loaded_functions = {name: get_function(name) for name in function_names}

    assert all(func is not None for func in loaded_functions.values())


def test_absorbed_semantic_defaults_match_vendored_cellprofiler_source() -> None:
    contracts = CellProfilerSemanticDefaultContract.registered_contracts()

    assert {contract.module_name for contract in contracts} >= {
        "MedianFilter",
        "Watershed",
    }
    for contract in contracts:
        contract.validate()


def test_absorbed_watershed_accepts_grayscale_volumes() -> None:
    from benchmark.cellprofiler_library.functions.watershed import watershed

    image = np.zeros((5, 12, 12), dtype=np.float32)
    image[:, 2:5, 2:5] = 1.0
    image[:, 7:10, 7:10] = 1.0
    raw_watershed = watershed
    while hasattr(raw_watershed, "__wrapped__"):
        raw_watershed = raw_watershed.__wrapped__

    output, stats, labels = raw_watershed(image, footprint=3)

    assert output.shape == image.shape
    assert labels.shape == image.shape
    assert labels.dtype == np.int32
    assert stats.object_count >= 1


def test_watershed_marker_mode_preserves_marker_label_identity() -> None:
    from benchmark.cellprofiler_library.functions.watershed import watershed

    image = np.zeros((12, 12), dtype=np.float32)
    image[1:5, 1:5] = 1.0
    image[7:11, 7:11] = 1.0
    markers = np.zeros_like(image, dtype=np.int32)
    markers[2, 2] = 3
    markers[8, 8] = 9
    raw_watershed = watershed
    while hasattr(raw_watershed, "__wrapped__"):
        raw_watershed = raw_watershed.__wrapped__

    _output, _stats, labels = raw_watershed(
        image,
        markers=markers,
        watershed_method="markers",
        declump_method="shape",
        use_advanced_settings=False,
    )

    assert set(np.unique(labels)) == {0, 1, 2}


def test_resize_objects_preserves_leading_axes_for_volume_stacks() -> None:
    from benchmark.cellprofiler_library.functions.resizeobjects import resize_objects

    image = np.zeros((2, 3, 4, 5), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[:, :, 1:3, 1:3] = 1
    raw_resize_objects = resize_objects
    while hasattr(raw_resize_objects, "__wrapped__"):
        raw_resize_objects = raw_resize_objects.__wrapped__

    _output, stats, relationship, resized = raw_resize_objects(
        image,
        labels,
        method="factor",
        factor_x=2.0,
        factor_y=2.0,
        factor_z=1.0,
    )

    assert resized.shape == (2, 3, 8, 10)
    assert relationship.parent_ids == (1,)
    assert relationship.child_ids == (1,)
    assert stats.original_height == 4
    assert stats.original_width == 5
    assert stats.new_height == 8
    assert stats.new_width == 10


def test_resize_preserves_resized_image_mask() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    raw_resize = resize
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__

    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
    )

    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 2)
    np.testing.assert_array_equal(resized.mask, np.array([[True, False], [False, True]]))


def test_resize_volumetric_preserves_resized_image_mask() -> None:
    image = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :2, :2] = True

    raw_resize = resize_volumetric
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__

    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
        resizing_factor_z=1.0,
    )

    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(resized.mask, mask[:, ::2, ::2])


def test_erode_objects_preserves_leading_axes_for_volume_stacks() -> None:
    from benchmark.cellprofiler_library.functions.erodeobjects import erode_objects

    image = np.zeros((2, 3, 7, 7), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[:, :, 2:5, 2:5] = 1
    raw_erode_objects = erode_objects
    while hasattr(raw_erode_objects, "__wrapped__"):
        raw_erode_objects = raw_erode_objects.__wrapped__

    _output, stats, relationship, eroded = raw_erode_objects(
        image,
        labels,
        structuring_element="ball",
        size=1,
    )

    assert eroded.shape == labels.shape
    assert relationship.parent_ids == (1,)
    assert relationship.child_ids == (1,)
    assert stats.input_object_count == 1
    assert stats.output_object_count == 1


def test_erode_image_preserves_leading_axes_for_volume_stacks() -> None:
    from benchmark.cellprofiler_library.functions.erodeimage import erode_image

    image = np.zeros((2, 3, 7, 7), dtype=np.float32)
    image[:, :, 2:5, 2:5] = 1
    raw_erode_image = erode_image
    while hasattr(raw_erode_image, "__wrapped__"):
        raw_erode_image = raw_erode_image.__wrapped__

    eroded = raw_erode_image(
        image,
        structuring_element="ball",
        size=1,
    )

    assert eroded.shape == image.shape
    assert np.count_nonzero(eroded) < np.count_nonzero(image)


def test_convert_objects_to_image_accepts_volume_label_stacks() -> None:
    from benchmark.cellprofiler_library.functions.convertobjectstoimage import (
        convert_objects_to_image,
    )

    labels = np.zeros((2, 3, 4, 5), dtype=np.int32)
    labels[:, :, 1:3, 1:3] = 1
    raw_convert_objects_to_image = convert_objects_to_image
    while hasattr(raw_convert_objects_to_image, "__wrapped__"):
        raw_convert_objects_to_image = raw_convert_objects_to_image.__wrapped__

    converted = raw_convert_objects_to_image(
        np.zeros_like(labels, dtype=np.float32),
        labels,
        image_mode="color",
    )

    assert converted.shape == labels.shape
    assert converted.dtype == np.float32
    assert np.all(converted[labels == 0] == 0.0)
    assert np.all(converted[labels == 1] > 0.0)


def test_convert_objects_to_image_uint16_preserves_integer_object_ids() -> None:
    from benchmark.cellprofiler_library.functions.convertobjectstoimage import (
        convert_objects_to_image,
    )

    labels = np.array([[0, 1, 3]], dtype=np.int32)
    raw_convert_objects_to_image = convert_objects_to_image
    while hasattr(raw_convert_objects_to_image, "__wrapped__"):
        raw_convert_objects_to_image = raw_convert_objects_to_image.__wrapped__

    converted = raw_convert_objects_to_image(
        np.zeros_like(labels, dtype=np.float32),
        labels,
        image_mode="uint16",
    )

    assert converted.dtype == np.int32
    np.testing.assert_array_equal(converted, labels)


def test_overlay_objects_aligns_labels_to_image_geometry() -> None:
    image = np.zeros((8, 10), dtype=np.float32)
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[1:3, 2:4] = 1
    raw_overlay_objects = overlay_objects
    while hasattr(raw_overlay_objects, "__wrapped__"):
        raw_overlay_objects = raw_overlay_objects.__wrapped__

    overlay = raw_overlay_objects(image, labels)

    assert overlay.shape == (8, 10, 3)
    assert overlay.dtype == np.float32
    assert np.any(overlay[2:6, 4:8] > 0.0)


def test_opening_default_backend_matches_skimage_grayscale_opening() -> None:
    image = np.arange(15 * 17, dtype=np.float32).reshape(15, 17)
    image[3:8, 4:9] = 2.0
    footprint = skimage.morphology.disk(3)
    expected = skimage.morphology.opening(image, footprint)
    raw_opening = opening
    while hasattr(raw_opening, "__wrapped__"):
        raw_opening = raw_opening.__wrapped__

    observed = raw_opening(image, structuring_element="disk", size=3)

    np.testing.assert_array_equal(observed, expected)


def test_closing_default_backend_matches_skimage_grayscale_closing() -> None:
    image = np.arange(15 * 17, dtype=np.float32).reshape(15, 17)
    image[3:8, 4:9] = 2.0
    footprint = skimage.morphology.disk(3)
    expected = skimage.morphology.closing(image, footprint)
    raw_closing = closing
    while hasattr(raw_closing, "__wrapped__"):
        raw_closing = raw_closing.__wrapped__

    observed = raw_closing(image, structuring_element="disk", size=3)

    np.testing.assert_array_equal(observed, expected)


def test_threshold_unwraps_image_metadata_payload() -> None:
    payload = ImageMetadataPayload(
        data=np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    binary, measurements = threshold(
        payload,
        predefined_threshold=0.5,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(
        binary,
        np.array([[False, True], [False, True]], dtype=np.float32),
    )
    assert measurements.final_threshold == 0.5


def test_threshold_uses_and_preserves_input_image_mask() -> None:
    payload = MaskedImagePayload(
        data=np.array(
            [
                [0.0, 1.0, 1.0],
                [0.25, 0.75, 1.0],
            ],
            dtype=np.float32,
        ),
        mask=np.array(
            [
                [True, True, False],
                [True, True, False],
            ],
            dtype=bool,
        ),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    binary, measurements = threshold(
        payload,
        predefined_threshold=0.5,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(
        image_payload_data(binary),
        np.array(
            [
                [False, True, False],
                [False, True, False],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(image_payload_mask(binary), payload.mask)
    assert measurements.final_threshold == 0.5


def test_smooth_accepts_cellprofiler_display_setting_literals():
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0

    result = smooth(
        image,
        smoothing_method="Gaussian Filter",
        auto_object_size=False,
        object_size=3.0,
        dtype_config=DtypeConfig(),
    )

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.max(result) > 0


def test_smooth_matches_cellprofiler_masked_gaussian():
    from scipy.ndimage import gaussian_filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    image[1, 1] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:3, :3] = False
    payload = image_payload_with_context(image, mask=mask)
    object_size = 3.0

    result = smooth(
        payload,
        smoothing_method="Gaussian Filter",
        auto_object_size=False,
        object_size=object_size,
        dtype_config=DtypeConfig(),
    )

    sigma = object_size / 2.35
    masked_image = np.zeros(image.shape, dtype=image.dtype)
    masked_image[mask] = image[mask]
    weights = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0)
    expected = (
        gaussian_filter(masked_image, sigma, mode="constant", cval=0)
        / (weights + np.finfo(float).eps)
    )
    assert np.allclose(image_payload_data(result), expected.astype(np.float32))


def test_smooth_matches_cellprofiler_unmasked_gaussian_edge_normalization():
    from scipy.ndimage import gaussian_filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[0, 0] = 1.0
    object_size = 3.0

    result = smooth(
        image,
        smoothing_method="Gaussian Filter",
        auto_object_size=False,
        object_size=object_size,
        dtype_config=DtypeConfig(),
    )

    sigma = object_size / 2.35
    mask = np.ones(image.shape, dtype=bool)
    weights = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0)
    expected = gaussian_filter(image, sigma, mode="constant", cval=0) / (
        weights + np.finfo(float).eps
    )
    assert np.allclose(image_payload_data(result), expected.astype(np.float32))


def test_enhance_edges_accepts_cellprofiler_display_setting_literals():
    image = np.zeros((9, 9), dtype=np.float32)
    image[:, 5:] = 1.0

    result = enhance_edges(
        image,
        method="Sobel",
        direction="All",
        dtype_config=DtypeConfig(),
    )

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.max(result) > 0


def test_enhance_edges_uses_and_preserves_runtime_mask():
    import centrosome.filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[:, 5:] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:, :4] = False
    payload = image_payload_with_context(image, mask=mask)

    result = enhance_edges(
        payload,
        method="Sobel",
        direction="All",
        dtype_config=DtypeConfig(),
    )

    assert np.allclose(
        image_payload_data(result),
        centrosome.filter.sobel(image, mask).astype(np.float32),
    )
    assert np.array_equal(image_payload_mask(result), mask)


def test_closing_preserves_runtime_mask_context():
    from skimage.morphology import closing as skimage_closing
    from skimage.morphology import disk

    image = np.zeros((9, 9), dtype=np.float32)
    image[3:6, 3:6] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:2, :] = False
    payload = image_payload_with_context(image, mask=mask)

    result = closing(
        payload,
        structuring_element="disk",
        size=1,
        dtype_config=DtypeConfig(),
    )

    assert np.array_equal(image_payload_data(result), skimage_closing(image, disk(1)))
    assert np.array_equal(image_payload_mask(result), mask)


def test_cellprofiler_disk_structuring_element_uses_radius_setting():
    from benchmark.cellprofiler_library.functions.structuring_elements import (
        StructuringElement,
        build_structuring_element,
    )
    from skimage.morphology import disk

    np.testing.assert_array_equal(
        build_structuring_element(StructuringElement.DISK, 5),
        disk(5),
    )


def test_cellprofiler_ball_structuring_element_uses_radius_setting():
    from benchmark.cellprofiler_library.functions.structuring_elements import (
        StructuringElement,
        build_structuring_element,
    )
    from skimage.morphology import ball

    np.testing.assert_array_equal(
        build_structuring_element(StructuringElement.BALL, 2),
        ball(2),
    )


def test_cellprofiler_structuring_element_rank_adapts_by_center_section():
    from benchmark.cellprofiler_library.functions.structuring_elements import (
        StructuringElement,
        adapt_structuring_element_rank,
        build_structuring_element,
    )
    from skimage.morphology import disk

    footprint = build_structuring_element(StructuringElement.BALL, 2)

    np.testing.assert_array_equal(
        adapt_structuring_element_rank(footprint, 2),
        disk(2),
    )


def test_examplefly_absorbed_functions_import_cleanly():
    function_names = (
        "IdentifyPrimaryObjects",
        "IdentifySecondaryObjects",
        "IdentifyTertiaryObjects",
        "MeasureObjectSizeShape",
        "MeasureObjectIntensity",
        "MeasureTexture",
        "MeasureObjectNeighbors",
        "MeasureColocalization",
        "MeasureImageIntensity",
    )

    loaded_functions = {name: get_function(name) for name in function_names}

    assert all(func is not None for func in loaded_functions.values())


def test_measure_colocalization_object_costes_preserves_undefined_ratios():
    ratios = _divide_costes_measurements([0.0, 2.0], [0.0, 4.0])
    row = _object_colocalization_row(
        1,
        costes_m1=ratios[0],
        costes_m2=ratios[1],
    )

    assert np.isnan(row.costes_m1)
    assert row.costes_m2 == 0.5


def test_measure_colocalization_objects_accepts_unmasked_finite_images():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        )
    )
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)

    output, rows = measure_colocalization_objects.__wrapped__(
        image,
        labels,
        do_costes=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
    )

    assert np.array_equal(output, image[0:1])
    assert [row.object_label for row in rows] == [1, 2]
    assert all(np.isfinite(row.correlation) for row in rows)


def test_measure_colocalization_costes_first_threshold_snaps_to_scale_bin():
    assert (
        _costes_first_channel_bin_threshold(0.06666672229766846, 255)
        == 17 / 255
    )
    assert _costes_first_channel_bin_threshold(0.08594463765621185, 255) == (
        0.08594463765621185
    )


def test_measure_colocalization_costes_matches_scaled_bin_semantics():
    first = np.array([0, 0, 0, 5, 10, 20, 50], dtype=np.float32) / 255
    second = np.array([2, 4, 5, 6, 10, 20, 40], dtype=np.float32) / 255

    threshold_1, threshold_2 = _bisection_costes(first, second, 255)

    assert threshold_1 == 0
    assert threshold_2 == 5 / 255


def test_measure_colocalization_respects_masked_payload_pixels():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[4.0, 3.0], [2.0, 100.0]], dtype=np.float32),
        )
    )
    mask = np.array([[True, True], [True, False]])

    output, measurement = measure_colocalization.__wrapped__(
        MaskedImagePayload(data=image, mask=mask),
        do_costes=False,
    )

    assert measurement.correlation == -1.0
    assert isinstance(output, MaskedImagePayload)
    assert np.array_equal(image_payload_data(output), image[0:1])
    assert np.array_equal(image_payload_mask(output), mask)


def test_measure_colocalization_records_direct_reverse_slope():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [4.0, 9.0]], dtype=np.float32),
        )
    )

    _, measurement = measure_colocalization.__wrapped__(
        image,
        do_costes=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
    )

    first = image[0].ravel()
    second = image[1].ravel()
    expected_forward = np.linalg.lstsq(
        np.array((first, np.ones_like(first))).T,
        second,
        rcond=None,
    )[0][0]
    expected_reverse = np.linalg.lstsq(
        np.array((second, np.ones_like(second))).T,
        first,
        rcond=None,
    )[0][0]

    assert measurement.slope == pytest.approx(expected_forward)
    assert measurement.slope_reverse == pytest.approx(expected_reverse)


def test_measure_colocalization_threshold_metrics_match_numpy_semantics():
    first = np.array([0.0, 0.2, 0.2, 0.7, 0.9, 1.0], dtype=np.float64)
    second = np.array([0.1, 0.1, 0.4, 0.6, 0.95, 0.2], dtype=np.float64)
    threshold_percent = 15.0

    thr_fi = threshold_percent * np.max(first) / 100
    thr_si = threshold_percent * np.max(second) / 100
    thr_fi_out = first > thr_fi
    thr_si_out = second > thr_si
    combined = thr_fi_out & thr_si_out
    first_thresholded = first[combined]
    second_thresholded = second[combined]
    total_first = first[thr_fi_out].sum()
    total_second = second[thr_si_out].sum()

    rank1 = np.lexsort([first])
    rank2 = np.lexsort([second])
    rank1_u = np.hstack([[False], first[rank1[:-1]] != first[rank1[1:]]])
    rank2_u = np.hstack([[False], second[rank2[:-1]] != second[rank2[1:]]])
    rank1_s = np.cumsum(rank1_u)
    rank2_s = np.cumsum(rank2_u)
    rank_im1 = np.zeros(first.shape, dtype=int)
    rank_im2 = np.zeros(second.shape, dtype=int)
    rank_im1[rank1] = rank1_s
    rank_im2[rank2] = rank2_s
    rank_count = max(rank_im1.max(), rank_im2.max()) + 1
    weight = (rank_count - np.abs(rank_im1 - rank_im2)) / rank_count
    product_sum = (first_thresholded * second_thresholded).sum()

    expected = (
        first_thresholded.sum() / total_first,
        second_thresholded.sum() / total_second,
        (first_thresholded * weight[combined]).sum() / total_first,
        (second_thresholded * weight[combined]).sum() / total_second,
        product_sum
        / np.sqrt((first_thresholded**2).sum() * (second_thresholded**2).sum()),
        product_sum / (first_thresholded**2).sum(),
        product_sum / (second_thresholded**2).sum(),
    )

    observed = _thresholded_colocalization_metrics_numba(
        first,
        second,
        threshold_percent,
        True,
        True,
        True,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_cellprofiler_multiotsu_threshold_ignores_robust_background_settings():
    image = np.tile(
        np.array([0.05, 0.2, 0.75, 0.95], dtype=np.float32),
        (16, 16),
    )

    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.2,
        upper_outlier_fraction=0.2,
        averaging_method=CellProfilerAveragingMethod.MEDIAN,
        variance_method=CellProfilerVarianceMethod.MEDIAN_ABSOLUTE_DEVIATION,
        number_of_deviations=4,
        manual_threshold=0.5,
    )

    assert binary.dtype == np.bool_
    assert 0.0 < final_threshold < 1.0
    assert final_threshold == original_threshold


def test_minimum_cross_entropy_threshold_uses_default_numba_primitive():
    image = np.tile(
        np.array([0.0, 0.03, 0.08, 0.16, 0.4, 0.75], dtype=np.float32),
        (8, 8),
    )
    mask = np.ones(image.shape, dtype=bool)
    from openhcs.constants.constants import MemoryType
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    expected = ThresholdPrimitiveBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
    ).minimum_cross_entropy_threshold(
        image,
        mask=mask,
    )

    observed = cellprofiler_get_global_threshold(
        image,
        mask=mask,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    )

    assert observed == expected


def test_cellprofiler_three_class_otsu_honors_log_transform():
    from openhcs.processing.backends.cellprofiler.thresholding import (
        threshold_primitives,
    )

    image = np.array(
        [[0.01, 0.05, 0.06, 0.20, 0.25, 0.75, 0.90]] * 8,
        dtype=np.float32,
    )
    primitives = threshold_primitives()
    log_image, conversion = primitives.log_transform(image)
    log_values = log_image.ravel()
    expected_log_threshold = _threshold_multiotsu(
        log_values,
        nbins=CELLPROFILER_LOG_MULTI_OTSU_BINS,
    )[0] + (
        _threshold_histogram_bin_width(
            log_values,
            CELLPROFILER_LOG_MULTI_OTSU_BINS,
        )
        * CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET
    )
    expected_threshold = primitives.inverse_log_transform(
        expected_log_threshold,
        conversion,
    )

    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=True,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.2,
        upper_outlier_fraction=0.2,
        averaging_method=CellProfilerAveragingMethod.MEDIAN,
        variance_method=CellProfilerVarianceMethod.MEDIAN_ABSOLUTE_DEVIATION,
        number_of_deviations=4,
        manual_threshold=0.5,
    )

    np.testing.assert_allclose(final_threshold, expected_threshold)
    assert final_threshold == original_threshold
    np.testing.assert_array_equal(
        binary,
        image >= expected_threshold,
    )


def test_identify_primary_objects_basic_mode_fills_holes_like_cellprofiler():
    assert _fill_before_declump_requested(
        use_advanced_settings=False,
        fill_holes=FillHolesOption.AFTER_DECLUMP,
    )
    assert _fill_after_declump_requested(
        use_advanced_settings=False,
        fill_holes=FillHolesOption.NEVER,
    )
    assert not _fill_before_declump_requested(
        use_advanced_settings=True,
        fill_holes=FillHolesOption.AFTER_DECLUMP,
    )


def test_cellprofiler_basic_threshold_uses_native_default_smoothing(monkeypatch):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = {}

    def smooth_image(pixel_data, mask, smoothing, **_kwargs):
        calls["threshold_smoothing"] = smoothing
        return np.asarray(pixel_data), smoothing

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_method"] = kwargs["threshold_method"]
        return 0.25

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_smoothing"] = smoothing
        return np.asarray(pixel_data) >= threshold, 0.0

    monkeypatch.setattr(
        thresholding_module,
        "_threshold_smoothed_image",
        smooth_image,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_apply_threshold",
        apply_threshold,
    )

    cellprofiler_threshold(
        np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3),
        use_advanced_settings=False,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=True,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
    )

    assert calls == {
        "threshold_method": CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        "application_smoothing": (
            thresholding_module.CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE
        ),
    }


def test_cellprofiler_threshold_passes_mask_to_native_thresholds(monkeypatch):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = []
    image = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]])

    def get_global_threshold(pixel_data, **kwargs):
        calls.append(("threshold", kwargs["mask"]))
        return 0.5 * kwargs["threshold_correction_factor"]

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls.append(("apply", mask))
        return np.asarray(pixel_data) >= threshold, 0.0

    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_apply_threshold",
        apply_threshold,
    )

    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=0.7,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=10,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        mask=mask,
    )

    assert final_threshold == 0.35
    assert original_threshold == 0.5
    np.testing.assert_array_equal(binary, np.array([[False, False], [True, False]]))
    assert [name for name, _mask in calls] == ["threshold", "apply"]
    assert all(np.array_equal(_mask, mask) for _name, _mask in calls)


def test_identify_primary_objects_applies_threshold_smoothing_to_binary_mask(
    monkeypatch,
) -> None:
    calls = {}

    def fake_threshold(pixel_data, **kwargs):
        calls["threshold_smoothing_scale"] = kwargs["threshold_smoothing_scale"]
        calls["smooth_threshold_application"] = kwargs[
            "smooth_threshold_application"
        ]
        return np.zeros_like(pixel_data, dtype=bool), 0.1, 0.1

    monkeypatch.setattr(
        identifyprimaryobjects_module,
        "cellprofiler_threshold",
        fake_threshold,
    )

    identify_primary_objects(
        np.zeros((8, 8), dtype=np.float32),
        use_advanced_settings=True,
        smoothing_filter_size=10,
        threshold_smoothing_scale=1.3488,
        min_diameter=2,
        max_diameter=8,
        dtype_config=DtypeConfig(),
    )

    assert calls["threshold_smoothing_scale"] == pytest.approx(1.3488)
    assert calls["smooth_threshold_application"] is True


def test_identify_primary_objects_coerces_cellprofiler_literal_enums_directly():
    image = np.zeros((8, 8), dtype=np.float32)
    image[2:6, 2:6] = 1.0

    _image, stats, labels = identify_primary_objects(
        image,
        min_diameter=2,
        max_diameter=8,
        exclude_size=False,
        exclude_border_objects=False,
        unclump_method="None",
        watershed_method="None",
        fill_holes="After both thresholding and declumping",
        limit_erase="Continue",
        threshold_method="Manual",
        manual_threshold=0.5,
        dtype_config=DtypeConfig(),
    )

    assert stats.object_count == 1
    assert labels.labels.max() == 1


def test_identify_primary_objects_threshold_diagnostics_use_pre_fill_binary(
    monkeypatch,
):
    threshold_binary = np.zeros((7, 7), dtype=bool)
    threshold_binary[2:5, 2:5] = True
    threshold_binary[3, 3] = False
    captured = {}

    def fake_threshold(pixel_data, **kwargs):
        return threshold_binary.copy(), 0.5, 0.5

    def fake_diagnostics(image, binary, **kwargs):
        captured["binary"] = np.asarray(binary, dtype=bool).copy()
        return types.SimpleNamespace(
            original_threshold=0.5,
            weighted_variance=0.0,
            sum_of_entropies=0.0,
        )

    monkeypatch.setattr(
        identifyprimaryobjects_module,
        "cellprofiler_threshold",
        fake_threshold,
    )
    monkeypatch.setattr(
        identifyprimaryobjects_module,
        "cellprofiler_threshold_diagnostics",
        fake_diagnostics,
    )

    identify_primary_objects(
        np.ones((7, 7), dtype=np.float32),
        min_diameter=1,
        max_diameter=10,
        exclude_size=False,
        exclude_border_objects=False,
        unclump_method="None",
        watershed_method="None",
        fill_holes="After both thresholding and declumping",
        threshold_method="Manual",
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(captured["binary"], threshold_binary)


def test_identify_primary_objects_declumping_maxima_geometry_matches_public_semantics():
    assert _declumping_maxima_geometry(
        min_diameter=8,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    ) == (1.0, 8.0 / 1.5)
    assert _declumping_maxima_geometry(
        min_diameter=8,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
        median_initial_object_radius=6.2,
    ) == (1.0, 8.0 / 1.5)
    assert _declumping_maxima_geometry(
        min_diameter=8,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    ) == (1.0, 8.0 / 1.5)
    assert _declumping_maxima_geometry(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    ) == (0.5, CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE)
    assert _declumping_maxima_geometry(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=False,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    ) == (0.5, 4.0)
    assert _declumping_maxima_geometry(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    ) == (0.5, CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE)
    assert _declumping_maxima_geometry(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=False,
        maxima_suppression_size=7.0,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    ) == (0.5, 4.0)
    assert _declumping_maxima_geometry(
        min_diameter=4,
        low_res_maxima=True,
        automatic_suppression=False,
        maxima_suppression_size=4.0,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    ) == (1.0, 4.0)
    assert _manual_declumping_size(0) == 0.0
    assert _manual_declumping_size(4) == 4.0


def test_identify_primary_objects_declumping_footprint_respects_min_diameter():
    class FakeMorphology:
        def __init__(self):
            self.calls = []

        def declumping_suppression_footprint(
            self,
            suppress_size,
            *,
            min_diameter,
            declump_method,
        ):
            self.calls.append((suppress_size, min_diameter, declump_method))
            return np.ones((1, 1), dtype=bool)

    morphology = FakeMorphology()

    _declumping_suppression_footprint(
        morphology,
        4,
        min_diameter=5,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    )
    _declumping_suppression_footprint(
        morphology,
        4,
        min_diameter=4,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    )
    _declumping_suppression_footprint(
        morphology,
        2,
        min_diameter=1,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    )

    assert morphology.calls == [
        (4, 5, CellProfilerDeclumpMethod.INTENSITY),
        (4, 4, CellProfilerDeclumpMethod.INTENSITY),
        (2, 1, CellProfilerDeclumpMethod.SHAPE),
    ]


def test_cellprofiler_threshold_can_apply_unsmoothed_threshold(monkeypatch):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_smoothing"] = smoothing
        return np.asarray(pixel_data) >= threshold, 0.0

    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_apply_threshold",
        apply_threshold,
    )

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.TRIANGLE,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=False,
    )

    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    assert calls["application_smoothing"] == 0.0


def test_cellprofiler_global_otsu_uses_raw_threshold_estimate(
    monkeypatch,
):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = {"smooth_count": 0}

    def smooth_image(pixel_data, mask, smoothing, **_kwargs):
        calls["smooth_count"] += 1
        smoothed = np.asarray(pixel_data) + 1
        calls["smoothed_pixels"] = smoothed
        return smoothed, smoothing

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_pixels"] = np.asarray(pixel_data).copy()
        calls["application_smoothing"] = smoothing
        return np.asarray(pixel_data) >= threshold, 0.0

    monkeypatch.setattr(
        thresholding_module,
        "_threshold_smoothed_image",
        smooth_image,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_apply_threshold",
        apply_threshold,
    )

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=True,
    )

    assert calls["smooth_count"] == 0
    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    np.testing.assert_array_equal(calls["application_pixels"], image)
    assert calls["application_smoothing"] == 2.0


def test_cellprofiler_global_robust_background_uses_raw_threshold_estimate(
    monkeypatch,
):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = {"smooth_count": 0}

    def smooth_image(pixel_data, mask, smoothing, **_kwargs):
        calls["smooth_count"] += 1
        smoothed = np.asarray(pixel_data) + 1
        calls["smoothed_pixels"] = smoothed
        return smoothed, smoothing

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_pixels"] = np.asarray(pixel_data).copy()
        calls["application_smoothing"] = smoothing
        return np.asarray(pixel_data) >= threshold, 0.0

    monkeypatch.setattr(
        thresholding_module,
        "_threshold_smoothed_image",
        smooth_image,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_apply_threshold",
        apply_threshold,
    )

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.ROBUST_BACKGROUND,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=True,
    )

    assert calls["smooth_count"] == 0
    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    np.testing.assert_array_equal(calls["application_pixels"], image)
    assert calls["application_smoothing"] == 2.0


def test_cellprofiler_minimum_cross_entropy_uses_unsmoothed_threshold_estimate(
    monkeypatch,
):
    from benchmark.cellprofiler_library.functions import thresholding as thresholding_module

    calls = {}

    def smooth_image(pixel_data, mask, smoothing, **_kwargs):
        calls["smoothed_pixels"] = np.asarray(pixel_data) + 1
        return calls["smoothed_pixels"], smoothing

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    monkeypatch.setattr(
        thresholding_module,
        "_threshold_smoothed_image",
        smooth_image,
    )
    monkeypatch.setattr(
        thresholding_module,
        "cellprofiler_get_global_threshold",
        get_global_threshold,
    )

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
    )

    np.testing.assert_array_equal(calls["threshold_pixels"], image)


def test_cellprofiler_threshold_diagnostics_matches_reference_formula():
    import centrosome.threshold

    rng = np.random.default_rng(7)
    image = rng.random((16, 17), dtype=np.float32)
    mask = rng.random(image.shape) > 0.2
    binary = image > 0.45

    diagnostics = cellprofiler_threshold_diagnostics(
        image,
        binary,
        final_threshold=0.45,
        original_threshold=0.4,
        mask=mask,
    )

    np.testing.assert_allclose(
        diagnostics.weighted_variance,
        centrosome.threshold.weighted_variance(image, mask, binary),
    )
    np.testing.assert_allclose(
        diagnostics.sum_of_entropies,
        centrosome.threshold.sum_of_entropies(image, mask, binary),
    )


def test_identify_primary_objects_filters_crop_mask_border_objects():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1, 2] = 2
    labels[4, 4] = 3
    mask = np.zeros_like(labels, dtype=bool)
    mask[1:5, 1:5] = True

    filtered = _filter_border_objects(labels, image_mask=mask)

    assert 1 in filtered
    assert 2 not in filtered
    assert 3 not in filtered


def test_identify_primary_objects_ignores_threshold_only_mask_border():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1, 2] = 2
    mask = np.zeros_like(labels, dtype=bool)
    mask[1:5, 1:5] = True
    metadata = ImagePayloadMetadata(mask_defines_border=False)

    filtered = _filter_border_objects(
        labels,
        image_mask=mask,
        image_metadata=metadata,
    )

    assert 1 in filtered
    assert 2 in filtered


def test_identify_primary_objects_keeps_crop_local_nonphysical_edges():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1:3, 0:2] = 2
    mask = np.ones_like(labels, dtype=bool)
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(12, 12),
        output_shape_yx=labels.shape,
        offset_yx=(3, 4),
        physical_border_edges_yx=(False, False, False, False),
    )

    filtered = _filter_border_objects(
        labels,
        image_mask=mask,
        image_metadata=metadata,
    )

    assert 1 in filtered
    assert 2 in filtered


def test_identify_primary_objects_removes_true_physical_edge_objects():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[0:2, 2:4] = 2
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 10),
        output_shape_yx=labels.shape,
        offset_yx=(0, 2),
        physical_border_edges_yx=(True, False, False, False),
    )

    filtered = _filter_border_objects(
        labels,
        image_mask=np.ones_like(labels, dtype=bool),
        image_metadata=metadata,
    )

    assert 1 in filtered
    assert 2 not in filtered


def test_identify_primary_objects_filters_stacked_label_sizes_planewise():
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 3:6, 3:6] = 2
    labels[1, 1:4, 1:4] = 10
    labels[1, 4:6, 4:6] = 20

    small_removed, final = identifyprimaryobjects_module._filter_labels_by_diameter_range(
        labels,
        min_diameter=2.5,
        max_diameter=3.5,
    )

    assert 1 not in small_removed[0]
    assert 2 in small_removed[0]
    assert 10 in small_removed[1]
    assert 20 not in small_removed[1]
    assert 1 not in final[0]
    assert 2 in final[0]
    assert 10 in final[1]
    assert 20 not in final[1]


def test_identify_primary_objects_filters_stacked_border_objects_planewise():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 0:2, 3:5] = 2
    labels[1, 2:4, 2:4] = 10
    labels[1, 3:5, 0:2] = 20

    filtered = _filter_border_objects(
        labels,
        image_mask=np.ones_like(labels, dtype=bool),
    )

    assert 1 in filtered[0]
    assert 2 not in filtered[0]
    assert 10 in filtered[1]
    assert 20 not in filtered[1]


def test_cellprofiler_legacy_watershed_keeps_descending_pixel_priority():
    from benchmark.cellprofiler_library.functions.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    markers = np.array([[1, 0, 2]], dtype=np.int32)

    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        connectivity=np.ones((1, 3), dtype=bool),
    )

    np.testing.assert_array_equal(labels, np.array([[1, 1, 2]], dtype=np.int32))


def test_cellprofiler4_marker_watershed_uses_legacy_priority_semantics():
    import inspect

    from benchmark.cellprofiler_library.functions.watershed import watershed

    image = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    markers = np.array([[1, 0, 2]], dtype=np.int32)
    raw_watershed = inspect.unwrap(watershed)
    _image, _stats, labels = raw_watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        watershed_method="markers",
        declump_method="shape",
        use_advanced_settings=False,
        runtime_family="cellprofiler4",
    )

    np.testing.assert_array_equal(labels, np.array([[1, 1, 2]], dtype=np.int32))


def test_cellprofiler_fast_legacy_watershed_matches_reference_path():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    rng = np.random.default_rng(123)
    for _case in range(20):
        image = rng.integers(0, 6, size=(8, 7)).astype(float) / 5.0
        mask = rng.random((8, 7)) > 0.15
        markers = np.zeros((8, 7), dtype=np.int32)
        coords = np.argwhere(mask)
        selected = coords[rng.choice(len(coords), size=4, replace=False)]
        for label, (y, x) in enumerate(selected, start=1):
            markers[y, x] = label

        reference = cellprofiler_legacy_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=np.ones((3, 3), dtype=bool),
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
        fast = cellprofiler_legacy_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=np.ones((3, 3), dtype=bool),
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )

        np.testing.assert_array_equal(fast, reference)


def test_cellprofiler_legacy_watershed_handles_stacked_planes_planewise():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.stack(
        (
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.5, 0.0]], dtype=np.float64),
        )
    )
    markers = np.stack(
        (
            np.array([[1, 0, 2]], dtype=np.int32),
            np.array([[10, 0, 20]], dtype=np.int32),
        )
    )
    mask = np.ones_like(image, dtype=bool)

    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=np.ones((1, 3), dtype=bool),
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    expected = np.stack(
        (
            np.array([[1, 1, 2]], dtype=np.int32),
            np.array([[10, 10, 20]], dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(labels, expected)


def test_cellprofiler_legacy_watershed_scalar_connectivity_is_volumetric():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.zeros((2, 3, 3), dtype=np.float64)
    markers = np.zeros_like(image, dtype=np.int32)
    markers[0, 1, 1] = 1
    mask = np.ones_like(image, dtype=bool)

    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=1,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    np.testing.assert_array_equal(labels, np.ones_like(markers))


def test_cellprofiler_fast_legacy_watershed_uses_required_numba_backend():
    from openhcs.constants.constants import MemoryType
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
        NumbaNumpyLegacyWatershedBackendStrategy,
    )

    watershed_backend = importlib.import_module(
        "openhcs.processing.backends.cellprofiler.watershed"
    )

    assert watershed_backend._legacy_watershed_raveled_numba is not None
    assert type(LegacyWatershedBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyLegacyWatershedBackendStrategy
    )


def test_measure_texture_uses_cellprofiler_haralick_backend(monkeypatch):
    from benchmark.cellprofiler_library.functions.measuretexture import measure_texture
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((pixel_data.copy(), distance, ignore_zeros))
        features = np.arange(52, dtype=float).reshape(4, 13)
        features[1, 2] = np.nan
        return features

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)

    image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)

    _, measurements = measure_texture(
        image,
        scale=2,
        gray_levels=8,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )

    assert calls[0][1:] == (2, False)
    assert calls[0][0].dtype == np.uint8
    assert calls[0][0].max() <= 7
    assert measurements[0].contrast == 1.0
    assert measurements[1].correlation == 0.0


def test_measure_texture_emits_all_requested_scales(monkeypatch):
    from benchmark.cellprofiler_library.functions.measuretexture import measure_texture
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((distance, ignore_zeros))
        return np.full((4, 13), float(distance), dtype=float)

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)

    image = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)

    _, measurements = measure_texture(
        image,
        scale=(2, 4),
        gray_levels=8,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )

    assert calls == [(2, False), (4, False)]
    assert [measurement.scale for measurement in measurements] == [2] * 4 + [4] * 4


def test_measure_texture_objects_uses_cellprofiler_object_backend(monkeypatch):
    from benchmark.cellprofiler_library.functions.measuretexture import (
        measure_texture_objects,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((pixel_data.copy(), distance, ignore_zeros))
        features = np.ones((4, 13), dtype=float)
        features[0, 3] = np.inf
        return features

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)

    image = np.full((5, 5), 0.5, dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    _, measurements = measure_texture_objects(
        image,
        labels,
        scale=1,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )

    assert calls[0][1:] == (1, True)
    assert calls[0][0].dtype == np.uint8
    assert measurements[0].object_label == 1
    assert measurements[0].variance == 0.0


def test_numba_haralick_backend_matches_mahotas_reference():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.texture import (
        HaralickTextureBackendStrategy,
    )

    rng = np.random.default_rng(7123)
    native_backend = HaralickTextureBackendStrategy.for_memory_type(
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    numba_backend = HaralickTextureBackendStrategy.for_memory_type(
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    for ignore_zeros in (False, True):
        for scale in (1, 2, 3):
            image = rng.integers(0, 16, size=(14, 13), dtype=np.uint8)
            image[0:2, 0:2] = 0
            expected = native_backend.haralick_features(
                image,
                scale=scale,
                ignore_zeros=ignore_zeros,
            )
            actual = numba_backend.haralick_features(
                image,
                scale=scale,
                ignore_zeros=ignore_zeros,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_object_texture_crop_backend_matches_regionprops_intensity_images():
    from skimage.measure import regionprops

    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:4, 1:4] = 3
    labels[2, 2] = 0
    labels[0:2, 4:6] = 7
    labels[5, 0] = 12

    backend = ObjectTextureCropBackendStrategy.for_callable(
        test_object_texture_crop_backend_matches_regionprops_intensity_images
    )
    object_labels, intensity_crops = backend.object_intensity_crops(image, labels)

    expected_props = regionprops(labels, intensity_image=image)
    assert object_labels.tolist() == [prop.label for prop in expected_props]
    assert len(intensity_crops) == len(expected_props)
    for crop, prop in zip(intensity_crops, expected_props, strict=True):
        np.testing.assert_array_equal(crop, prop.intensity_image)


def test_measure_object_intensity_uses_cellprofiler_mad_interpolation():
    from benchmark.cellprofiler_library.functions.measureobjectintensity import (
        measure_object_intensity,
    )

    image = np.array([[0.0, 10.0, 20.0]], dtype=np.float32)
    labels = np.array([[1, 1, 1]], dtype=np.int32)

    _, measurements = measure_object_intensity(
        image,
        labels,
        dtype_config=DtypeConfig(),
    )

    assert measurements[0].median_intensity == 15.0
    assert measurements[0].mad_intensity == 10.0


def test_measure_object_intensity_accepts_replicated_rgb_grayscale_like_cellprofiler():
    from benchmark.cellprofiler_library.functions.measureobjectintensity import (
        measure_object_intensity,
    )

    plane = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    image = np.repeat(plane[..., None], 3, axis=-1)
    labels = np.array([[1, 1], [0, 2]], dtype=np.int32)

    _, measurements = measure_object_intensity(
        image,
        labels,
        dtype_config=DtypeConfig(),
    )

    by_label = {measurement.object_label: measurement for measurement in measurements}
    assert by_label[1].integrated_intensity == 3.0
    assert by_label[2].integrated_intensity == 4.0


def test_measure_object_intensity_measures_3d_objects_as_single_volume_domain():
    from benchmark.cellprofiler_library.functions.measureobjectintensity import (
        measure_object_intensity,
    )

    image = np.ones((3, 4, 5), dtype=np.float32)
    labels = np.zeros((3, 4, 5), dtype=np.int32)
    labels[:, 1:3, 2:4] = 1

    _, measurements = measure_object_intensity(
        image,
        labels,
        dtype_config=DtypeConfig(),
    )

    assert len(measurements) == 1
    assert measurements[0].integrated_intensity == 12.0
    assert measurements[0].center_mass_intensity_z == 1.0
    assert measurements[0].max_intensity_z == 2.0


def test_measure_object_intensity_maximum_position_tie_matches_cellprofiler():
    from benchmark.cellprofiler_library.functions.measureobjectintensity import (
        measure_object_intensity,
    )

    image = np.array(
        [
            [1.0, 5.0],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )
    labels = np.ones(image.shape, dtype=np.int32)

    _, measurements = measure_object_intensity(
        image,
        labels,
        dtype_config=DtypeConfig(),
    )

    assert len(measurements) == 1
    assert measurements[0].max_intensity_x == 1.0
    assert measurements[0].max_intensity_y == 1.0


def test_measure_object_intensity_rejects_true_color_images_like_cellprofiler():
    from benchmark.cellprofiler_library.functions.measureobjectintensity import (
        measure_object_intensity,
    )

    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[..., 1] = 1.0
    labels = np.ones((2, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="requires a 2-D grayscale image"):
        measure_object_intensity(
            image,
            labels,
            dtype_config=DtypeConfig(),
        )


def test_measure_image_quality_uses_openhcs_image_quality_backend():
    from benchmark.cellprofiler_library.functions.measureimagequality import (
        _calculate_correlation,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(100, dtype=np.float32).reshape(10, 10) / 100.0

    default_value = _calculate_correlation(image, 2)
    centrosome_value = _calculate_correlation(
        image,
        2,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    assert np.isclose(default_value, centrosome_value, rtol=1e-6, atol=1e-6)


def test_measure_image_quality_uses_openhcs_power_spectrum_backend(monkeypatch):
    import benchmark.cellprofiler_library.functions.measureimagequality as module
    from benchmark.cellprofiler_library.functions.measureimagequality import (
        _calculate_power_spectrum_slope,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    class Backend:
        def radial_power_spectrum(self, pixel_data):
            calls.append(pixel_data.copy())
            radii = np.array([1.0, 2.0, 4.0])
            magnitude = np.array([1.0, 1.0, 1.0])
            power = radii ** -2
            return radii, magnitude, power

    def backend_factory(*, backend_provider=None):
        assert backend_provider is CellProfilerBackendProvider.NUMBA
        return Backend()

    monkeypatch.setattr(module, "image_quality_backend", backend_factory)

    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    assert np.isclose(
        _calculate_power_spectrum_slope(
            image,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        ),
        -2.0,
    )
    np.testing.assert_array_equal(calls[0], image)


def test_measure_image_quality_uses_numba_otsu():
    from benchmark.cellprofiler_library.functions.measureimagequality import (
        ThresholdMethod,
        _calculate_threshold,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        threshold_primitives,
    )

    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    expected = threshold_primitives().weighted_otsu_threshold(
        image.astype(np.float32, copy=False),
    )

    assert _calculate_threshold(image, ThresholdMethod.OTSU) == expected


def test_measure_image_quality_constancy_check_matches_numpy_unique():
    from benchmark.cellprofiler_library.functions.measureimagequality import (
        _has_multiple_unique_values,
    )

    cases = (
        np.array([[1.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[np.nan, np.nan]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[0.0, -0.0]], dtype=np.float32),
    )

    for image in cases:
        expected = len(np.unique(image)) > 1
        assert _has_multiple_unique_values(image) is expected


def test_measure_image_quality_log_log_slope_matches_lstsq():
    import scipy.linalg

    from benchmark.cellprofiler_library.functions.measureimagequality import (
        _least_squares_log_log_slope_numba,
    )

    radii = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    power = radii**-1.75
    idx = np.isfinite(np.log(power))
    design = np.hstack(
        (
            np.log(radii)[idx][:, np.newaxis],
            np.ones(radii.shape)[idx][:, np.newaxis],
        )
    )
    expected = scipy.linalg.lstsq(
        design,
        np.log(power)[idx][:, np.newaxis],
    )[0][0]

    assert np.isclose(
        _least_squares_log_log_slope_numba(radii, power),
        float(np.asarray(expected).ravel()[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_measure_image_quality_local_focus_matches_grid_semantics():
    from scipy.ndimage import mean as ndimage_mean, sum as ndimage_sum

    from benchmark.cellprofiler_library.functions.measureimagequality import (
        _calculate_local_focus_score,
    )

    image = np.arange(35, dtype=np.float32).reshape(5, 7) / 10.0
    scale = 3
    shape = image.shape
    i, j = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    m, n = (np.array(shape) + scale - 1) // scale
    i = (i * float(m) / float(shape[0])).astype(int)
    j = (j * float(n) / float(shape[1])).astype(int)
    grid = i * n + j + 1
    grid_range = np.arange(0, m * n + 1, dtype=np.int32)
    local_means = np.nan_to_num(
        ndimage_mean(image, grid, grid_range),
        nan=0.0,
    )
    local_squared_normalized = (image - local_means[grid]) ** 2
    grid_mask = (local_means != 0) & np.isfinite(local_means)
    nz_grid_range = grid_range[grid_mask]
    if nz_grid_range[0] == 0:
        nz_grid_range = nz_grid_range[1:]
        local_means = local_means[1:]
        grid_mask = grid_mask[1:]
    sums = ndimage_sum(local_squared_normalized, grid, nz_grid_range)
    pixel_counts = ndimage_sum(np.ones(shape), grid, nz_grid_range)
    valid_means = (
        local_means[grid_mask]
        if len(local_means) > len(nz_grid_range)
        else local_means[: len(nz_grid_range)]
    )
    expected_values = sums / (pixel_counts * valid_means[: len(sums)])
    expected_values = expected_values[np.isfinite(expected_values)]
    expected = float(np.var(expected_values) / np.median(expected_values))

    assert np.isclose(
        _calculate_local_focus_score(image, scale),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_measure_object_neighbors_accepts_explicit_centrosome_morphology(monkeypatch):
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from benchmark.cellprofiler_library.functions.measureobjectneighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    disk_calls = []
    outline_calls = []

    def strel_disk(radius):
        disk_calls.append(radius)
        return np.ones((3, 3), dtype=bool)

    def outline(labels):
        outline_calls.append(labels.copy())
        result = np.zeros_like(labels)
        result[labels > 0] = labels[labels > 0]
        return result

    def centers_of_labels(labels):
        centers = []
        for label in range(1, int(labels.max()) + 1):
            coords = np.argwhere(labels == label)
            centers.append(coords.mean(axis=0) if coords.size else (0.0, 0.0))
        return np.asarray(centers).T

    centrosome_module = types.ModuleType("centrosome")
    cpmorphology_module = types.ModuleType("centrosome.cpmorphology")
    outline_module = types.ModuleType("centrosome.outline")
    cpmorphology_module.strel_disk = strel_disk
    cpmorphology_module.centers_of_labels = centers_of_labels
    outline_module.outline = outline
    centrosome_module.cpmorphology = cpmorphology_module
    centrosome_module.outline = outline_module
    monkeypatch.setitem(sys.modules, "centrosome", centrosome_module)
    monkeypatch.setitem(sys.modules, "centrosome.cpmorphology", cpmorphology_module)
    monkeypatch.setitem(sys.modules, "centrosome.outline", outline_module)

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2, 2] = 1
    labels[2, 4] = 2

    _, measurements = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        labels,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=4,
        dtype_config=DtypeConfig(),
        morphology_backend_provider=CellProfilerBackendProvider.CENTROSOME,
        outline_backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    assert disk_calls == [4, 4.5]
    np.testing.assert_array_equal(outline_calls[0], labels)
    assert len(measurements) == 2


def test_measure_object_neighbors_counts_small_removed_discarded_neighbors():
    from benchmark.cellprofiler_library.functions.measureobjectneighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 1] = 1
    small_removed = labels.copy()
    small_removed[3, 3] = 2

    _, with_discarded = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        labels,
        small_removed_labels=small_removed,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        consider_discarded_objects=True,
        dtype_config=DtypeConfig(),
    )
    _, without_discarded = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        labels,
        small_removed_labels=small_removed,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        consider_discarded_objects=False,
        dtype_config=DtypeConfig(),
    )

    assert with_discarded[0].number_of_neighbors == 1
    assert without_discarded[0].number_of_neighbors == 0


def test_measure_object_neighbors_returns_retained_count_image():
    from benchmark.cellprofiler_library.functions.measureobjectneighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 2] = 1
    labels[3, 4] = 2

    count_image, measurements = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        labels,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        retain_neighbor_count_image=True,
        neighbor_count_colormap="hot",
        dtype_config=DtypeConfig(),
    )

    assert count_image.shape == (7, 7, 3)
    np.testing.assert_array_equal(count_image[labels == 0], 0)
    assert np.any(count_image[labels > 0] > 0)
    assert [measurement.number_of_neighbors for measurement in measurements] == [1, 1]


def test_medianfilter_matches_cellprofiler_constant_default():
    from scipy.ndimage import median_filter as scipy_median_filter

    from benchmark.cellprofiler_library.functions.medianfilter import medianfilter

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    image[1, 2] = 100.0
    image[3, 5] = -20.0

    observed = medianfilter(
        image,
        window_size=3,
        dtype_config=DtypeConfig(),
    )
    expected = scipy_median_filter(image, size=3, mode="constant").astype(image.dtype)

    np.testing.assert_array_equal(observed, expected)


def test_medianfilter_honors_explicit_reflect_mode():
    from scipy.ndimage import median_filter as scipy_median_filter

    from benchmark.cellprofiler_library.functions.medianfilter import medianfilter

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    image[1, 2] = 100.0
    image[3, 5] = -20.0

    observed = medianfilter(
        image,
        window_size=3,
        mode="reflect",
        dtype_config=DtypeConfig(),
    )
    expected = scipy_median_filter(image, size=3, mode="reflect").astype(image.dtype)

    np.testing.assert_array_equal(observed, expected)


def test_image_math_coerces_cellprofiler_operation_strings():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)

    result = image_math(image, operation="Invert", dtype_config=DtypeConfig())

    np.testing.assert_allclose(result[0], 1 - image)


def test_image_math_preserves_or_ignores_masked_image_payload():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])
    payload = MaskedImagePayload(data=image, mask=mask)

    preserved = image_math(payload, operation="Invert", dtype_config=DtypeConfig())
    ignored = image_math(
        payload,
        operation="Invert",
        ignore_masks=True,
        dtype_config=DtypeConfig(),
    )

    assert isinstance(preserved, MaskedImagePayload)
    np.testing.assert_array_equal(preserved.mask, mask)
    np.testing.assert_allclose(preserved.data[0], (1 - image) * mask)
    assert isinstance(ignored, np.ndarray)
    np.testing.assert_allclose(ignored[0], 1 - image)


def test_image_math_combines_operand_masks_without_reexpanding_single_output():
    image = np.stack(
        (
            np.full((2, 3, 4), 0.1, dtype=np.float32),
            np.full((2, 3, 4), 0.2, dtype=np.float32),
            np.full((2, 3, 4), 0.3, dtype=np.float32),
        )
    )
    mask = np.stack(
        (
            np.ones((2, 3, 4), dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
        )
    )
    mask[0, 0, 0, 0] = False
    mask[1, 0, 0, 1] = False
    mask[2, 0, 0, 2] = False
    payload = MaskedImagePayload(data=image, mask=mask)

    result = image_math(
        payload,
        operation="Add",
        factors=(1.0, 1.0, 1.0),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, MaskedImagePayload)
    expected_mask = mask[0] & mask[1] & mask[2]
    assert result.data.shape == image.shape[1:]
    np.testing.assert_array_equal(result.mask, expected_mask)
    np.testing.assert_allclose(result.data, image.sum(axis=0) * expected_mask)


def test_correct_illumination_apply_preserves_source_image_metadata() -> None:
    image = np.stack(
        (
            np.full((2, 2), 0.5, dtype=np.float32),
            np.full((2, 2), 0.25, dtype=np.float32),
        )
    )
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(
            channel_intensity_scales=(65535.0, None),
            channel_source_dtypes=("uint16", None),
        ),
    )

    result = correct_illumination_apply(
        payload,
        dtype_config=DtypeConfig(),
    )

    assert result.metadata.intensity_scale == 65535.0
    assert result.metadata.source_dtype == "uint16"
    np.testing.assert_allclose(image_payload_data(result), np.ones((1, 2, 2)))


def test_correct_illumination_apply_handles_multiple_image_function_pairs() -> None:
    image = np.stack(
        (
            np.full((2, 2), 0.8, dtype=np.float32),
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
            np.full((2, 2), 0.3, dtype=np.float32),
        )
    )

    first, second = correct_illumination_apply(
        image,
        method=("Subtract", "Divide"),
        truncate_low=False,
        truncate_high=True,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(
        image_payload_data(first),
        np.full((1, 2, 2), 0.6, dtype=np.float32),
    )
    np.testing.assert_allclose(
        image_payload_data(second),
        np.ones((1, 2, 2), dtype=np.float32),
    )


def test_correct_illumination_apply_handles_interleaved_multi_site_stack() -> None:
    image = np.stack(
        (
            np.full((2, 2), 0.8, dtype=np.float32),
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
            np.full((2, 2), 0.3, dtype=np.float32),
        )
    )

    result = correct_illumination_apply(
        image,
        method="Subtract",
        truncate_low=False,
        truncate_high=True,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(
        tuple(image_payload_data(output) for output in result),
        (
            np.full((1, 2, 2), 0.6, dtype=np.float32),
            np.full((1, 2, 2), 0.3, dtype=np.float32),
        ),
    )


def test_correct_illumination_apply_rejects_mismatched_method_count() -> None:
    image = np.stack(
        (
            np.full((2, 2), 0.8, dtype=np.float32),
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
            np.full((2, 2), 0.3, dtype=np.float32),
        )
    )

    with pytest.raises(ValueError, match="method count must match"):
        correct_illumination_apply(
            image,
            method=("Subtract",),
            dtype_config=DtypeConfig(),
        )


def test_correct_illumination_apply_handles_repeated_methods_for_interleaved_stack() -> None:
    image = np.stack(
        (
            np.full((2, 2), 0.8, dtype=np.float32),
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
            np.full((2, 2), 0.3, dtype=np.float32),
        )
    )

    first, second = correct_illumination_apply(
        image,
        method="Subtract",
        truncate_low=False,
        truncate_high=True,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(
        np.stack(
            (
                image_payload_data(first),
                image_payload_data(second),
            )
        ),
        np.stack(
            (
                np.full((1, 2, 2), 0.6, dtype=np.float32),
                np.full((1, 2, 2), 0.3, dtype=np.float32),
            )
        ),
    )


def test_measure_colocalization_uses_payload_metadata_for_costes_scale() -> None:
    image = np.stack(
        (
            np.array([[0.0, 1.0], [0.3, 0.4]], dtype=np.float32),
            np.array([[0.1, 0.8], [0.2, 0.5]], dtype=np.float32),
        )
    )
    payload = MaskedImagePayload(
        data=image,
        mask=np.ones((2, 2), dtype=bool),
        metadata=ImagePayloadMetadata(
            channel_intensity_scales=(65535.0, 65535.0),
        ),
    )

    _, metadata_measurements = measure_colocalization(
        payload,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        dtype_config=DtypeConfig(),
    )
    _, explicit_measurements = measure_colocalization(
        image,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        scale_max=65535,
        dtype_config=DtypeConfig(),
    )

    assert metadata_measurements.costes_threshold_1 == explicit_measurements.costes_threshold_1
    assert metadata_measurements.costes_threshold_2 == explicit_measurements.costes_threshold_2


def test_legacy_cellprofiler_module_aliases_resolve_to_canonical_functions():
    assert canonical_module_name("MeasureCorrelation") == "MeasureColocalization"
    assert get_contract("MeasureCorrelation") == get_contract("MeasureColocalization")
    assert get_function("MeasureCorrelation") is get_function("MeasureColocalization")
    assert canonical_module_name("Erosion") == "ErodeImage"
    assert canonical_module_name("Dilation") == "DilateImage"
    assert get_contract("Erosion") == get_contract("ErodeImage")
    assert get_function("Erosion") is get_function("ErodeImage")


def test_export_to_spreadsheet_module_imports_cleanly():
    module = importlib.import_module(
        "benchmark.cellprofiler_library.functions.exporttospreadsheet"
    )

    assert module is not None


def test_absorbed_processing_contract_metadata_does_not_act_as_validator():
    image = np.ones((8, 8), dtype=np.float32)

    result, stats = correct_illumination_calculate(image, dtype_config=DtypeConfig())

    assert result.shape == image.shape
    assert stats.calculation_type == "regular"
    assert (
        correct_illumination_calculate.__processing_contract__
        is ProcessingContract.FLEXIBLE
    )
    assert opening.__processing_contract__ is ProcessingContract.PURE_2D


def test_illumination_functions_accept_cellprofiler_enum_literals():
    image = np.ones((8, 8), dtype=np.float32)

    illumination, stats = correct_illumination_calculate(
        image,
        intensity_choice="Regular",
        rescale_option="No",
        smoothing_method="No smoothing",
        dtype_config=DtypeConfig(),
    )
    corrected = correct_illumination_apply(
        np.stack((image, np.full_like(image, 0.25))),
        method="Subtract",
        truncate_low=False,
        truncate_high=False,
        dtype_config=DtypeConfig(),
    )

    assert illumination.shape == image.shape
    assert stats.calculation_type == "regular"
    assert stats.smoothing_method == "none"
    np.testing.assert_array_equal(
        corrected,
        np.full((1, 8, 8), 0.75, dtype=np.float32),
    )


def test_correct_illumination_fit_polynomial_matches_dense_design_matrix():
    from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
        _fit_polynomial_surface,
    )

    image = (np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0) ** 2
    mask = np.ones(image.shape, dtype=bool)
    mask[1::3, 2::4] = False
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    y = y / h - 0.5
    x = x / w - 0.5
    valid = mask.flatten()
    design = np.column_stack(
        [
            (x**2).flatten()[valid],
            (y**2).flatten()[valid],
            (x * y).flatten()[valid],
            x.flatten()[valid],
            y.flatten()[valid],
            np.ones(valid.sum()),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(
        design,
        image.flatten()[valid],
        rcond=None,
    )
    full_design = np.column_stack(
        [
            (x**2).flatten(),
            (y**2).flatten(),
            (x * y).flatten(),
            x.flatten(),
            y.flatten(),
            np.ones(h * w),
        ]
    )
    expected = (full_design @ coeffs).reshape(h, w)

    np.testing.assert_allclose(
        _fit_polynomial_surface(image, mask),
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_correct_illumination_background_uses_blockwise_minima():
    image = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1) / 100

    illumination, _ = correct_illumination_calculate(
        image,
        intensity_choice="Background",
        block_size=2,
        smoothing_method="No smoothing",
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )

    expected = np.array(
        [
            [0.01, 0.01, 0.03, 0.03],
            [0.01, 0.01, 0.03, 0.03],
            [0.09, 0.09, 0.11, 0.11],
            [0.09, 0.09, 0.11, 0.11],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_automatic_filter_size_matches_cellprofiler_source():
    from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
        AutomaticSmoothingFilterSizeStrategy,
        SmoothingFilterSizeRequest,
    )

    request = SmoothingFilterSizeRequest(
        image_shape=(1116, 1112),
        object_width=10,
        manual_filter_size=10,
    )

    assert AutomaticSmoothingFilterSizeStrategy().calculate(request) == 27.9


def test_correct_illumination_background_respects_image_mask():
    image = np.array(
        [
            [0.01, 0.99],
            [0.03, 0.04],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    illumination, _ = correct_illumination_calculate(
        MaskedImagePayload(data=image, mask=mask),
        intensity_choice="Background",
        block_size=2,
        smoothing_method="No smoothing",
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(image_payload_mask(illumination), mask)
    np.testing.assert_array_equal(
        image_payload_data(illumination),
        np.array(
            [
                [0.01, 0.0],
                [0.01, 0.01],
            ],
            dtype=np.float32,
        ),
    )


def test_correct_illumination_all_scope_averages_stack_before_smoothing():
    stack = np.stack(
        (
            np.full((4, 4), 0.25, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        )
    )

    illumination, stats = correct_illumination_calculate(
        stack,
        calculation_scope="All: First cycle",
        smoothing_method="No smoothing",
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )

    assert illumination.shape == (4, 4)
    np.testing.assert_array_equal(
        illumination,
        np.full((4, 4), 0.5, dtype=np.float32),
    )
    assert stats.mean_value == 0.5


def test_correct_illumination_median_smoothing_uses_local_rank_structuring_disk(
    monkeypatch,
):
    import skimage.filters

    from openhcs.constants.constants import MemoryType

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy

    calls = []

    def median(image, positional_footprint=None, **kwargs):
        calls.append(("median", image.dtype, positional_footprint, kwargs))
        return image

    monkeypatch.setattr(skimage.filters, "median", median)

    illumination, _ = correct_illumination_calculate(
        np.full((4, 4), 0.5, dtype=np.float32),
        smoothing_method="Median Filter",
        filter_size_method="Manually",
        manual_filter_size=2.35,
        rescale_option="No",
        dtype_config=DtypeConfig(),
        rank_median_backend_provider=CellProfilerBackendProvider.NATIVE,
    )

    assert len(calls) == 1
    assert calls[0][0:2] == ("median", np.dtype("uint16"))
    np.testing.assert_array_equal(
        calls[0][2],
        MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY).disk_footprint(1),
    )
    assert calls[0][3] == {"behavior": "rank"}
    expected = np.full((4, 4), 32767 / 65535, dtype=np.float32)
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_median_smoothing_fast_minimum_majority_path():
    image = np.ones((16, 16), dtype=np.float32)
    image[1::4, 1::4] = 0.25

    illumination, _ = correct_illumination_calculate(
        image,
        intensity_choice="Background",
        block_size=4,
        smoothing_method="Median Filter",
        filter_size_method="Manually",
        manual_filter_size=16,
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )

    expected = np.full((16, 16), np.uint16(0.25 * 65535) / 65535, dtype=np.float32)
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_median_smoothing_falls_back_when_minimum_not_majority():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(25, dtype=np.float32).reshape((5, 5)) / 24

    accelerated, _ = correct_illumination_calculate(
        image,
        smoothing_method="Median Filter",
        filter_size_method="Manually",
        manual_filter_size=2.35,
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )
    reference, _ = correct_illumination_calculate(
        image,
        smoothing_method="Median Filter",
        filter_size_method="Manually",
        manual_filter_size=2.35,
        rescale_option="No",
        dtype_config=DtypeConfig(),
        rank_median_backend_provider=CellProfilerBackendProvider.NATIVE,
    )

    np.testing.assert_array_equal(accelerated, reference)


def test_correct_illumination_convex_hull_smoothing_suppresses_sparse_spikes():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.zeros((7, 7), dtype=np.float32)
    image[1, 1] = 1.0
    image[1, 5] = 1.0
    image[5, 1] = 1.0

    illumination, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        rescale_option="No",
        convex_hull_backend_provider=CellProfilerBackendProvider.EXACT,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(illumination, np.zeros(image.shape, dtype=np.float32))
    assert illumination.dtype == np.float32


def test_correct_illumination_exact_convex_hull_matches_native_reference():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    rng = np.random.default_rng(123)
    image = np.zeros((30, 40), dtype=np.float32)
    rows = rng.integers(0, image.shape[0], 20)
    columns = rng.integers(0, image.shape[1], 20)
    values = rng.choice(np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32), 20)
    image[rows, columns] = values

    accelerated, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        rescale_option="No",
        convex_hull_backend_provider=CellProfilerBackendProvider.EXACT,
        dtype_config=DtypeConfig(),
    )
    reference, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        rescale_option="No",
        convex_hull_backend_provider=CellProfilerBackendProvider.NATIVE_EXACT,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(accelerated, reference)


def test_correct_illumination_convex_hull_default_uses_centrosome_backend():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(49, dtype=np.float32).reshape(7, 7) / 100

    illumination, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        filter_size_method="Manually",
        manual_filter_size=3,
        rescale_option="No",
        dtype_config=DtypeConfig(),
    )
    expected, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        filter_size_method="Manually",
        manual_filter_size=3,
        rescale_option="No",
        convex_hull_backend_provider=CellProfilerBackendProvider.CENTROSOME,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_convex_hull_legacy_fast_backend_is_explicit():
    from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(49, dtype=np.float32).reshape(7, 7) / 100

    illumination, _ = correct_illumination_calculate(
        image,
        smoothing_method="Convex Hull",
        filter_size_method="Manually",
        manual_filter_size=3,
        rescale_option="No",
        convex_hull_backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        dtype_config=DtypeConfig(),
    )

    expected = grey_dilation(
        maximum_filter(grey_erosion(image, size=3), size=3),
        size=3,
    ).astype(np.float32)
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_convex_hull_unregistered_backend_is_explicit_error():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    with pytest.raises(NotImplementedError, match="No CellProfiler"):
        correct_illumination_calculate(
            np.ones((4, 4), dtype=np.float32),
            smoothing_method="Convex Hull",
            convex_hull_backend_provider=CellProfilerBackendProvider.CUCIM,
            dtype_config=DtypeConfig(),
        )


def test_correct_illumination_strategy_registries_use_json_stable_keys():
    from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
        FilterSizeMethod,
        SmoothingFilterSizeStrategy,
        SmoothingMethod,
        SmoothingPlaneStrategy,
    )

    assert set(SmoothingFilterSizeStrategy.__registry__) == {
        method.value for method in FilterSizeMethod
    }
    assert set(SmoothingPlaneStrategy.__registry__) == {
        method.value for method in SmoothingMethod
    }
    assert all(isinstance(key, str) for key in SmoothingFilterSizeStrategy.__registry__)
    assert all(isinstance(key, str) for key in SmoothingPlaneStrategy.__registry__)
    assert type(
        SmoothingFilterSizeStrategy.for_method(FilterSizeMethod.AUTOMATIC)
    ) is SmoothingFilterSizeStrategy.__registry__[FilterSizeMethod.AUTOMATIC.value]
    assert type(SmoothingPlaneStrategy.for_method(SmoothingMethod.NONE)) is (
        SmoothingPlaneStrategy.__registry__[SmoothingMethod.NONE.value]
    )


def test_pure_2d_contract_wrapper_aggregates_tuple_outputs_per_slice():
    registry = OpenHCSRegistry()
    wrapped = registry.apply_contract_wrapper(
        correct_illumination_calculate,
        ProcessingContract.PURE_2D,
    )
    image = np.stack(
        (
            np.full((8, 8), 1.0, dtype=np.float32),
            np.full((8, 8), 2.0, dtype=np.float32),
        )
    )

    result, stats = wrapped(image, dtype_config=DtypeConfig())

    assert result.shape == image.shape
    assert len(stats) == 2
    assert [item.slice_index for item in stats] == [0, 1]
    assert all(item.mean_value > 0 for item in stats)


def test_unmix_colors_returns_one_output_per_stain_row():
    image = np.full((8, 9, 3), 0.5, dtype=np.float32)

    outputs = unmix_colors(
        image,
        stain_names=("Hematoxylin", "Eosin", "Custom"),
        custom_absorbances=(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.1, 0.2, 0.3),
        ),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(outputs, tuple)
    assert [output.shape for output in outputs] == [(8, 9), (8, 9), (8, 9)]
    assert all(output.dtype == np.float32 for output in outputs)
    assert unmix_colors.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_crop_preserves_hwc_color_image_domain() -> None:
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)

    cropped, mask, measurements = crop(
        image,
        removal_method=RemovalMethod.ALL,
        left_right_rectangle_positions=(2, 7),
        top_bottom_rectangle_positions=(1, 6),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(cropped, MaskedImagePayload)
    assert cropped.shape == (5, 5, 3)
    assert mask.shape == (8, 9)
    assert measurements.area_retained == 25
    np.testing.assert_array_equal(cropped.data, image[1:6, 2:7])
    np.testing.assert_array_equal(cropped.mask, np.ones((5, 5), dtype=bool))
    assert cropped.metadata.spatial_origin_yx == (1, 2)
    assert cropped.metadata.source_spatial_shape_yx == (8, 9)
    assert cropped.metadata.physical_border_edges_for_shape(cropped.shape[:2]) == (
        False,
        False,
        False,
        False,
    )


def test_crop_no_removal_returns_masked_zeroed_image_domain() -> None:
    image = np.ones((4, 5), dtype=np.float32)

    cropped, mask, measurements = crop(
        image,
        removal_method=RemovalMethod.NO,
        left_right_rectangle_positions=(1, 4),
        top_bottom_rectangle_positions=(1, 3),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(cropped, MaskedImagePayload)
    assert cropped.shape == image.shape
    assert measurements.area_retained == 6
    assert cropped.metadata.mask_defines_border is False
    np.testing.assert_array_equal(mask, np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=bool,
    ))
    np.testing.assert_array_equal(mask, cropped.mask)
    assert np.all(cropped.data[~mask] == 0)
    assert np.all(cropped.data[mask] == 1)
    assert cropped.metadata.spatial_origin_yx == (0, 0)
    assert cropped.metadata.source_spatial_shape_yx == image.shape
    assert cropped.metadata.physical_border_edges_for_shape(cropped.shape) == (
        True,
        True,
        True,
        True,
    )


def test_crop_previous_cropping_accepts_typed_mask_input() -> None:
    image = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    previous_mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=bool,
    )

    cropped, crop_mask, measurements = crop(
        image,
        mask_plane=previous_mask,
        crop_shape=CropShape.CROPPING,
        removal_method=RemovalMethod.EDGES,
        dtype_config=DtypeConfig(),
    )

    assert isinstance(cropped, MaskedImagePayload)
    np.testing.assert_array_equal(crop_mask, previous_mask)
    np.testing.assert_array_equal(cropped.data, image[1:3, 1:4])
    np.testing.assert_array_equal(cropped.mask, np.ones((2, 3), dtype=bool))
    assert measurements.area_retained == 6


def test_crop_objects_accepts_dense_label_stack_as_foreground_union() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    labels = np.zeros((2, 4, 5), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[1, 2, 3] = 2

    cropped, mask, measurements = crop(
        image,
        crop_shape=CropShape.OBJECTS,
        removal_method=RemovalMethod.NO,
        cropping_labels=labels,
        dtype_config=DtypeConfig(),
    )

    expected_mask = np.zeros((4, 5), dtype=bool)
    expected_mask[1, 1] = True
    expected_mask[2, 3] = True
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(cropped.mask, expected_mask)
    assert cropped.metadata.mask_defines_border is False
    assert np.all(cropped.data[~expected_mask] == 0)
    assert np.all(cropped.data[expected_mask] == 1)
    assert measurements.area_retained == 2


def test_measure_image_area_occupied_runs_mixed_rows():
    binary = np.zeros((5, 6), dtype=np.float32)
    binary[1:3, 1:4] = 1.0
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[2:4, 2:5] = 1

    retained, measurements = measure_image_area_occupied(
        binary,
        operand_choices=("binary_image", "objects"),
        input_names=("DNA", "Nuclei"),
        retained_image_names=(None, "OccupiedNuclei"),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert retained.shape == labels.shape
    assert [measurement.slice_index for measurement in measurements] == [0, 1]
    assert all(measurement.area_occupied == 6.0 for measurement in measurements)
    assert [measurement.source_image_name for measurement in measurements] == [
        "DNA",
        "Nuclei",
    ]
    assert measure_image_area_occupied.__processing_contract__ is (
        ProcessingContract.FLEXIBLE
    )


def test_measure_image_area_occupied_reduces_label_stacks_as_2d_planes():
    image = np.zeros((2, 5, 6), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 1:3, 1:4] = 1
    labels[1, 2:4, 2:5] = 1

    retained, measurements = measure_image_area_occupied(
        image,
        operand_choices=("objects",),
        input_names=("Nuclei",),
        retained_image_names=("OccupiedNuclei",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert retained.shape == labels.shape
    assert len(measurements) == 1
    assert measurements[0].area_occupied == 12.0
    assert measurements[0].total_area == 60.0
    assert measurements[0].perimeter > 0


def test_mask_image_applies_2d_object_mask_to_singleton_image_stack():
    image = np.ones((1, 5, 6), dtype=np.float32)
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 2:5] = 1

    masked = mask_image(
        image,
        labels,
        mask_source="objects",
        dtype_config=DtypeConfig(),
    )

    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.count_nonzero(image_payload_data(masked)[0]) == 9
    assert np.all(image_payload_data(masked)[0, labels == 0] == 0)
    assert np.array_equal(image_payload_mask(masked), (labels > 0)[np.newaxis, ...])


def test_mask_image_accepts_source_backed_singleton_image_plane():
    image = image_payload_with_context(
        np.ones((1, 5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="source.tif"),
    )
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 2:5] = 1

    masked = mask_image(
        image,
        labels,
        mask_source="objects",
        dtype_config=DtypeConfig(),
    )

    assert masked.shape == (1, 5, 6)
    assert np.count_nonzero(image_payload_data(masked)[0]) == 9
    assert np.array_equal(image_payload_mask(masked), (labels > 0)[np.newaxis, ...])


def test_mask_image_uses_aligned_mask_stack_planes():
    image = np.ones((2, 5, 6), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[0, 1:3, 1:3] = 1.0
    mask[1, 2:5, 3:6] = 1.0

    masked = mask_image(
        image,
        mask,
        mask_source="image",
        dtype_config=DtypeConfig(),
    )

    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.count_nonzero(image_payload_data(masked)[0]) == 4
    assert np.count_nonzero(image_payload_data(masked)[1]) == 9
    assert np.array_equal(image_payload_mask(masked), mask > 0)


def test_mask_image_projects_mask_stack_to_single_image_plane():
    image = np.ones((5, 6), dtype=np.float32)
    mask = np.zeros((2, 5, 6), dtype=np.int32)
    mask[0, 1:3, 1:3] = 1
    mask[1, 2:5, 3:6] = 2

    masked = mask_image(
        image,
        mask,
        mask_source="objects",
        dtype_config=DtypeConfig(),
    )

    expected_mask = np.any(mask > 0, axis=0)
    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.array_equal(image_payload_mask(masked), expected_mask)
    assert np.count_nonzero(image_payload_data(masked)) == np.count_nonzero(
        expected_mask
    )


def test_mask_image_projects_volume_mask_stack_to_image_planes():
    image = np.ones((3, 5, 6), dtype=np.float32)
    mask = np.zeros((2, 3, 5, 6), dtype=np.int32)
    mask[0, :, 1:3, 1:3] = 1
    mask[1, :, 2:5, 3:6] = 2

    masked = mask_image(
        image,
        mask,
        mask_source="objects",
        dtype_config=DtypeConfig(),
    )

    expected_mask = np.any(mask > 0, axis=0)
    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.array_equal(image_payload_mask(masked), expected_mask)
    assert np.count_nonzero(image_payload_data(masked)) == np.count_nonzero(
        expected_mask
    )


def test_mask_image_projects_resized_volume_mask_stack_to_image_planes():
    image = np.ones((3, 5, 6), dtype=np.float32)
    mask = np.zeros((2, 2, 3, 4), dtype=np.int32)
    mask[0, :, 1:, 1:] = 1
    mask[1, :, :2, :2] = 2

    masked = mask_image(
        image,
        mask,
        mask_source="objects",
        dtype_config=DtypeConfig(),
    )

    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert image_payload_mask(masked).shape == image.shape
    assert all(
        np.array_equal(image_payload_mask(masked)[0], image_payload_mask(masked)[index])
        for index in range(1, image.shape[0])
    )
    assert np.count_nonzero(image_payload_data(masked)) > 0


def test_mask_image_projects_flat_grouped_mask_planes_to_image_planes():
    image = np.ones((3, 5, 6), dtype=np.float32)
    mask = np.zeros((2, 3, 5, 6), dtype=np.float32)
    mask[0, 0, 1:3, 1:3] = 1.0
    mask[0, 1, 2:4, 2:4] = 1.0
    mask[1, 1, 0:2, 0:2] = 1.0
    flattened_mask = mask.reshape((-1, *mask.shape[-2:]))

    masked = mask_image(
        image,
        flattened_mask,
        mask_source="image",
        dtype_config=DtypeConfig(),
    )

    expected_mask = np.any(mask > 0, axis=0)
    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.array_equal(image_payload_mask(masked), expected_mask)


def test_relate_objects_aligns_parent_label_stack_to_child_plane():
    parent_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    child_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    output, relationships, measurements = relate_objects.__wrapped__(
        np.zeros_like(child_plane, dtype=np.float32),
        np.stack((parent_plane, parent_plane)),
        child_plane,
        calculate_distances=DistanceMethod.NONE,
    )

    assert output.shape == child_plane.shape
    assert relationships.parent_ids == (1, 2)
    assert relationships.child_ids == (1, 2)
    assert measurements.parent_object_count == 2
    assert measurements.child_object_count == 2


def test_relate_objects_numba_distance_measurements():
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels = np.zeros_like(parent_labels)
    child_labels[2, 2] = 1

    _output, relationships, measurements = relate_objects.__wrapped__(
        np.zeros_like(parent_labels, dtype=np.float32),
        parent_labels,
        child_labels,
        calculate_distances=DistanceMethod.BOTH,
    )

    assert relationships.parent_ids == (1,)
    assert relationships.child_ids == (1,)
    assert measurements.mean_centroid_distance == pytest.approx(0.0)
    assert measurements.mean_minimum_distance == pytest.approx(1.0)


def test_mask_image_combines_existing_image_mask_with_mask_input():
    image = np.ones((5, 6), dtype=np.float32)
    existing_mask = np.zeros_like(image, dtype=bool)
    existing_mask[1:5, 1:5] = True
    mask = np.zeros_like(image, dtype=np.float32)
    mask[0:3, 2:6] = 1.0

    masked = mask_image(
        MaskedImagePayload(data=image, mask=existing_mask),
        mask,
        mask_source="image",
        dtype_config=DtypeConfig(),
    )

    expected_mask = existing_mask & (mask > 0)
    assert isinstance(masked, MaskedImagePayload)
    assert np.array_equal(image_payload_mask(masked), expected_mask)
    assert np.count_nonzero(image_payload_data(masked)) == int(expected_mask.sum())


def test_align_returns_two_registered_images_and_shift_measurements():
    first = np.zeros((8, 8), dtype=np.float32)
    first[2:5, 2:5] = 1.0
    second = np.zeros_like(first)
    second[3:6, 2:5] = 1.0

    aligned_first, aligned_second, measurements = align(
        np.stack((first, second)),
        crop_mode="Keep size",
        dtype_config=DtypeConfig(),
    )

    assert image_payload_data(aligned_first).shape == first.shape
    assert image_payload_data(aligned_second).shape == second.shape
    assert image_payload_mask(aligned_first) is None
    assert image_payload_mask(aligned_second) is not None
    assert measurements[0] == AlignShiftMeasurement(
        slice_index=0,
        output_index=0,
        x_shift=0.0,
        y_shift=0.0,
    )
    assert measurements[1].output_index == 1
    assert measurements[1].x_shift == 0.0
    assert measurements[1].y_shift > 0.0
    assert align.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_align_applies_similar_shift_to_additional_images():
    first = np.zeros((8, 8), dtype=np.float32)
    first[2:5, 2:5] = 1.0
    second = np.zeros_like(first)
    second[3:6, 2:5] = 1.0
    additional = np.zeros_like(first)
    additional[4:7, 4:7] = 2.0

    aligned_first, aligned_second, aligned_additional, measurements = align(
        np.stack((first, second, additional)),
        crop_mode="Keep size",
        additional_alignment_modes=("Similarly",),
        dtype_config=DtypeConfig(),
    )

    assert image_payload_data(aligned_first).shape == first.shape
    assert image_payload_data(aligned_second).shape == second.shape
    assert image_payload_data(aligned_additional).shape == additional.shape
    assert len(measurements) == 3
    assert measurements[2].output_index == 2
    assert measurements[2].x_shift == measurements[1].x_shift
    assert measurements[2].y_shift == measurements[1].y_shift


def test_overlay_outlines_runs_mixed_image_and_object_rows():
    base = np.zeros((8, 8), dtype=np.float32)
    outline_image = np.zeros_like(base)
    outline_image[1:6, 1] = 1.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1

    output = overlay_outlines(
        np.stack((base, outline_image)),
        outline_source_kinds=("image", "objects"),
        outline_colors=("Red", "Green"),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (8, 8, 3)
    assert output[..., 0].max() > 0
    assert output[..., 1].max() > 0
    assert overlay_outlines.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_overlay_outlines_accepts_hex_color_literals():
    base = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1

    output = overlay_outlines(
        base,
        outline_source_kinds=("objects",),
        outline_colors=("#0800F7",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (8, 8, 3)
    assert output[..., 2].max() > 0.9
    assert output[..., 0].max() < 0.1


def test_overlay_outlines_accepts_secondary_named_color_literals():
    base = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1

    output = overlay_outlines(
        base,
        outline_source_kinds=("objects",),
        outline_colors=("Magenta",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (8, 8, 3)
    assert output[..., 0].max() > 0.9
    assert output[..., 1].max() < 0.1
    assert output[..., 2].max() > 0.9


def test_overlay_outlines_uses_cellprofiler_mark_boundaries_semantics():
    base = np.zeros((8, 8), dtype=np.float32)
    base[1:7, 1:7] = 0.25
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1

    output = overlay_outlines(
        base,
        line_mode="Inner",
        outline_source_kinds=("objects",),
        outline_colors=("Green",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )
    expected = skimage.segmentation.mark_boundaries(
        np.dstack((base, base, base)),
        labels,
        color=(0.0, 1.0, 0.0),
        mode="inner",
    ).astype(np.float32)

    np.testing.assert_array_equal(output, expected)


def test_overlay_outlines_runs_plane_stack_object_rows():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 2:5, 2:5] = 1
    labels[1, 3:6, 3:6] = 1

    output = overlay_outlines(
        image,
        outline_source_kinds=("objects",),
        outline_colors=("Green",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (2, 8, 8, 3)
    assert output[..., 1].max() > 0


def test_overlay_outlines_ignores_empty_label_planes():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)

    output = overlay_outlines(
        image,
        outline_source_kinds=("objects",),
        object_labels=(labels,),
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (2, 8, 8, 3)
    assert float(output.max()) == 0.0
