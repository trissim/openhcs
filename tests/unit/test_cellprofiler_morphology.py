import numpy as np
import pytest

from openhcs.constants.constants import MemoryType
from openhcs.core.config import DtypeConfig
from openhcs.core.memory import numpy
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler import morphology as morphology_module

from openhcs.processing.backends.cellprofiler.morphology import (
    CentrosomeNumpyMorphologyBackendStrategy,
    CellProfilerDeclumpMethod,
    CombineObjectsStrategy,
    MorphologyBackendStrategy,
    NumbaNumpyMorphologyBackendStrategy,
    NumpyMorphologyBackendStrategy,
    SparseBooleanCubicMapCoordinatesThreshold,
)


MORPHOLOGY = MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)


def test_combine_objects_strategies_are_registered_by_enum_value() -> None:
    expected = {
        "merge": "MergeCombineObjectsStrategy",
        "preserve": "PreserveCombineObjectsStrategy",
        "discard": "DiscardCombineObjectsStrategy",
        "segment": "SegmentCombineObjectsStrategy",
    }

    assert {
        key: type(CombineObjectsStrategy.for_method(key)).__name__
        for key in expected
    } == expected


def test_fill_labeled_holes_fills_enclosed_background_only() -> None:
    labels = np.zeros((5, 7), dtype=np.int32)
    labels[1:4, 1:4] = 4
    labels[2, 2] = 0
    labels[1:4, 5] = 9

    filled = MORPHOLOGY.fill_labeled_holes(labels)

    assert filled[2, 2] == 4
    np.testing.assert_array_equal(filled[:, 0], labels[:, 0])
    np.testing.assert_array_equal(filled[:, 6], labels[:, 6])


def test_fill_labeled_holes_honors_size_predicate() -> None:
    binary = np.ones((7, 7), dtype=bool)
    binary[2, 2] = False
    binary[4:6, 4:6] = False

    filled = MORPHOLOGY.fill_labeled_holes(
        binary,
        size_predicate=lambda size, _is_foreground: size < 2,
    )

    assert filled[2, 2]
    assert not filled[4, 4]


def test_fill_labeled_holes_handles_stacked_planes_planewise() -> None:
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[:, 1:5, 1:5] = 3
    labels[0, 2, 2] = 0
    labels[1, 2:4, 2:4] = 0

    filled = MORPHOLOGY.fill_labeled_holes_below_size(labels, 2)

    assert filled[0, 2, 2] == 3
    assert filled[1, 2, 2] == 0
    np.testing.assert_array_equal(
        filled[0],
        MORPHOLOGY.fill_labeled_holes_below_size(labels[0], 2),
    )
    np.testing.assert_array_equal(
        filled[1],
        MORPHOLOGY.fill_labeled_holes_below_size(labels[1], 2),
    )


def test_connected_components_handles_stacked_planes_planewise() -> None:
    mask = np.zeros((2, 5, 5), dtype=bool)
    mask[0, 1:3, 1:3] = True
    mask[1, 1:3, 1:3] = True
    mask[1, 4, 4] = True

    labels, count = MORPHOLOGY.connected_components(mask)

    assert count == 3
    assert labels[0, 1, 1] == 1
    assert labels[1, 1, 1] == 2
    assert labels[1, 4, 4] == 3
    assert labels[0, 0, 0] == 0


def test_smooth_image_for_declumping_handles_stacked_planes_planewise() -> None:
    image = np.zeros((2, 9, 9), dtype=np.float32)
    image[0, 4, 4] = 1.0
    image[1, 2:5, 2:5] = 1.0
    mask = image > 0

    smoothed = MORPHOLOGY.smooth_image_for_declumping(image, mask, 3.0)

    np.testing.assert_allclose(
        smoothed[0],
        MORPHOLOGY.smooth_image_for_declumping(image[0], mask[0], 3.0),
    )
    np.testing.assert_allclose(
        smoothed[1],
        MORPHOLOGY.smooth_image_for_declumping(image[1], mask[1], 3.0),
    )


def test_declumping_seed_points_handles_stacked_planes_planewise() -> None:
    image = np.zeros((2, 9, 9), dtype=np.float32)
    labels = np.zeros((2, 9, 9), dtype=np.int32)
    labels[:, 1:8, 1:8] = 1
    image[0, 4, 4] = 10.0
    image[1, 3, 3] = 10.0
    image[1, 5, 5] = 9.0
    footprint = MORPHOLOGY.disk_footprint(1)

    seeds = MORPHOLOGY.declumping_seed_points(image, labels, footprint, 1.0)

    np.testing.assert_array_equal(
        seeds[0],
        MORPHOLOGY.declumping_seed_points(image[0], labels[0], footprint, 1.0),
    )
    np.testing.assert_array_equal(
        seeds[1],
        MORPHOLOGY.declumping_seed_points(image[1], labels[1], footprint, 1.0),
    )


def test_local_maxima_by_label_isolates_adjacent_labels() -> None:
    image = np.array(
        [
            [0.0, 3.0, 0.0, 9.0],
            [0.0, 2.0, 0.0, 1.0],
        ]
    )
    labels = np.array(
        [
            [0, 1, 0, 2],
            [0, 1, 0, 2],
        ],
        dtype=np.int32,
    )

    maxima = MORPHOLOGY.local_maxima_by_label(
        image,
        labels,
        MORPHOLOGY.disk_footprint(1),
    )

    expected = np.array(
        [
            [False, True, False, True],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(maxima, expected)


def test_local_maxima_by_label_handles_sparse_label_ids() -> None:
    from scipy import ndimage as ndi

    image = np.zeros((8, 9), dtype=float)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 17
    labels[4:7, 5:8] = 203
    image[2, 2] = 5.0
    image[5, 6] = 7.0
    footprint = MORPHOLOGY.disk_footprint(1)

    maxima = MORPHOLOGY.local_maxima_by_label(
        image,
        labels,
        footprint,
    )
    expected = np.zeros(labels.shape, dtype=bool)
    for label_id in (17, 203):
        label_mask = labels == label_id
        masked_image = np.where(label_mask, image, -np.inf)
        local_max = ndi.maximum_filter(
            masked_image,
            footprint=footprint,
            mode="constant",
            cval=-np.inf,
        )
        expected |= label_mask & (image == local_max)

    np.testing.assert_array_equal(maxima, expected)


def test_block_labels_match_cellprofiler_scaled_partitioning() -> None:
    from centrosome.cpmorphology import block

    for image_shape, block_size in (
        ((5, 5), 3),
        ((10, 12), 3),
        ((10, 12), 4),
        ((835, 1255), 40),
    ):
        labels, indexes = MORPHOLOGY.block_labels(image_shape, block_size)
        expected_labels, expected_indexes = block(
            image_shape,
            (block_size, block_size),
        )

        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(indexes, expected_indexes)


def test_numba_blockwise_minimum_matches_scipy_backend() -> None:
    scipy_backend = NumpyMorphologyBackendStrategy()
    numba_backend = NumbaNumpyMorphologyBackendStrategy()
    image = (np.arange(10 * 12, dtype=np.float32).reshape(10, 12) + 1) / 100
    image[2:8, 3:9] = image[2:8, 3:9][::-1, ::-1]
    mask = np.ones(image.shape, dtype=bool)
    mask[1:4, 2:5] = False
    mask[8:, 9:] = False

    expected = scipy_backend.blockwise_minimum(image, mask, 3)
    observed = numba_backend.blockwise_minimum(image, mask, 3)

    np.testing.assert_array_equal(observed, expected)


def test_numba_blockwise_minimum_handles_color_planes() -> None:
    scipy_backend = NumpyMorphologyBackendStrategy()
    numba_backend = NumbaNumpyMorphologyBackendStrategy()
    base = np.arange(8 * 9, dtype=np.float32).reshape(8, 9) / 50
    image = np.stack((base, base[::-1], base[:, ::-1]), axis=2)
    mask = np.ones(base.shape, dtype=bool)
    mask[:2, :3] = False

    expected = scipy_backend.blockwise_minimum(image, mask, 4)
    observed = numba_backend.blockwise_minimum(image, mask, 4)

    np.testing.assert_array_equal(observed, expected)


def test_explicit_centrosome_block_labels_match_centrosome_partitioning() -> None:
    from centrosome.cpmorphology import block

    centrosome_backend = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    labels, indexes = centrosome_backend.block_labels((10, 12), 3)
    expected_labels, expected_indexes = block((10, 12), (3, 3))

    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(indexes, expected_indexes)


def test_declumping_footprint_matches_cellprofiler_strel_disk() -> None:
    footprint = MORPHOLOGY.declumping_suppression_footprint(
        2,
        min_diameter=1,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    )

    np.testing.assert_array_equal(
        footprint,
        np.ones((3, 3), dtype=bool),
    )


def test_declumping_footprint_keeps_noninteger_disk_geometry() -> None:
    footprint = MORPHOLOGY.declumping_suppression_footprint(
        4,
        min_diameter=5,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    )

    np.testing.assert_array_equal(footprint, MORPHOLOGY.disk_footprint(3.5))


def test_declumping_footprint_uses_cellprofiler_suppression_disk() -> None:
    shape_footprint = MORPHOLOGY.declumping_suppression_footprint(
        4,
        min_diameter=4,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    )
    intensity_footprint = MORPHOLOGY.declumping_suppression_footprint(
        4,
        min_diameter=4,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
    )

    assert shape_footprint.shape == (7, 7)
    assert intensity_footprint.shape == (7, 7)
    y, x = np.ogrid[-3:4, -3:4]
    expected = (x * x + y * y) <= 3.5 * 3.5
    np.testing.assert_array_equal(shape_footprint, expected)
    np.testing.assert_array_equal(intensity_footprint, expected)


def test_declumping_smoothing_profile_matches_cellprofiler_source() -> None:
    shape_kernel = MORPHOLOGY.declumping_smoothing_kernel(
        4,
        declump_method=CellProfilerDeclumpMethod.SHAPE,
    )
    intensity_below_minimum_kernel = MORPHOLOGY.declumping_smoothing_kernel(
        4,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
        suppress_size=4,
        min_diameter=5,
    )
    intensity_at_minimum_kernel = MORPHOLOGY.declumping_smoothing_kernel(
        4,
        declump_method=CellProfilerDeclumpMethod.INTENSITY,
        suppress_size=4,
        min_diameter=4,
    )

    np.testing.assert_array_equal(intensity_below_minimum_kernel, shape_kernel)
    np.testing.assert_array_equal(intensity_at_minimum_kernel, shape_kernel)


def test_numba_convex_hull_image_matches_native_provider() -> None:
    native = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    numba_backend = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )
    mask = np.zeros((7, 7), dtype=bool)
    mask[1, 1] = True
    mask[1, 5] = True
    mask[5, 1] = True

    assert (
        NumbaNumpyMorphologyBackendStrategy.convex_hull_image
        is not NumpyMorphologyBackendStrategy.convex_hull_image
    )
    np.testing.assert_array_equal(
        numba_backend.convex_hull_image(mask),
        native.convex_hull_image(mask),
    )


def test_numba_convex_hull_image_matches_native_provider_for_sparse_masks() -> None:
    native = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    numba_backend = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )
    rng = np.random.default_rng(123)

    for shape in ((3, 3), (5, 7), (12, 13), (24, 31)):
        for _ in range(25):
            mask = rng.random(shape) < 0.15
            if not mask.any():
                mask[0, 0] = True
            np.testing.assert_array_equal(
                numba_backend.convex_hull_image(mask),
                native.convex_hull_image(mask),
            )


def test_shrink_components_to_seed_points_returns_one_seed_per_component() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3, 4] = True

    seeds = MORPHOLOGY.shrink_components_to_seed_points(mask)

    assert int(seeds.sum()) == 2
    assert seeds[3, 4]


def test_shrink_components_to_seed_points_matches_centrosome_binary_shrink() -> None:
    from centrosome.cpmorphology import binary_shrink

    mask = np.zeros((12, 13), dtype=bool)
    mask[1:4, 2:5] = True
    mask[1, 5] = True
    mask[8:11, 9:12] = True
    mask[3, 10] = True
    mask[4, 11] = True

    seeds = MORPHOLOGY.shrink_components_to_seed_points(mask)

    expected = np.asarray(binary_shrink(mask), dtype=bool)
    np.testing.assert_array_equal(seeds, expected)


def test_declumping_seed_points_matches_legacy_resize_path() -> None:
    image = np.zeros((18, 20), dtype=float)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:8, 1:9] = 1
    labels[9:17, 10:19] = 2
    image[3, 4] = 8.0
    image[5, 6] = 7.0
    image[12, 14] = 9.0
    image[14, 16] = 6.0
    footprint = MORPHOLOGY.disk_footprint(1)
    resize_factor = 0.5

    expected = LegacyDeclumpingSeedPointsAuthority(
        image,
        labels,
        footprint,
        resize_factor,
    ).execute()

    seeds = MORPHOLOGY.declumping_seed_points(
        image,
        labels,
        footprint,
        resize_factor,
    )

    np.testing.assert_array_equal(seeds, expected)


def test_declumping_integer_lattice_fast_path_matches_legacy_resize_path() -> None:
    rng = np.random.default_rng(123)
    footprint = MORPHOLOGY.disk_footprint(2)
    labels = np.zeros((31, 29), dtype=np.int32)
    labels[1:15, 2:17] = 10
    labels[16:30, 12:28] = 30
    labels[6:10, 20:25] = 40

    for step in (2, 3, 4):
        image = rng.random(labels.shape)
        image[labels == 0] = 0.0
        resize_factor = 1.0 / step

        expected = LegacyDeclumpingSeedPointsAuthority(
            image,
            labels,
            footprint,
            resize_factor,
        ).execute()
        seeds = MORPHOLOGY.declumping_seed_points(
            image,
            labels,
            footprint,
            resize_factor,
        )

        np.testing.assert_array_equal(seeds, expected)


def test_sparse_boolean_cubic_declumping_resize_matches_dense_scipy() -> None:
    from scipy import ndimage as ndi

    source = np.zeros((84, 100), dtype=bool)
    source[12, 17] = True
    source[41, 53] = True
    source[73, 88] = True
    target_shape = (1000, 1200)
    divisor = 1000.0 / 84.0

    coordinates = morphology_module._declumping_resize_coordinates(
        target_shape,
        divisor,
    )
    expected = ndi.map_coordinates(source.astype(float), coordinates) > 0.5

    resized = SparseBooleanCubicMapCoordinatesThreshold(
        source,
        target_shape,
        divisor,
    ).execute()

    np.testing.assert_array_equal(resized, expected)


def test_declumping_downsample_grid_uses_cellprofiler_pixel_origins() -> None:
    image = np.zeros((6, 6), dtype=float)
    labels = np.ones(image.shape, dtype=np.int32)
    image[0, 0] = 1.0
    image[2, 2] = 0.8
    footprint = np.ones((3, 3), dtype=bool)

    seeds = MORPHOLOGY.declumping_seed_points(
        image,
        labels,
        footprint,
        0.5,
    )

    np.testing.assert_array_equal(np.argwhere(seeds), np.array([[0, 0]]))


def test_numba_morphology_backend_matches_native_seed_extraction() -> None:
    native = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    numba_backend = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )
    rng = np.random.default_rng(42)
    labels = np.zeros((24, 22), dtype=np.int32)
    labels[1:12, 1:11] = 5
    labels[13:23, 8:21] = 19
    labels[5:10, 15:20] = 33
    image = rng.random(labels.shape)
    image[labels == 0] = 0.0
    footprint = native.disk_footprint(2)

    native_maxima = native.local_maxima_by_label(image, labels, footprint)
    numba_maxima = numba_backend.local_maxima_by_label(image, labels, footprint)
    native_seeds = native.shrink_components_to_seed_points(native_maxima)
    numba_seeds = numba_backend.shrink_components_to_seed_points(numba_maxima)
    native_relabeled, native_count = native.relabel_sequential(labels)
    numba_relabeled, numba_count = numba_backend.relabel_sequential(labels)

    assert type(numba_backend) is NumbaNumpyMorphologyBackendStrategy
    np.testing.assert_array_equal(numba_maxima, native_maxima)
    np.testing.assert_array_equal(numba_seeds, native_seeds)
    assert numba_count == native_count
    np.testing.assert_array_equal(numba_relabeled, native_relabeled)


def test_relabel_sequential_compacts_positive_labels() -> None:
    labels = np.array([[0, 7, 7], [3, 0, 9]], dtype=np.int32)

    relabeled, count = MORPHOLOGY.relabel_sequential(labels)

    assert count == 3
    np.testing.assert_array_equal(
        relabeled,
        np.array([[0, 2, 2], [1, 0, 3]], dtype=np.int32),
    )


def test_relabel_sequential_compacts_stacked_labels_globally() -> None:
    labels = np.array(
        [
            [[0, 7, 7], [3, 0, 0]],
            [[0, 9, 0], [11, 11, 0]],
        ],
        dtype=np.int32,
    )

    relabeled, count = MORPHOLOGY.relabel_sequential(labels)

    assert count == 4
    np.testing.assert_array_equal(
        relabeled,
        np.array(
            [
                [[0, 2, 2], [1, 0, 0]],
                [[0, 3, 0], [4, 4, 0]],
            ],
            dtype=np.int32,
        ),
    )


def test_morphology_backend_resolves_from_openhcs_memory_contract() -> None:
    @numpy
    def local_numpy_function(image):
        return image

    strategy = MorphologyBackendStrategy.for_callable(local_numpy_function)

    assert strategy.memory_type is MemoryType.NUMPY


def test_specialized_morphology_backends_are_explicit_opt_in() -> None:
    default_strategy = MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)
    native_strategy = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    centrosome_strategy = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        prefer_centrosome=True,
    )

    assert type(default_strategy) is NumbaNumpyMorphologyBackendStrategy
    assert type(native_strategy) is NumpyMorphologyBackendStrategy
    assert type(centrosome_strategy) is CentrosomeNumpyMorphologyBackendStrategy


def test_morph_convex_hull_routes_through_morphology_backend() -> None:
    from benchmark.cellprofiler_library.functions.morph import MorphOperation, morph

    image = np.zeros((7, 7), dtype=np.float32)
    image[1, 1] = 1
    image[1, 5] = 1
    image[5, 1] = 1
    expected = MORPHOLOGY.convex_hull_image(image > 0).astype(np.float32)

    result = morph(
        image,
        operation=MorphOperation.CONVEX_HULL,
        morphology_backend_provider=CellProfilerBackendProvider.NUMBA,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_array_equal(result, expected)


def test_fill_objects_convex_hull_routes_through_morphology_backend() -> None:
    from benchmark.cellprofiler_library.functions.fillobjects import FillMode, fill_objects

    image = np.zeros((7, 7), dtype=np.float32)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1, 1] = 4
    labels[1, 5] = 4
    labels[5, 1] = 4

    _, result = fill_objects(
        image,
        labels,
        mode=FillMode.CONVEX_HULL,
        morphology_backend_provider=CellProfilerBackendProvider.NUMBA,
        dtype_config=DtypeConfig(),
    )
    expected_mask = MORPHOLOGY.convex_hull_image(labels == 4)

    np.testing.assert_array_equal(result == 4, expected_mask)


def test_split_or_merge_convex_hull_routes_through_morphology_backend() -> None:
    from benchmark.cellprofiler_library.functions.splitormergeobjects import (
        MergeMethod,
        Operation,
        OutputObjectType,
        split_or_merge_objects,
    )

    image = np.zeros((7, 7), dtype=np.float32)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1, 1] = 1
    labels[1, 5] = 2
    labels[5, 1] = 3
    parent_labels = np.zeros_like(labels)
    parent_labels[labels > 0] = 9

    _, _, result = split_or_merge_objects(
        image,
        labels,
        operation=Operation.MERGE,
        merge_method=MergeMethod.PER_PARENT,
        output_object_type=OutputObjectType.CONVEX_HULL,
        parent_labels=parent_labels,
        morphology_backend_provider=CellProfilerBackendProvider.NUMBA,
        dtype_config=DtypeConfig(),
    )
    expected_mask = MORPHOLOGY.convex_hull_image(labels > 0)

    assert result.max() == 1
    np.testing.assert_array_equal(result > 0, expected_mask)


def test_split_or_merge_per_parent_requires_parent_labels() -> None:
    from benchmark.cellprofiler_library.functions.splitormergeobjects import (
        MergeMethod,
        Operation,
        split_or_merge_objects,
    )

    with pytest.raises(ValueError, match="parent_labels are required"):
        split_or_merge_objects(
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.int32),
            operation=Operation.MERGE,
            merge_method=MergeMethod.PER_PARENT,
            dtype_config=DtypeConfig(),
        )


def test_morphology_backend_unregistered_provider_is_explicit_error() -> None:
    from benchmark.cellprofiler_library.functions.morph import MorphOperation, morph

    with pytest.raises(NotImplementedError, match="cucim"):
        morph(
            np.ones((3, 3), dtype=np.float32),
            operation=MorphOperation.CONVEX_HULL,
            morphology_backend_provider=CellProfilerBackendProvider.CUCIM,
            dtype_config=DtypeConfig(),
        )


class LegacyDeclumpingSeedPointsAuthority:
    """Dense SciPy reference for CellProfiler declumping seed extraction tests."""

    def __init__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> None:
        self.image = image
        self.labels = labels
        self.footprint = footprint
        self.image_resize_factor = image_resize_factor

    def execute(self) -> np.ndarray:
        from scipy import ndimage as ndi

        low_res_shape = np.maximum(
            1,
            np.ceil(np.asarray(self.image.shape) * self.image_resize_factor),
        ).astype(int)
        low_res_coordinates = np.mgrid[
            0 : low_res_shape[0],
            0 : low_res_shape[1],
        ].astype(float) / self.image_resize_factor
        low_res_image = ndi.map_coordinates(self.image, low_res_coordinates)
        low_res_labels = ndi.map_coordinates(
            self.labels,
            low_res_coordinates,
            order=0,
        ).astype(self.labels.dtype)
        expected = MORPHOLOGY.local_maxima_by_label(
            low_res_image,
            low_res_labels,
            self.footprint,
        )
        expected[low_res_image <= 0] = 0
        inverse_resize_factor = float(self.image.shape[0]) / float(expected.shape[0])
        high_res_coordinates = (
            np.mgrid[0 : self.image.shape[0], 0 : self.image.shape[1]].astype(float)
            / inverse_resize_factor
        )
        expected = ndi.map_coordinates(expected.astype(float), high_res_coordinates) > 0.5
        return MORPHOLOGY.shrink_components_to_seed_points(expected)
