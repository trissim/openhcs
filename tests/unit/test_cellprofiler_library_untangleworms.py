from __future__ import annotations

import numpy as np
import centrosome.cpmorphology

from benchmark.cellprofiler_library.functions.untangleworms import (
    OverlapStyle,
    untangle_worms,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    ObjectLabelPayload,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.processing.backends.cellprofiler.worm_geometry import (
    branchpoints,
    endpoints,
    skeletonize_worm_mask,
)


def test_untangle_worms_single_component_does_not_apply_cluster_min_path_length() -> None:
    image = np.zeros((24, 24), dtype=np.uint8)
    image[8:11, 4:20] = 1

    _image, measurements, _overlap, nonoverlap = untangle_worms.__wrapped__(
        image,
        overlap_style=OverlapStyle.WITHOUT_OVERLAP,
        min_worm_area=1.0,
        max_worm_area=1_000.0,
        min_path_length=100.0,
        max_path_length=200.0,
        cost_threshold=1_000.0,
        num_control_points=5,
        mean_angles=(),
    )

    assert len(measurements) == 1
    assert measurements[0]["object_name"] == "NonOverlappingWorms"
    assert measurements[0]["object_number"] == 1
    assert np.max(nonoverlap) == 1


def test_untangle_worms_labels_preserve_source_image_spatial_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((24, 24), dtype=np.uint8),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(5, 7),
                source_shape_yx=(40, 50),
            ),
        ),
    )
    image.data[8:11, 4:20] = 1

    _image, _measurements, _overlap, nonoverlap = untangle_worms.__wrapped__(
        image,
        overlap_style=OverlapStyle.WITHOUT_OVERLAP,
        min_worm_area=1.0,
        max_worm_area=1_000.0,
        min_path_length=100.0,
        max_path_length=200.0,
        cost_threshold=1_000.0,
        num_control_points=5,
        mean_angles=(),
    )

    assert isinstance(nonoverlap, ObjectLabelPayload)
    assert nonoverlap.spatial_origin_yx == (5, 7)
    assert nonoverlap.source_spatial_shape_yx == (40, 50)


def test_worm_geometry_matches_centrosome_for_interior_skeletons() -> None:
    rng = np.random.RandomState(11)

    for shape in ((8, 8), (16, 16), (24, 24)):
        for _index in range(8):
            mask = rng.rand(*shape) > 0.72
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False

            np.testing.assert_array_equal(
                skeletonize_worm_mask(mask),
                centrosome.cpmorphology.skeletonize(mask),
            )


def test_worm_lookup_geometry_matches_centrosome() -> None:
    rng = np.random.RandomState(13)

    for shape in ((8, 8), (16, 16)):
        for _index in range(8):
            skeleton = rng.rand(*shape) > 0.82

            np.testing.assert_array_equal(
                branchpoints(skeleton),
                centrosome.cpmorphology.branchpoints(skeleton),
            )
            np.testing.assert_array_equal(
                endpoints(skeleton),
                centrosome.cpmorphology.endpoints(skeleton),
            )
