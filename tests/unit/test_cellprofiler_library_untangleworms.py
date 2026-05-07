from __future__ import annotations

import numpy as np

from benchmark.cellprofiler_library.functions.untangleworms import (
    OverlapStyle,
    untangle_worms,
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
