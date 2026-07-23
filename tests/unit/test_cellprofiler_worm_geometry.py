from __future__ import annotations

from unittest.mock import patch

import centrosome.cpmorphology
import numpy as np

from openhcs.processing.backends.cellprofiler.worm_geometry import (
    rebuild_worm_from_control_points_approx,
)


def test_fractional_control_points_match_canonical_centrosome_pixels() -> None:
    control_points = np.array(
        [
            [10.2, 10.2],
            [10.8, 10.8],
            [11.2, 13.2],
        ]
    )
    radii = np.array([2.0, 1.0, 2.0])
    expected_pixels = np.array(
        [
            [8, 10],
            [9, 9],
            [9, 10],
            [9, 11],
            [9, 12],
            [10, 8],
            [10, 9],
            [10, 10],
            [10, 11],
            [10, 12],
            [10, 13],
            [11, 9],
            [11, 10],
            [11, 11],
            [11, 12],
            [11, 13],
            [11, 14],
            [12, 10],
            [12, 12],
            [12, 13],
        ]
    )

    canonical_get_line_pts = centrosome.cpmorphology.get_line_pts
    with patch.object(
        centrosome.cpmorphology,
        "get_line_pts",
        wraps=canonical_get_line_pts,
    ) as get_line_pts:
        rows, columns = rebuild_worm_from_control_points_approx(
            control_points,
            radii,
            (24, 24),
        )

    get_line_pts.assert_called_once()
    for actual, expected in zip(
        get_line_pts.call_args.args,
        (
            control_points[:-1, 0],
            control_points[:-1, 1],
            control_points[1:, 0],
            control_points[1:, 1],
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    actual_pixels = np.column_stack((rows, columns))
    row_major_order = np.lexsort((actual_pixels[:, 1], actual_pixels[:, 0]))
    np.testing.assert_array_equal(actual_pixels[row_major_order], expected_pixels)
