from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openhcs.processing.backends.assemblers.self_supervised_stitcher import (
    self_supervised_stitcher_func,
)


def test_empty_stitcher_result_preserves_declared_return_shape() -> None:
    empty_tiles = torch.empty((0, 4, 5))

    positions = self_supervised_stitcher_func(empty_tiles)
    positions_with_homographies, homographies = self_supervised_stitcher_func(
        empty_tiles,
        return_homographies=True,
    )

    assert positions.shape == (1, 0, 2)
    assert positions_with_homographies.shape == positions.shape
    assert homographies.shape == (0, 3, 3)
