"""Behavior gates retained after backend-leaf wrapper deletion."""

from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    )
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.processing.backends.cellprofiler._backend import (
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.alignment import (
    AlignCropModeStrategy,
    AlignExecution,
    AlignModule,
)


@pytest.mark.parametrize(
    ("image_count", "additional_modes", "message"),
    (
        (1, (), "at least two declared source image planes"),
        (2, ("Similarly",), "without extra images"),
        (3, ("Similarly", "Similarly"), "mode count must match"),
    ),
)
def test_align_execution_retains_input_and_additional_mode_invariants(
    image_count: int,
    additional_modes: tuple[str, ...],
    message: str,
) -> None:
    image = np.zeros((image_count, 4, 5), dtype=np.float32)
    execution = AlignExecution(
        image=ImagePayloadMetadata(
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=tuple(
                            f"/input/image-{index}.tif"
                            for index in range(image_count)
                        )
                    )
                ),
                plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            ).payload_with(image, None),
        method="Mutual Information",
        crop_mode="Keep size",
        additional_alignment_modes=additional_modes,
        alignment_backend_provider=DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    )

    with pytest.raises(ValueError, match=message):
        execution.execute()


@pytest.mark.parametrize(
    ("crop_mode", "expected"),
    (
        (
            AlignModule.CropMode.KEEP_SIZE,
            (((0, 0), (2, -1)), ((5, 6), (4, 7))),
        ),
        (
            AlignModule.CropMode.PAD_IMAGES,
            (((0, 1), (2, 0)), ((6, 7), (6, 7))),
        ),
        (
            AlignModule.CropMode.CROP_TO_ALIGNED_REGION,
            (((-2, 0), (0, -1)), ((3, 6), (3, 6))),
        ),
    ),
)
def test_align_crop_strategy_owns_geometry_algorithm(
    crop_mode: AlignModule.CropMode,
    expected: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ],
) -> None:
    result = AlignCropModeStrategy.for_crop_mode(crop_mode).apply(
        ((0, 0), (2, -1)),
        ((5, 6), (4, 7)),
    )

    assert result == expected
