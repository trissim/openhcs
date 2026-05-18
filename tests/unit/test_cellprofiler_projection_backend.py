from __future__ import annotations

import numpy as np

from openhcs.processing.backends.cellprofiler.projection import (
    ProjectionRequest,
    ProjectionStrategy,
    ProjectionType,
    make_projection,
)


def test_projection_strategies_materialize_float32_for_all_projection_types() -> None:
    stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 6.0], [9.0, 12.0]],
        ],
        dtype=np.float64,
    )

    for projection_type in ProjectionType:
        request = ProjectionRequest(
            image=stack,
            projection_type=projection_type,
            frequency=6.0,
        )
        result = ProjectionStrategy.for_projection_type(projection_type).apply(request)
        stats = request.stats(result)

        assert result.dtype == np.float32
        assert stats.projection_type == projection_type.value
        assert stats.input_slices == stack.shape[0]
        assert stats.output_min == float(np.min(result))
        assert stats.output_max == float(np.max(result))
        assert stats.output_mean == float(np.mean(result))


def test_make_projection_preserves_core_projection_values() -> None:
    stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 6.0], [9.0, 12.0]],
        ],
        dtype=np.float64,
    )

    expected = {
        ProjectionType.AVERAGE: np.mean(stack, axis=0),
        ProjectionType.MAXIMUM: np.max(stack, axis=0),
        ProjectionType.MINIMUM: np.min(stack, axis=0),
        ProjectionType.SUM: np.sum(stack, axis=0),
        ProjectionType.VARIANCE: np.var(stack.astype(np.float64), axis=0),
        ProjectionType.MASK: np.ones(stack.shape[1:], dtype=bool),
    }

    for projection_type, expected_result in expected.items():
        result, _stats = make_projection(stack, projection_type=projection_type)

        np.testing.assert_allclose(result, expected_result.astype(np.float32))
