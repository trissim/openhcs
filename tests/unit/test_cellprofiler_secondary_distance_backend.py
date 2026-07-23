import numpy as np

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    NumbaSecondaryDistanceTransformBackendStrategy,
    NumpySecondaryDistanceTransformBackendStrategy,
    SecondaryDistanceTransformBackendStrategy,
)


def test_distance_n_default_matches_cellprofiler_scipy_tie_breaking() -> None:
    labels = np.array(
        [
            [0, 1, 0],
            [2, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    backend = SecondaryDistanceTransformBackendStrategy.for_memory_type(
        MemoryType.NUMPY
    )

    assert type(backend) is NumpySecondaryDistanceTransformBackendStrategy
    np.testing.assert_array_equal(
        backend.nearest_label_expansion(labels, max_distance=3.0),
        np.array(
            [
                [2, 1, 1],
                [2, 2, 1],
                [2, 2, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_distance_n_numba_backend_remains_explicitly_selectable() -> None:
    backend = SecondaryDistanceTransformBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )

    assert type(backend) is NumbaSecondaryDistanceTransformBackendStrategy
