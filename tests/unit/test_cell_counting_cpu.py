import numpy as np

from openhcs.processing.backends.analysis.cell_counting_common import DetectionMethod
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    count_cells_multi_channel,
)


def test_count_cells_multi_channel_accepts_single_channel_result_contract():
    image_stack = np.zeros((2, 32, 32), dtype=np.uint16)
    image_stack[0, 10, 10] = 5000
    image_stack[1, 11, 11] = 5000

    output_stack, results = count_cells_multi_channel(
        image_stack,
        0,
        1,
        chan_1_method=DetectionMethod.BLOB_LOG,
        chan_2_method=DetectionMethod.BLOB_LOG,
        chan_1_threshold=0.01,
        chan_2_threshold=0.01,
        chan_1_enable_preprocessing=False,
        chan_2_enable_preprocessing=False,
    )

    assert output_stack.shape == image_stack.shape
    assert len(results) == 1
