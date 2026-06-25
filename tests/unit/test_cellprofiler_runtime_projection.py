import numpy as np

from openhcs.interop.cellprofiler.runtime.projection import (
    RuntimePlaneImagePayloadStack,
)


def test_runtime_plane_projection_does_not_treat_hwc_color_slice_as_stack():
    payload = np.zeros((11, 13, 3), dtype=np.float32)

    assert not RuntimePlaneImagePayloadStack(payload).is_projectable


def test_runtime_plane_projection_accepts_nhwc_color_stack():
    payload = np.zeros((2, 11, 13, 3), dtype=np.float32)

    assert RuntimePlaneImagePayloadStack(payload).is_projectable
