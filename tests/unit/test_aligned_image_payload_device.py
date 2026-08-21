import numpy as np

from arraybridge import MemoryType

import openhcs.core.aligned_image_payload as aligned_image_payload
from openhcs.core.aligned_image_payload import ImagePayloadBundleContext


def test_image_bundle_stacks_on_the_payload_framework_device(monkeypatch) -> None:
    payloads = (
        np.zeros((3, 4), dtype=np.float32),
        np.ones((3, 4), dtype=np.float32),
    )
    observed = []
    monkeypatch.setattr(
        aligned_image_payload,
        "detect_memory_type",
        lambda _payload: MemoryType.CUPY.value,
    )
    monkeypatch.setattr(
        MemoryType,
        "device_id_of",
        lambda memory_type, _payload, _module=None: (
            7 if memory_type is MemoryType.CUPY else None
        ),
    )
    monkeypatch.setattr(
        aligned_image_payload,
        "stack_runtime_slices",
        lambda values, memory_type, device_id: observed.append(
            (tuple(values), memory_type, device_id)
        )
        or "stacked",
    )

    context = ImagePayloadBundleContext(payloads)

    assert context.compose_unmasked(payloads) == "stacked"
    assert observed == [(payloads, MemoryType.CUPY.value, 7)]
