from __future__ import annotations

import numpy as np

from openhcs.runtime import viewer_shared_memory


class _SharedMemoryProbe:
    def __init__(self, source: np.ndarray) -> None:
        self._name = "/sender-owned"
        self.buf = bytearray(source.tobytes())
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_windows_attachment_does_not_start_posix_resource_tracker(monkeypatch) -> None:
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    memory = _SharedMemoryProbe(source)
    unregister_calls = []
    monkeypatch.setattr(viewer_shared_memory, "_USE_POSIX", False)
    monkeypatch.setattr(
        viewer_shared_memory.shared_memory,
        "SharedMemory",
        lambda *, name: memory,
    )
    monkeypatch.setattr(
        viewer_shared_memory.resource_tracker,
        "unregister",
        lambda *args: unregister_calls.append(args),
    )

    copied = viewer_shared_memory.SenderOwnedSharedMemoryAttachment.copy_array(
        name="sender-owned",
        shape=source.shape,
        dtype=source.dtype,
    )

    np.testing.assert_array_equal(copied, source)
    assert unregister_calls == []
    assert memory.closed


def test_posix_attachment_releases_receiver_resource_tracking(monkeypatch) -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    memory = _SharedMemoryProbe(source)
    unregister_calls = []
    monkeypatch.setattr(viewer_shared_memory, "_USE_POSIX", True)
    monkeypatch.setattr(
        viewer_shared_memory.shared_memory,
        "SharedMemory",
        lambda *, name: memory,
    )
    monkeypatch.setattr(
        viewer_shared_memory.resource_tracker,
        "unregister",
        lambda *args: unregister_calls.append(args),
    )

    copied = viewer_shared_memory.SenderOwnedSharedMemoryAttachment.copy_array(
        name="sender-owned",
        shape=source.shape,
        dtype=source.dtype,
    )

    np.testing.assert_array_equal(copied, source)
    assert unregister_calls == [(memory._name, "shared_memory")]
    assert memory.closed
