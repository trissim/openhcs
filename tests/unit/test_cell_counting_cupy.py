import numpy as np

from arraybridge import MemoryType

from openhcs.processing.backends.analysis.cell_counting_cupy import (
    _watershed_with_cpu_partition,
)


def test_watershed_partition_roundtrips_through_declared_cupy_device(
    monkeypatch,
) -> None:
    restored_device_ids = []
    monkeypatch.setattr(
        MemoryType,
        "device_id_of",
        lambda memory_type, _value: 3
        if memory_type is MemoryType.CUPY
        else None,
    )
    monkeypatch.setattr(
        MemoryType,
        "to_numpy",
        lambda _memory_type, value: np.asarray(value),
    )
    monkeypatch.setattr(
        MemoryType,
        "from_numpy",
        lambda _memory_type, value, device_id: (
            restored_device_ids.append(device_id),
            value,
        )[1],
    )
    image = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    markers = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
    mask = np.ones_like(markers, dtype=bool)

    labels = _watershed_with_cpu_partition(image, markers, mask=mask)

    assert labels.shape == image.shape
    assert restored_device_ids == [3]
