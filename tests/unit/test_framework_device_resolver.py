import pytest

from arraybridge import MemoryType

from openhcs.core.compiled_step_plan import FrameworkDeviceAssignment
from openhcs.core.framework_device_resolver import FrameworkDeviceResolver
from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_empty_footprint_does_not_import_frameworks(monkeypatch) -> None:
    monkeypatch.setattr(
        MemoryType,
        "import_if_installed",
        lambda _memory_type: (_ for _ in ()).throw(
            AssertionError("empty footprint imported a GPU runtime")
        ),
    )

    assert FrameworkDeviceResolver().resolve(frozenset()).bindings == ()


def test_cpu_only_authority_rejects_gpu_footprint_before_import(monkeypatch) -> None:
    monkeypatch.setenv(OpenHCSProcessEnvironment.cpu_only_key, "true")
    monkeypatch.setattr(
        MemoryType,
        "import_if_installed",
        lambda _memory_type: (_ for _ in ()).throw(
            AssertionError("CPU-only resolution imported a GPU runtime")
        ),
    )

    with pytest.raises(ValueError, match="GPU imports are disabled"):
        FrameworkDeviceResolver().resolve(frozenset({MemoryType.TORCH}))


def test_framework_namespaces_receive_independent_local_devices(monkeypatch) -> None:
    modules = {
        MemoryType.CUPY: object(),
        MemoryType.TORCH: object(),
    }
    available = {
        MemoryType.CUPY: (2, 3),
        MemoryType.TORCH: (5,),
    }
    monkeypatch.delenv(OpenHCSProcessEnvironment.cpu_only_key, raising=False)
    monkeypatch.delenv(
        OpenHCSProcessEnvironment.subprocess_no_gpu_key,
        raising=False,
    )
    monkeypatch.delenv(
        OpenHCSProcessEnvironment.polystore_subprocess_no_gpu_key,
        raising=False,
    )
    monkeypatch.setattr(
        MemoryType,
        "import_if_installed",
        lambda memory_type: modules[memory_type],
    )
    monkeypatch.setattr(
        MemoryType,
        "available_device_ids",
        lambda memory_type, _module=None: available[memory_type],
    )

    assignment = FrameworkDeviceResolver(preferred_device_id=3).resolve(
        frozenset(modules)
    )

    assert assignment.device_id_for(MemoryType.CUPY) == 3
    assert assignment.device_id_for(MemoryType.TORCH) == 5


def test_compiled_cleanup_visits_only_assigned_framework_devices(monkeypatch) -> None:
    cleaned = []
    monkeypatch.setattr(
        MemoryType,
        "cleanup_loaded",
        lambda memory_type, device_id=None: cleaned.append(
            (memory_type, device_id)
        ),
    )
    assignment = FrameworkDeviceAssignment.from_mapping(
        {
            MemoryType.CUPY: 2,
            MemoryType.TORCH: 5,
        }
    )

    assignment.cleanup_loaded()

    assert cleaned == [
        (MemoryType.CUPY, 2),
        (MemoryType.TORCH, 5),
    ]


def test_compiled_cleanup_attempts_every_binding_before_reporting_failures(
    monkeypatch,
) -> None:
    cleaned = []

    def cleanup(memory_type, device_id=None):
        cleaned.append((memory_type, device_id))
        if memory_type is MemoryType.CUPY:
            raise RuntimeError("cupy cleanup failed")

    monkeypatch.setattr(MemoryType, "cleanup_loaded", cleanup)
    assignment = FrameworkDeviceAssignment.from_mapping(
        {
            MemoryType.CUPY: 2,
            MemoryType.JAX: 4,
            MemoryType.TORCH: 5,
        }
    )

    with pytest.raises(ExceptionGroup) as caught:
        assignment.cleanup_loaded()

    assert cleaned == [
        (MemoryType.CUPY, 2),
        (MemoryType.JAX, 4),
        (MemoryType.TORCH, 5),
    ]
    assert [str(error) for error in caught.value.exceptions] == [
        "cupy cleanup failed"
    ]
