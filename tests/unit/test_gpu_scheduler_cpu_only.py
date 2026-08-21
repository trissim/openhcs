from openhcs.core.orchestrator import gpu_scheduler
from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_gpu_detection_honors_cpu_only_process_authority(monkeypatch) -> None:
    def fail_on_gpu_probe(_library_name: str) -> None:
        raise AssertionError("CPU-only detection imported a GPU runtime")

    monkeypatch.setenv(OpenHCSProcessEnvironment.cpu_only_key, "true")
    monkeypatch.setattr(
        gpu_scheduler,
        "check_gpu_capability",
        fail_on_gpu_probe,
    )

    assert gpu_scheduler._detect_available_gpus() == []
