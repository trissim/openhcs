from pathlib import PureWindowsPath

from openhcs.agent.runtime_platform import WindowsAgentRuntimePlatformAuthority
from openhcs.resources.brand import BRAND_PRODUCT_NAME
from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_process_environment_owns_inherited_mode_selectors() -> None:
    assert OpenHCSProcessEnvironment.child_process_environment_keys() == (
        OpenHCSProcessEnvironment.cpu_only_key,
        OpenHCSProcessEnvironment.headless_key,
        OpenHCSProcessEnvironment.numba_cache_key,
        OpenHCSProcessEnvironment.use_threading_key,
    )


def test_process_environment_owns_numba_cache_location(monkeypatch, tmp_path) -> None:
    local_data = tmp_path / "local-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    cache_path = OpenHCSProcessEnvironment.numba_cache_path(
        WindowsAgentRuntimePlatformAuthority()
    )

    assert cache_path == (local_data / BRAND_PRODUCT_NAME / "numba").resolve()


def test_windows_numba_cache_preserves_legacy_path_budget(
    monkeypatch, tmp_path
) -> None:
    local_data = tmp_path / "local-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))
    cache_path = OpenHCSProcessEnvironment.numba_cache_path(
        WindowsAgentRuntimePlatformAuthority()
    )
    relative_cache = cache_path.relative_to(local_data.resolve())
    windows_cache = PureWindowsPath(
        r"C:\Users\runneradmin\AppData\Local",
        *relative_cache.parts,
    )
    generated_cache_path = (
        windows_cache
        / ("cellprofiler_926ba2ac16708c16c48cca82b797fbb54b5840a7")
        / (
            "thresholding_threshold_numba_diagnostics_quantized."
            "_threshold_diagnostics_unmasked_finite_quantized_numba-187."
            "py312.nbi.tmp.e379105096cf4ba9"
        )
    )

    assert len(str(generated_cache_path)) < 260


def test_process_environment_resolves_boolean_modes() -> None:
    environment = {
        OpenHCSProcessEnvironment.cpu_only_key: "YES",
        OpenHCSProcessEnvironment.headless_key: "0",
        OpenHCSProcessEnvironment.use_threading_key: "on",
    }

    assert OpenHCSProcessEnvironment.cpu_only_mode(environment) is True
    assert OpenHCSProcessEnvironment.headless_mode(environment) is False
    assert OpenHCSProcessEnvironment.use_threading_mode(environment) is True


def test_cpu_only_mode_projects_gpu_import_policy_to_dependencies() -> None:
    environment = {}

    OpenHCSProcessEnvironment.enable_cpu_only_mode(environment)

    assert environment == {
        OpenHCSProcessEnvironment.cpu_only_key: "true",
        OpenHCSProcessEnvironment.subprocess_no_gpu_key: "1",
        OpenHCSProcessEnvironment.polystore_subprocess_no_gpu_key: "1",
    }
    assert OpenHCSProcessEnvironment.gpu_imports_disabled(environment) is True


def test_subprocess_gpu_suppression_projects_without_enabling_cpu_only() -> None:
    environment = {OpenHCSProcessEnvironment.subprocess_no_gpu_key: "1"}

    OpenHCSProcessEnvironment.project_dependency_gpu_import_policy(environment)

    assert OpenHCSProcessEnvironment.cpu_only_mode(environment) is False
    assert environment[OpenHCSProcessEnvironment.subprocess_no_gpu_key] == "1"
    assert environment[OpenHCSProcessEnvironment.polystore_subprocess_no_gpu_key] == "1"
