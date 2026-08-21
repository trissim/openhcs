from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_process_environment_owns_inherited_mode_selectors() -> None:
    assert OpenHCSProcessEnvironment.child_process_environment_keys() == (
        OpenHCSProcessEnvironment.cpu_only_key,
        OpenHCSProcessEnvironment.headless_key,
        OpenHCSProcessEnvironment.numba_cache_key,
        OpenHCSProcessEnvironment.use_threading_key,
    )


def test_process_environment_owns_numba_cache_location(tmp_path) -> None:
    assert (
        OpenHCSProcessEnvironment.numba_cache_path(tmp_path)
        == (tmp_path / "cache" / "numba").resolve()
    )


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
