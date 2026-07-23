from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_process_environment_owns_inherited_mode_selectors() -> None:
    assert OpenHCSProcessEnvironment.child_process_environment_keys() == (
        OpenHCSProcessEnvironment.cpu_only_key,
        OpenHCSProcessEnvironment.headless_key,
        OpenHCSProcessEnvironment.use_threading_key,
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
