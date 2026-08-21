from __future__ import annotations

import json
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

from openhcs.pyqt_gui.services import desktop_update_worker
from openhcs.resources.brand import BrandAsset, brand_asset_path

BACKGROUND_LAUNCH_SPEC = desktop_update_worker.ResolvedProcessLaunchSpec(
    creationflags=73,
    start_new_session=False,
)
DETACHED_LAUNCH_SPEC = desktop_update_worker.ResolvedProcessLaunchSpec(
    creationflags=91,
    start_new_session=False,
)
WORKER_LAUNCH_ARGUMENTS = [
    "--background-creationflags=73",
    "--detached-creationflags=91",
]
NATIVE_WHEEL_POLICY = "llvmlite,numba,opencv-python,opencv-python-headless"


class _ProgressProbe:
    def __init__(
        self,
        *,
        action: desktop_update_worker.DesktopUpdateProgressAction = (
            desktop_update_worker.DesktopUpdateProgressAction.EXIT
        ),
    ) -> None:
        self.action = action
        self.phases = []
        self.outputs = []
        self.failures = []
        self.completed = False

    def phase(self, phase) -> None:
        self.phases.append(phase)

    def output(self, message: str) -> None:
        self.outputs.append(message)

    def failure(self, message: str):
        self.failures.append(message)
        return self.action

    def complete(self) -> None:
        self.completed = True

    def run(self, operation) -> int:
        return operation()


class _StreamingProcess:
    def __init__(self, returncode: int, output: str) -> None:
        self._returncode = returncode
        self.stdout = io.StringIO(output)

    def wait(self) -> int:
        return self._returncode


def _progress_arguments(tmp_path: Path) -> list[str]:
    return [
        "--progress-theme-file",
        str(tmp_path / "desktop-update-theme.json"),
        "--progress-brand-file",
        str(tmp_path / "desktop-update-brand.png"),
    ]


def _update_plan(tmp_path: Path) -> desktop_update_worker.DesktopUpdatePlan:
    candidate = tmp_path / "env-1234abcd"
    return desktop_update_worker.DesktopUpdatePlan(
        update_executable=str(tmp_path / "uv"),
        base_python_executable=str(tmp_path / "base-python"),
        previous_environment=str(tmp_path / "env-current"),
        candidate_environment=str(candidate),
        candidate_python_executable=str(candidate / "bin" / "python"),
        package_requirement=(
            "openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.7.1"
        ),
        binary_only_packages=NATIVE_WHEEL_POLICY,
        expected_version="0.7.1",
        installation_pointer=str(tmp_path / "current"),
    )


def _write_update_plan(tmp_path: Path) -> Path:
    path = tmp_path / "desktop-update-plan.json"
    _update_plan(tmp_path).write(path)
    return path


def _worker_arguments(
    tmp_path: Path,
    *,
    session_directory: Path | None = None,
    restart_executable: str = "openhcs",
    restart_arguments: tuple[str, ...] = (),
) -> list[str]:
    session = tmp_path if session_directory is None else session_directory
    arguments = [
        "--parent-pid",
        "42",
        "--session-directory",
        str(session),
        "--update-plan-file",
        str(_write_update_plan(tmp_path)),
        "--restart-executable",
        restart_executable,
    ]
    arguments.extend(f"--restart-argument={argument}" for argument in restart_arguments)
    arguments.extend(
        (
            "--error-file",
            str(tmp_path / "update-error.txt"),
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        )
    )
    return arguments


def test_update_plan_round_trips_windows_managed_paths(tmp_path: Path) -> None:
    plan = desktop_update_worker.DesktopUpdatePlan(
        update_executable="C:/OpenHCS/bootstrap/uv/uv.exe",
        base_python_executable="C:/OpenHCS/python/python.exe",
        previous_environment="C:/OpenHCS/env-current",
        candidate_environment="C:/OpenHCS/env-1234abcd",
        candidate_python_executable="C:/OpenHCS/env-1234abcd/Scripts/python.exe",
        package_requirement="openhcs[gui,mcp]==0.7.24",
        binary_only_packages=NATIVE_WHEEL_POLICY,
        expected_version="0.7.24",
        installation_pointer="C:/OpenHCS/Launch-OpenHCS.ps1",
    )
    path = tmp_path / "plan.json"

    plan.write(path)

    assert desktop_update_worker.DesktopUpdatePlan.read(path) == plan


def test_update_plan_rejects_candidate_outside_current_environment_parent(
    tmp_path: Path,
) -> None:
    plan = _update_plan(tmp_path)
    invalid = desktop_update_worker.DesktopUpdatePlan(
        **{
            **asdict(plan),
            "candidate_environment": str(tmp_path / "elsewhere" / "env-1234abcd"),
            "candidate_python_executable": str(
                tmp_path / "elsewhere" / "env-1234abcd" / "bin" / "python"
            ),
        }
    )

    with pytest.raises(ValueError, match="beside the current environment"):
        invalid.validate()


def test_update_plan_accepts_authority_owned_candidate_name(tmp_path: Path) -> None:
    plan = _update_plan(tmp_path)
    candidate = tmp_path / "release-candidate"
    projected = desktop_update_worker.DesktopUpdatePlan(
        **{
            **asdict(plan),
            "candidate_environment": str(candidate),
            "candidate_python_executable": str(candidate / "bin" / "python"),
        }
    )

    projected.validate()


def test_update_plan_rejects_current_environment_as_candidate(tmp_path: Path) -> None:
    plan = _update_plan(tmp_path)
    current = Path(plan.previous_environment)
    invalid = desktop_update_worker.DesktopUpdatePlan(
        **{
            **asdict(plan),
            "candidate_environment": str(current),
            "candidate_python_executable": str(current / "bin" / "python"),
        }
    )

    with pytest.raises(ValueError, match="must differ"):
        invalid.validate()


def test_worker_never_reuses_or_removes_preexisting_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _update_plan(tmp_path)
    candidate = Path(plan.candidate_environment)
    candidate.mkdir()
    sentinel = candidate / "foreign.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("preexisting candidate must not run"),
    )

    execution = desktop_update_worker._run_update(
        plan,
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=_ProgressProbe(),
    )

    assert execution.error_message is not None
    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"


def test_worker_reports_bounded_install_failure(monkeypatch, tmp_path: Path) -> None:
    progress = _ProgressProbe()
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _StreamingProcess(7, "failure detail\n"),
    )

    execution = desktop_update_worker._run_update(
        _update_plan(tmp_path),
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert execution.error_message == (
        "OpenHCS could not create the replacement environment (exit code 7)."
        "\n\nfailure detail"
    )
    assert execution.restart_executable is None
    assert progress.phases == [
        desktop_update_worker.DesktopUpdatePhase.PREPARING_ENVIRONMENT
    ]
    assert progress.outputs == ["failure detail"]
    assert not Path(_update_plan(tmp_path).candidate_environment).exists()


def test_worker_stages_and_verifies_replacement_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "created environment\n"),
            _StreamingProcess(0, "resolved packages\ninstalled OpenHCS\n"),
            _StreamingProcess(0, "dependencies verified\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(
                0,
                '{"platform": "macos", '
                f'"restart_executable": "{tmp_path / "OpenHCS.app"}"}}\n',
            ),
        )
    )

    def _popen(command, **kwargs):
        calls.append((command, kwargs))
        return next(processes)

    monkeypatch.setattr(desktop_update_worker.subprocess, "Popen", _popen)

    execution = desktop_update_worker._run_update(
        _update_plan(tmp_path),
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    plan = _update_plan(tmp_path)
    assert execution == desktop_update_worker.DesktopUpdateExecution(
        restart_executable=str(tmp_path / "OpenHCS.app")
    )
    assert calls[0][0] == [
        plan.update_executable,
        "--no-config",
        "venv",
        "--python",
        plan.base_python_executable,
        "--seed",
        plan.candidate_environment,
    ]
    assert calls[1][0][-1] == plan.package_requirement
    assert "--prefer-binary" in calls[1][0]
    assert calls[2][0][1:4] == ["-m", "pip", "check"]
    assert calls[3][0][0] == plan.candidate_python_executable
    assert calls[3][0][-1] == "0.7.1"
    assert calls[4][0][-2:] == [
        f"--installation-pointer={tmp_path / 'current'}",
        "--json",
    ]
    assert calls[0][1]["creationflags"] == 73
    assert calls[1][1]["creationflags"] == 73
    assert progress.phases == [
        desktop_update_worker.DesktopUpdatePhase.PREPARING_ENVIRONMENT,
        desktop_update_worker.DesktopUpdatePhase.INSTALLING,
        desktop_update_worker.DesktopUpdatePhase.VERIFYING,
        desktop_update_worker.DesktopUpdatePhase.REFRESHING_DESKTOP,
    ]
    assert progress.outputs == [
        "created environment",
        "resolved packages",
        "installed OpenHCS",
        "dependencies verified",
        "version verified",
        (
            '{"platform": "macos", "restart_executable": '
            f'"{tmp_path / "OpenHCS.app"}"}}'
        ),
    ]
    assert Path(plan.candidate_environment).is_dir()


def test_worker_refreshes_installer_managed_desktop_after_verification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "created environment\n"),
            _StreamingProcess(0, "installed OpenHCS\n"),
            _StreamingProcess(0, "dependencies verified\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(
                0,
                '{"platform": "windows", '
                '"restart_executable": "C:/OpenHCS/OpenHCS.exe"}\n',
            ),
        )
    )

    def _popen(command, **kwargs):
        calls.append((command, kwargs))
        return next(processes)

    monkeypatch.setattr(desktop_update_worker.subprocess, "Popen", _popen)

    execution = desktop_update_worker._run_update(
        _update_plan(tmp_path),
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert execution.error_message is None
    assert execution.restart_executable == "C:/OpenHCS/OpenHCS.exe"
    assert calls[4][0] == [
        _update_plan(tmp_path).candidate_python_executable,
        "-I",
        "-m",
        "openhcs.desktop_deployment_cli",
        f"--installation-pointer={tmp_path / 'current'}",
        "--json",
    ]
    assert progress.phases == [
        desktop_update_worker.DesktopUpdatePhase.PREPARING_ENVIRONMENT,
        desktop_update_worker.DesktopUpdatePhase.INSTALLING,
        desktop_update_worker.DesktopUpdatePhase.VERIFYING,
        desktop_update_worker.DesktopUpdatePhase.REFRESHING_DESKTOP,
    ]


def test_worker_reports_desktop_refresh_failure_without_switching(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "created environment\n"),
            _StreamingProcess(0, "installed OpenHCS\n"),
            _StreamingProcess(0, "dependencies verified\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(1, "shortcut publication failed\n"),
        )
    )
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(processes),
    )

    execution = desktop_update_worker._run_update(
        _update_plan(tmp_path),
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert execution.error_message is not None
    assert "could not publish the verified replacement" in execution.error_message
    assert "shortcut publication failed" in execution.error_message
    assert not Path(_update_plan(tmp_path).candidate_environment).exists()


def test_worker_retains_published_candidate_when_deployment_report_is_invalid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processes = iter(
        (
            _StreamingProcess(0, "created environment\n"),
            _StreamingProcess(0, "installed OpenHCS\n"),
            _StreamingProcess(0, "dependencies verified\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(0, "publication completed without report\n"),
        )
    )
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(processes),
    )
    plan = _update_plan(tmp_path)

    execution = desktop_update_worker._run_update(
        plan,
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=_ProgressProbe(),
    )

    assert execution.error_message is not None
    assert "published the update" in execution.error_message
    assert Path(plan.candidate_environment).is_dir()


def test_worker_restarts_prior_entry_with_saved_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    launched = []
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )

    desktop_update_worker._restart(
        "openhcs",
        ["--log-level", "INFO"],
        session_directory=tmp_path,
        restore_option="--restore-update-session",
        launch_spec=DETACHED_LAUNCH_SPEC,
    )

    assert launched[0][0] == [
        "openhcs",
        "--log-level",
        "INFO",
        "--restore-update-session",
        str(tmp_path),
    ]
    assert launched[0][1]["close_fds"] is True
    assert launched[0][1]["stdin"] is subprocess.DEVNULL
    assert launched[0][1]["stdout"] is subprocess.DEVNULL
    assert launched[0][1]["stderr"] is subprocess.DEVNULL
    assert launched[0][1]["creationflags"] == 91


def test_worker_relaunches_and_preserves_session_after_update_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    progress = _ProgressProbe(
        action=desktop_update_worker.DesktopUpdateProgressAction.REOPEN
    )
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: progress,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda pid: calls.append(("wait", pid)) or True,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: desktop_update_worker.DesktopUpdateExecution(
            error_message="network unavailable"
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda executable, arguments, *, session_directory, restore_option, launch_spec: calls.append(
            ("restart", executable, arguments, session_directory, launch_spec)
        ),
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(
        _worker_arguments(
            tmp_path,
            restart_arguments=("--log-level", "INFO"),
        )
    )

    assert result == 1
    assert error_file.read_text(encoding="utf-8") == "network unavailable"
    assert progress.failures == ["network unavailable"]
    assert calls == [
        ("wait", 42),
        (
            "restart",
            "openhcs",
            ["--log-level", "INFO"],
            tmp_path,
            DETACHED_LAUNCH_SPEC,
        ),
    ]


def test_successful_managed_update_restarts_through_deployment_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe()
    stable_launcher = "C:/Users/test/AppData/Local/OpenHCS/OpenHCS.exe"
    arguments = desktop_update_worker.parse_arguments(
        _worker_arguments(
            tmp_path,
            restart_executable="C:/OpenHCS/env-old/Scripts/openhcs-gui.exe",
        )
    )
    restarts: list[str] = []
    monkeypatch.setattr(
        desktop_update_worker, "_wait_for_parent_exit", lambda _pid: True
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: desktop_update_worker.DesktopUpdateExecution(
            restart_executable=stable_launcher
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda executable, *_args, **_kwargs: restarts.append(executable) or None,
    )

    result = desktop_update_worker._perform_update_transaction(
        arguments,
        progress=progress,
        background_launch_spec=BACKGROUND_LAUNCH_SPEC,
        detached_launch_spec=DETACHED_LAUNCH_SPEC,
    )

    assert result == 0
    assert restarts == [stable_launcher]
    assert progress.completed is True


def test_worker_cancels_before_update_when_parent_does_not_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe()
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: progress,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda _pid: False,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("update must not start while parent is alive")
        ),
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(_worker_arguments(tmp_path))

    assert result == 2
    assert tmp_path.exists()
    assert "cancelled before modifying" in error_file.read_text(encoding="utf-8")
    assert progress.failures == [error_file.read_text(encoding="utf-8")]


def test_worker_fails_closed_and_reopens_when_progress_window_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: (_ for _ in ()).throw(
            desktop_update_worker.DesktopUpdateProgressUnavailable(
                "progress unavailable; environment not modified"
            )
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda pid: calls.append(("wait", pid)) or True,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("update must not run without a visible progress surface")
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda executable, arguments, *, session_directory, restore_option, launch_spec: calls.append(
            ("restart", session_directory)
        )
        or None,
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(_worker_arguments(tmp_path))

    assert result == 3
    assert error_file.read_text(encoding="utf-8") == (
        "progress unavailable; environment not modified"
    )
    assert calls == [("wait", 42), ("restart", tmp_path)]


def test_unexpected_orchestration_exception_uses_visible_recovery_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe(
        action=desktop_update_worker.DesktopUpdateProgressAction.REOPEN
    )
    error_file = tmp_path / "update-error.txt"
    arguments = desktop_update_worker.parse_arguments(_worker_arguments(tmp_path))
    restarts = []
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("wait boundary exploded")),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda *_args, **_kwargs: restarts.append("reopened") or None,
    )

    result = desktop_update_worker._perform_update(
        arguments,
        progress=progress,
        background_launch_spec=BACKGROUND_LAUNCH_SPEC,
        detached_launch_spec=DETACHED_LAUNCH_SPEC,
    )

    expected = "OpenHCS could not complete the update: wait boundary exploded"
    assert result == 1
    assert progress.failures == [expected]
    assert error_file.read_text(encoding="utf-8") == expected
    assert restarts == ["reopened"]


def test_parser_preserves_leading_dash_restart_arguments(tmp_path: Path) -> None:
    arguments = desktop_update_worker.parse_arguments(
        _worker_arguments(
            tmp_path,
            session_directory=Path("/tmp/session"),
            restart_arguments=("--log-level", "DEBUG"),
        )
    )

    assert arguments.restart_argument == ["--log-level", "DEBUG"]
    assert arguments.update_plan_file == tmp_path / "desktop-update-plan.json"
    assert arguments.background_creationflags == 73
    assert arguments.detached_creationflags == 91
    assert arguments.progress_theme_file == tmp_path / "desktop-update-theme.json"
    assert arguments.progress_brand_file == tmp_path / "desktop-update-brand.png"


@pytest.mark.skipif(
    os.name == "nt"
    or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY")),
    reason="POSIX Tk progress-window probe requires a display",
)
def test_worker_process_waits_updates_restarts_and_restores_session(
    tmp_path: Path,
) -> None:
    session_directory = tmp_path / "pending"
    session_directory.mkdir()
    (session_directory / "session.py").write_text(
        "canonical session source",
        encoding="utf-8",
    )
    (session_directory / "objectstate-history.objectstate").write_text(
        "canonical history",
        encoding="utf-8",
    )
    desktop_update_worker.DesktopUpdateProgressTheme(
        window_bg="#2b2b2b",
        panel_bg="#1e1e1e",
        text_primary="#ffffff",
        text_secondary="#cccccc",
        text_accent="#00aaff",
        border_color="#555555",
        button_bg="#404040",
        button_text="#ffffff",
        error_color="#ff0000",
        progress_color="#0078d4",
    ).write(session_directory / "desktop-update-theme.json")
    shutil.copyfile(
        brand_asset_path(BrandAsset.ICON_RASTER),
        session_directory / "desktop-update-brand.png",
    )
    update_marker = tmp_path / "updated.txt"
    restore_marker = tmp_path / "restored.json"
    restore_script = tmp_path / "restore.py"
    restore_script.write_text(
        """
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from objectstate.object_state import ObjectStateRegistry
from openhcs.pyqt_gui.services.desktop_update import DesktopRestartSession

parser = argparse.ArgumentParser()
parser.add_argument("marker", type=Path)
parser.add_argument("--restore-update-session", required=True, type=Path)
args = parser.parse_args()
calls = []
ObjectStateRegistry.load_history_from_file = classmethod(
    lambda cls, path: calls.append(["history", Path(path).read_text(encoding="utf-8")])
)
plate_manager = SimpleNamespace(
    apply_code_document_source=lambda source: calls.append(["source", source]),
    update_item_list=lambda: calls.append(["refresh", None]),
)
main_window = SimpleNamespace(
    embedded_widgets=SimpleNamespace(
        require_plate_manager=lambda: plate_manager,
    ),
    time_travel_widget=SimpleNamespace(
        refresh=lambda: calls.append(["history-ui", None]),
    ),
)
error = DesktopRestartSession(args.restore_update_session).restore(main_window)
args.marker.write_text(
    json.dumps({"calls": calls, "error": error}),
    encoding="utf-8",
)
""".strip(),
        encoding="utf-8",
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.3)"],
    )
    parent_reaper = threading.Thread(target=parent.wait)
    parent_reaper.start()
    source_root = Path(__file__).resolve().parents[3]
    external_roots = (
        "external/ObjectState/src",
        "external/python-introspect/src",
        "external/metaclass-registry/src",
        "external/arraybridge/src",
        "external/pycodify/src",
        "external/PolyStore/src",
        "external/pyqt-reactive/src",
        "external/zmqruntime/src",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), *(str(source_root / path) for path in external_roots))
    )
    candidate = tmp_path / "env-1234abcd"
    candidate_python = candidate / "bin" / "python"
    fake_uv = tmp_path / "fake-uv"
    candidate_source = (
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"update_marker = Path({str(update_marker)!r})\n"
        "if sys.argv[1:4] == ['-m', 'pip', 'install']:\n"
        "    update_marker.write_text('updated', encoding='utf-8')\n"
        "elif 'openhcs.desktop_deployment_cli' in sys.argv:\n"
        f"    print(json.dumps({{'restart_executable': {sys.executable!r}}}))\n"
    )
    fake_uv.write_text(
        (
            f"#!{sys.executable}\n"
            "import sys\n"
            "from pathlib import Path\n"
            "candidate = Path(sys.argv[-1])\n"
            "python = candidate / 'bin' / 'python'\n"
            "python.parent.mkdir(parents=True, exist_ok=True)\n"
            f"python.write_text({candidate_source!r}, encoding='utf-8')\n"
            "python.chmod(0o755)\n"
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    plan = desktop_update_worker.DesktopUpdatePlan(
        update_executable=str(fake_uv),
        base_python_executable=sys.executable,
        previous_environment=str(tmp_path / "env-current"),
        candidate_environment=str(candidate),
        candidate_python_executable=str(candidate_python),
        package_requirement=f"openhcs=={distribution_version('openhcs')}",
        binary_only_packages=NATIVE_WHEEL_POLICY,
        expected_version=distribution_version("openhcs"),
        installation_pointer=str(tmp_path / "current"),
    )
    update_plan_path = session_directory / "desktop-update-plan.json"
    plan.write(update_plan_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "openhcs.pyqt_gui.services.desktop_update_worker",
            "--parent-pid",
            str(parent.pid),
            "--session-directory",
            str(session_directory),
            "--update-plan-file",
            str(update_plan_path),
            "--restart-executable",
            sys.executable,
            f"--restart-argument={restore_script}",
            f"--restart-argument={restore_marker}",
            "--error-file",
            str(session_directory / "update-error.txt"),
            "--restore-option=--restore-update-session",
            *_progress_arguments(session_directory),
            "--background-creationflags=0",
            "--detached-creationflags=0",
            "--detached-start-new-session",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    parent_reaper.join(timeout=2)
    deadline = time.monotonic() + 10
    while not restore_marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert completed.returncode == 0, completed.stderr
    assert update_marker.read_text(encoding="utf-8") == "updated"
    restored = json.loads(restore_marker.read_text(encoding="utf-8"))
    assert restored == {
        "calls": [
            ["source", "canonical session source"],
            ["history", "canonical history"],
            ["history-ui", None],
            ["refresh", None],
        ],
        "error": None,
    }
    assert not session_directory.exists()
