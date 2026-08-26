"""Start and close an installed OpenHCS GUI outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import distribution
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhcs.pyqt_gui.config import UIConfig

JsonScalar = str | int | bool | None


@dataclass(frozen=True, slots=True)
class InstalledGuiSmokeResult:
    """Validated installed-GUI lifecycle evidence."""

    execution_port: int
    execution_transport: str
    exit_code: int
    openhcs_version: str
    package_path: str
    qt_platform: str
    ready: bool
    startup_error: str | None
    timed_out: bool
    visible: bool

    def payload(self, *, working_directory: Path) -> dict[str, JsonScalar]:
        """Project the typed result into the script's JSON boundary."""

        return {
            **asdict(self),
            "working_directory": str(working_directory),
        }


@dataclass(slots=True)
class _StartupObservation:
    """Mutable callback state finalized into one validated result."""

    ready: bool = False
    visible: bool = False
    startup_error: str | None = None
    timed_out: bool = False

    def record_ready(self, *, visible: bool) -> None:
        self.ready = True
        self.visible = visible

    def record_failure(self, error: BaseException) -> None:
        self.startup_error = f"{type(error).__name__}: {error}"

    def record_timeout(self) -> None:
        self.timed_out = True

    def result(
        self,
        *,
        execution_port: int,
        execution_transport: str,
        exit_code: int,
        openhcs_version: str,
        package_path: Path,
        qt_platform: str,
        timeout_seconds: float,
    ) -> InstalledGuiSmokeResult:
        """Validate the observed lifecycle and freeze its evidence."""

        if self.startup_error is not None:
            raise AssertionError(f"Installed GUI startup failed: {self.startup_error}")
        if self.timed_out:
            raise TimeoutError(
                "Installed GUI did not reach its painted ready boundary within "
                f"{timeout_seconds:g} seconds."
            )
        if not self.ready or not self.visible:
            raise AssertionError(
                "Installed GUI did not become visibly ready: "
                f"ready={self.ready}, visible={self.visible}"
            )
        if exit_code != 0:
            raise AssertionError(f"Installed GUI exited with {exit_code}.")
        return InstalledGuiSmokeResult(
            execution_port=execution_port,
            execution_transport=execution_transport,
            exit_code=exit_code,
            openhcs_version=openhcs_version,
            package_path=str(package_path),
            qt_platform=qt_platform,
            ready=self.ready,
            startup_error=self.startup_error,
            timed_out=self.timed_out,
            visible=self.visible,
        )


def assert_not_source_checkout_import(
    *,
    package_path: Path,
    forbidden_root: Path,
) -> None:
    """Reject a source-owned import without rejecting an in-tree wheel venv."""

    if package_path.is_relative_to(forbidden_root / "openhcs"):
        raise AssertionError(
            "Smoke test imported the source checkout instead of the wheel: "
            f"{package_path}"
        )


def with_isolated_execution_endpoint(ui_config: UIConfig) -> UIConfig:
    """Return the UI configuration with one declaration-allocated endpoint."""

    from zmqruntime import DataControlPortPairAuthority

    transport_config = ui_config.zmq
    port_pair = DataControlPortPairAuthority.acquire(
        transport_config,
        transport_mode=transport_config.transport_mode,
        host=transport_config.client_host,
    )
    return replace(
        ui_config,
        zmq=replace(transport_config, default_port=port_pair.data_port),
    )


def run_installed_gui_smoke(
    *,
    forbidden_root: Path,
    timeout_seconds: float,
) -> InstalledGuiSmokeResult:
    """Exercise construction, deferred startup, paint, and orderly shutdown."""

    if timeout_seconds <= 0:
        raise ValueError("GUI startup smoke timeout must be positive.")
    os.environ["OPENHCS_CPU_ONLY"] = "true"

    from PyQt6.QtCore import QTimer

    import openhcs
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp
    from openhcs.pyqt_gui.config import (
        AgentUiBridgeConfig,
        PyQtGuiRuntimeContext,
        UIConfig,
    )

    package_path = Path(openhcs.__file__).resolve()
    assert_not_source_checkout_import(
        package_path=package_path,
        forbidden_root=forbidden_root.resolve(),
    )

    ui_config = with_isolated_execution_endpoint(
        UIConfig(
            check_for_updates_on_startup=False,
            agent_bridge=replace(AgentUiBridgeConfig(), enabled=False),
        )
    )
    runtime_context = PyQtGuiRuntimeContext(
        ui_config=ui_config,
        pipeline_runtime=GlobalPipelineConfig(),
    )
    application = OpenHCSPyQtApp(
        ["openhcs-gui-installed-smoke", "--no-gpu"],
        runtime_context=runtime_context,
    )
    observation = _StartupObservation()

    def close_application() -> None:
        main_window = application.main_window
        if main_window is None:
            application.exit(1)
            return
        main_window.close()

    def startup_ready() -> None:
        main_window = application.main_window
        observation.record_ready(
            visible=bool(main_window is not None and main_window.isVisible())
        )
        QTimer.singleShot(0, close_application)

    def startup_failed(error: BaseException) -> None:
        observation.record_failure(error)

    def startup_timed_out() -> None:
        if observation.ready:
            return
        observation.record_timeout()
        close_application()

    QTimer.singleShot(round(timeout_seconds * 1_000), startup_timed_out)
    exit_code = application.run(
        on_main_window_ready=startup_ready,
        on_startup_failure=startup_failed,
    )
    return observation.result(
        execution_port=ui_config.zmq.default_port,
        execution_transport=ui_config.zmq.transport_mode.value,
        exit_code=exit_code,
        openhcs_version=distribution("openhcs").version,
        package_path=package_path,
        qt_platform=application.platformName(),
        timeout_seconds=timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forbid-import-root",
        type=Path,
        required=True,
        help="Source checkout that must not own the imported openhcs package.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum time to wait for the painted main-window ready boundary.",
    )
    args = parser.parse_args(argv)
    original_working_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="openhcs-installed-gui-") as directory:
        working_directory = Path(directory).resolve()
        os.chdir(working_directory)
        try:
            result = run_installed_gui_smoke(
                forbidden_root=args.forbid_import_root,
                timeout_seconds=args.timeout_seconds,
            )
            payload = result.payload(working_directory=working_directory)
        finally:
            os.chdir(original_working_directory)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
