from __future__ import annotations

import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import patch

from pyqt_reactive.protocols import (
    ComponentSelectionProviderABC,
    FunctionSelectionProviderABC,
    LogDiscoveryProviderABC,
    ServerScanProviderABC,
)
from pyqt_reactive.services.zmq_server_scan_service import (
    EndpointObservationSnapshot,
    ZMQServerScanService,
)
from zmqruntime.execution.logs import ExecutionWorkerLogIdentity
from zmqruntime.messages import (
    PongResponse,
    ProcessIdentity,
    RunningExecutionInfo,
    ServerRole,
    WorkerState,
)

from openhcs.pyqt_gui.config import GuiLogLevel, LoggingConfig, UIConfig
from openhcs.pyqt_gui.services.logging_config import (
    GuiLoggingHandler,
    configure_gui_logging,
)
from openhcs.pyqt_gui.services.reactor_providers import (
    OpenHCSComponentSelectionProvider,
    OpenHCSFunctionSelectionProvider,
    OpenHCSLogDiscoveryProvider,
    OpenHCSServerScanProvider,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


def _owned_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, GuiLoggingHandler)
    ]


def test_host_providers_implement_their_nominal_pyqt_contracts() -> None:
    provider_contracts = (
        (OpenHCSLogDiscoveryProvider, LogDiscoveryProviderABC),
        (OpenHCSServerScanProvider, ServerScanProviderABC),
        (OpenHCSComponentSelectionProvider, ComponentSelectionProviderABC),
        (OpenHCSFunctionSelectionProvider, FunctionSelectionProviderABC),
    )

    for provider_type, contract_type in provider_contracts:
        assert issubclass(provider_type, contract_type)


def test_logging_config_owns_level_location_destinations_and_rotation(tmp_path) -> None:
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_disable = logging.root.manager.disable
    previous_handlers = tuple(root_logger.handlers)
    sentinel = logging.NullHandler()
    root_logger.addHandler(sentinel)
    config = LoggingConfig(
        level=GuiLogLevel.WARNING,
        log_directory=tmp_path / "custom-logs",
        enable_console_logging=False,
        max_file_size_mb=3,
        backup_count=7,
    )

    try:
        log_file = configure_gui_logging(config)

        assert log_file is not None
        assert log_file.parent == (tmp_path / "custom-logs").resolve()
        assert sentinel not in root_logger.handlers
        assert root_logger.level == logging.WARNING
        assert root_logger.handlers == _owned_handlers()
        assert len(_owned_handlers()) == 1
        file_handler = _owned_handlers()[0]
        assert isinstance(file_handler, RotatingFileHandler)
        assert file_handler.maxBytes == 3 * 1024 * 1024
        assert file_handler.backupCount == 7
    finally:
        for handler in _owned_handlers():
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.handlers = list(previous_handlers)
        root_logger.setLevel(previous_level)
        logging.disable(previous_disable)


def test_default_log_directory_follows_openhcs_data_authority(
    tmp_path,
    monkeypatch,
) -> None:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    assert LoggingConfig().resolved_log_directory() == data_home / "openhcs" / "logs"


def test_logging_config_replaces_bootstrap_console_handler() -> None:
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_disable = logging.root.manager.disable
    previous_handlers = tuple(root_logger.handlers)
    bootstrap_output = io.StringIO()
    configured_output = io.StringIO()
    bootstrap_handler = logging.StreamHandler(bootstrap_output)
    root_logger.handlers = [bootstrap_handler]

    try:
        config = LoggingConfig(
            level=GuiLogLevel.INFO,
            enable_console_logging=True,
            enable_file_logging=False,
        )
        with patch("sys.stdout", configured_output):
            configure_gui_logging(config)
            logging.getLogger(__name__).info("one configured record")

        assert bootstrap_output.getvalue() == ""
        assert configured_output.getvalue().count("one configured record") == 1
        assert root_logger.handlers == _owned_handlers()
    finally:
        for handler in _owned_handlers():
            handler.close()
        root_logger.handlers = list(previous_handlers)
        root_logger.setLevel(previous_level)
        logging.disable(previous_disable)


def test_log_discovery_derives_live_directory_from_ui_config(
    tmp_path,
    monkeypatch,
) -> None:
    current = UIConfig(logging=LoggingConfig(log_directory=tmp_path / "declared-logs"))
    provider = OpenHCSLogDiscoveryProvider(lambda: current)
    captured: dict[str, Path] = {}

    import openhcs.core.log_utils as log_utils

    monkeypatch.setattr(
        log_utils,
        "discover_logs",
        lambda **kwargs: captured.setdefault("directory", kwargs["log_directory"])
        and [],
    )
    assert provider.discover_logs(include_main_log=False) == []

    assert captured["directory"] == (tmp_path / "declared-logs").resolve()


def test_server_log_discovery_preserves_heartbeat_process_identity(
    tmp_path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "execution.log"
    log_path.write_text("ready\n", encoding="utf-8")
    execution_id = "execution-1"
    worker_log_path = ExecutionWorkerLogIdentity(
        execution_id=execution_id,
        worker_pid=4321,
    ).path(tmp_path)
    worker_log_path.write_text("working\n", encoding="utf-8")
    config = OpenHCSZMQConfig(server_scan_timeout_ms=137)
    process_identity = ProcessIdentity.current()
    captured = {}

    def scan_ports(scan_service, ports):
        captured["timeout_ms"] = scan_service.timeout_ms
        captured["ports"] = tuple(ports)
        return EndpointObservationSnapshot.from_responses(
            (
                PongResponse(
                    port=config.default_port,
                    control_port=(config.default_port + config.control_port_offset),
                    ready=True,
                    server="OpenHCSExecutionServer",
                    server_role=ServerRole.EXECUTION,
                    log_file_path=str(log_path),
                    process_identity=process_identity,
                    running_executions=(
                        RunningExecutionInfo(
                            execution_id=execution_id,
                            plate_id="plate-1",
                            start_time=1.0,
                            elapsed=2.0,
                        ),
                    ),
                    workers=(
                        WorkerState(
                            pid=4321,
                            status="running",
                            cpu_percent=1.0,
                            memory_mb=2.0,
                            create_time=3.0,
                        ),
                    ),
                ),
            )
        )

    monkeypatch.setattr(ZMQServerScanService, "scan_ports", scan_ports)

    discovered = OpenHCSServerScanProvider(lambda: config).scan_for_server_logs()

    assert len(discovered) == 2
    assert discovered[0].path == log_path
    assert discovered[0].process_identity == process_identity
    assert discovered[1].path == worker_log_path
    assert discovered[1].process_identity == ProcessIdentity(
        pid=4321,
        create_time=3.0,
    )
    assert captured["timeout_ms"] == config.server_scan_timeout_ms
    assert config.default_port in captured["ports"]
