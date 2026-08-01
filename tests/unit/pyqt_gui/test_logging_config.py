from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from openhcs.pyqt_gui.config import GuiLogLevel, LoggingConfig, UIConfig
from openhcs.pyqt_gui.services.logging_config import configure_gui_logging
from openhcs.pyqt_gui.services.reactor_providers import OpenHCSLogDiscoveryProvider


def _owned_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_openhcs_gui_logging_handler", False)
    ]


def test_logging_config_owns_level_location_destinations_and_rotation(tmp_path) -> None:
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_disable = logging.root.manager.disable
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
        assert sentinel in root_logger.handlers
        assert root_logger.level == logging.WARNING
        assert len(_owned_handlers()) == 1
        file_handler = _owned_handlers()[0]
        assert isinstance(file_handler, RotatingFileHandler)
        assert file_handler.maxBytes == 3 * 1024 * 1024
        assert file_handler.backupCount == 7
    finally:
        for handler in _owned_handlers():
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.removeHandler(sentinel)
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
