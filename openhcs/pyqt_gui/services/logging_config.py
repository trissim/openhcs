"""Runtime owner for the process-global GUI logging declaration."""

from __future__ import annotations

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from openhcs.pyqt_gui.config import GuiLogLevel, LoggingConfig

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class GuiLoggingHandler:
    """Nominal marker for root handlers owned by the GUI declaration."""


class GuiConsoleHandler(logging.StreamHandler, GuiLoggingHandler):
    """Console destination owned by ``LoggingConfig``."""


class GuiRotatingFileHandler(RotatingFileHandler, GuiLoggingHandler):
    """Rotating file destination owned by ``LoggingConfig``."""


def configure_gui_logging(
    config: LoggingConfig,
    *,
    level_override: str | None = None,
    log_file_override: Path | None = None,
) -> Path | None:
    """Atomically replace root handlers from one logging declaration.

    CLI values are ephemeral launch overrides. They alter the applied handler
    set without mutating or mirroring the persisted ``LoggingConfig``.
    """

    level = (
        GuiLogLevel.from_text(level_override)
        if level_override is not None
        else config.level
    )
    formatter = logging.Formatter(_FORMAT)
    handlers: list[logging.Handler] = []
    log_file: Path | None = None

    if level is not GuiLogLevel.SILENT:
        if config.enable_console_logging:
            console_handler = GuiConsoleHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level.logging_level)
            handlers.append(console_handler)

        if config.enable_file_logging or log_file_override is not None:
            if log_file_override is None:
                log_directory = config.resolved_log_directory()
                log_file = log_directory / (
                    f"openhcs_unified_{time.strftime('%Y%m%d_%H%M%S')}.log"
                )
            else:
                log_file = log_file_override.expanduser().resolve(strict=False)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = GuiRotatingFileHandler(
                log_file,
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level.logging_level)
            handlers.append(file_handler)

    root_logger = logging.getLogger()
    previous_handlers = tuple(root_logger.handlers)
    root_logger.handlers = handlers
    if level is GuiLogLevel.SILENT:
        logging.disable(logging.CRITICAL)
        root_logger.setLevel(logging.CRITICAL + 1)
    else:
        logging.disable(logging.NOTSET)
        root_logger.setLevel(level.logging_level)
        logging.getLogger("openhcs").setLevel(level.logging_level)
        logging.getLogger("PIL").setLevel(logging.WARNING)

    for handler in previous_handlers:
        if (
            handler not in handlers
            and isinstance(handler, GuiLoggingHandler)
        ):
            handler.close()

    return log_file
