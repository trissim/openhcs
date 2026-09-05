#!/usr/bin/env python3
"""
OpenHCS PyQt6 GUI Launcher

Launch script for the OpenHCS PyQt6 GUI application.
Provides command-line interface and application initialization.
"""

from __future__ import annotations

import argparse
from enum import Enum
import importlib.util
import logging
import os
import platform
from pathlib import Path
import sys
import traceback
from typing import Callable, Optional

# CRITICAL: Check for SILENT mode BEFORE any OpenHCS imports
# This prevents logger output during module imports
if "--log-level" in sys.argv:
    log_level_idx = sys.argv.index("--log-level")
    if log_level_idx + 1 < len(sys.argv) and sys.argv[log_level_idx + 1] == "SILENT":
        # Disable ALL logging before any imports
        logging.disable(logging.CRITICAL)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL + 1)

from openhcs.gui_startup import GuiStartupProgressReporterABC
from openhcs import __version__ as OPENHCS_VERSION


def is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    try:
        return "microsoft" in platform.uname().release.lower()
    except Exception:
        return False


class QtPlatformSystem(Enum):
    """Closed host-platform axis for Qt platform setup."""

    MACOS = "Darwin"
    LINUX = "Linux"
    DEFAULT = "default"

    @classmethod
    def from_current(cls) -> "QtPlatformSystem":
        current_system = platform.system()
        for platform_system in (cls.MACOS, cls.LINUX):
            if current_system == platform_system.value:
                return platform_system
        return cls.DEFAULT

    @property
    def uses_default_qt_platform(self) -> bool:
        return self is QtPlatformSystem.DEFAULT


def _setup_macos_qt_platform() -> None:
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    logging.info("macOS detected - setting QT_QPA_PLATFORM=cocoa")

    # Set plugin path to help Qt find the cocoa plugin.
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        return

    try:
        import PyQt6

        pyqt6_path = Path(PyQt6.__file__).parent
        plugin_path = pyqt6_path / "Qt6" / "plugins" / "platforms"
        if plugin_path.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path.parent)
            logging.info(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {plugin_path.parent}")
        else:
            logging.warning(f"PyQt6 plugins directory not found at: {plugin_path}")
    except Exception as e:
        logging.warning(f"Could not set QT_QPA_PLATFORM_PLUGIN_PATH: {e}")


def _setup_linux_qt_platform() -> None:
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    if is_wsl():
        logging.info("WSL2 detected - setting QT_QPA_PLATFORM=xcb")
    else:
        logging.info("Linux detected - setting QT_QPA_PLATFORM=xcb")
    # Disable shared memory for X11 (helps with display issues).
    os.environ["QT_X11_NO_MITSHM"] = "1"


QT_PLATFORM_SETUP: dict[QtPlatformSystem, Callable[[], None]] = {
    QtPlatformSystem.MACOS: _setup_macos_qt_platform,
    QtPlatformSystem.LINUX: _setup_linux_qt_platform,
}


def setup_qt_platform():
    """Setup Qt platform for different environments (macOS, Linux, WSL2, Windows)."""
    # Check if QT_QPA_PLATFORM is already set
    if "QT_QPA_PLATFORM" in os.environ:
        logging.debug(
            f"QT_QPA_PLATFORM already set to: {os.environ['QT_QPA_PLATFORM']}"
        )
        return

    platform_system = QtPlatformSystem.from_current()
    if platform_system.uses_default_qt_platform:
        # Windows and other platforms do not need QT_QPA_PLATFORM set.
        logging.debug(f"Platform {platform.system()} - using default Qt platform")
        return

    QT_PLATFORM_SETUP[platform_system]()


def setup_logging(config, *, log_level=None, log_file=None):
    """Apply the canonical UI logging declaration with optional CLI overrides."""

    from openhcs.pyqt_gui.services.logging_config import configure_gui_logging

    return configure_gui_logging(
        config,
        level_override=log_level,
        log_file_override=log_file,
    )


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="OpenHCS PyQt6 GUI - High-Content Screening Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Launch with default settings
  %(prog)s --log-level DEBUG        # Launch with debug logging
  %(prog)s --config config.json     # Launch with custom config
  %(prog)s --log-file app.log       # Launch with log file
        """,
    )

    from openhcs.pyqt_gui.config import GuiLogLevel

    parser.add_argument(
        "--log-level",
        choices=GuiLogLevel.choices(),
        help="Set logging level (default: INFO). Use SILENT to disable all logging.",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path (default: auto-generated timestamped file)",
    )

    parser.add_argument("--config", type=Path, help="Custom configuration file path")

    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--version", action="version", version=f"OpenHCS PyQt6 GUI {OPENHCS_VERSION}"
    )

    from openhcs.pyqt_gui.services.desktop_update import UPDATE_SESSION_ARGUMENT

    parser.add_argument(
        UPDATE_SESSION_ARGUMENT,
        type=Path,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def load_configuration(config_path: Optional[Path] = None):
    """
    Load application configuration with cache support (matches TUI pattern).

    Args:
        config_path: Optional custom configuration file path

    Returns:
        Global configuration object
    """
    from openhcs.core.config import GlobalPipelineConfig

    try:
        if config_path and config_path.exists():
            # Load custom configuration
            # This would need to be implemented based on config format
            logging.info(f"Loading custom configuration from: {config_path}")
            # For now, use default config
            config = GlobalPipelineConfig()
        else:
            # Load cached configuration (matches TUI pattern)
            from openhcs.core.config_cache import load_cached_global_config_sync

            config = load_cached_global_config_sync()

        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        logging.info("Falling back to default configuration")
        return GlobalPipelineConfig()


def check_dependencies():
    """
    Check for required dependencies.

    Returns:
        True if all dependencies are available, False otherwise
    """
    missing_deps = []

    # Check PyQt6
    try:
        from PyQt6 import QtCore

        logging.debug(f"PyQt6 version: {QtCore.PYQT_VERSION_STR}")
    except ImportError:
        missing_deps.append("PyQt6")

    # Check optional dependencies without importing their runtimes during startup.
    optional_deps = {
        "pyqtgraph": "System monitor graphs",
        "cupy": "GPU acceleration",
        "dill": "Pipeline serialization",
        "psutil": "System monitoring",
    }

    for dep, description in optional_deps.items():
        if importlib.util.find_spec(dep) is not None:
            logging.debug(f"{dep} available for {description}")
        else:
            logging.warning(f"{dep} not available - {description} may be limited")

    if missing_deps:
        logging.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        return False

    return True


def main(
    *,
    arguments: argparse.Namespace | None = None,
    startup_progress: GuiStartupProgressReporterABC | None = None,
):
    """
    Main entry point for the OpenHCS PyQt6 GUI launcher.

    Returns:
        Exit code
    """
    # Parse command line arguments
    args = parse_arguments() if arguments is None else arguments

    from openhcs.utils.environment import OpenHCSProcessEnvironment

    if args.no_gpu:
        OpenHCSProcessEnvironment.enable_cpu_only_mode()

    # Setup Qt platform (must be done before creating QApplication)
    setup_qt_platform()

    try:
        # Check dependencies
        if not check_dependencies():
            logging.error("Dependency check failed")
            if startup_progress is not None:
                startup_progress.fail(
                    "OpenHCS is missing a required desktop dependency.",
                    "Install OpenHCS with the 'gui' extra and try again.",
                )
            return 1

        # Load configuration
        from openhcs.pyqt_gui.config import (
            PyQtGuiRuntimeContext,
            load_cached_ui_config_sync,
        )

        ui_config = load_cached_ui_config_sync()
        log_file = setup_logging(
            ui_config.logging,
            log_level=args.log_level,
            log_file=args.log_file,
        )
        logging.info("Starting OpenHCS PyQt6 GUI...")
        logging.info("Python version: %s", sys.version)
        logging.info("Platform: %s", sys.platform)
        if log_file is not None:
            logging.info("Log file: %s", log_file)

        config = load_configuration(args.config)
        runtime_context = PyQtGuiRuntimeContext(
            ui_config,
            pipeline_runtime=config,
        )

        if OpenHCSProcessEnvironment.cpu_only_mode():
            logging.info("CPU-only execution enabled")

        # Create and run application
        from openhcs.pyqt_gui.app import OpenHCSPyQtApp
        from pyqt_reactive.process_signals import QtProcessSignalRelay
        from pyqt_reactive.utils.window_utils import install_global_window_bounds_filter

        logging.info("Initializing PyQt6 application...")
        app = OpenHCSPyQtApp(sys.argv, runtime_context=runtime_context)
        process_signal_relay = QtProcessSignalRelay(app)
        install_global_window_bounds_filter(app)  # install once, early

        def _main_window_ready() -> None:
            from openhcs.pyqt_gui.services.desktop_update import DesktopRestartSession

            session = (
                DesktopRestartSession(args.restore_update_session)
                if args.restore_update_session is not None
                else DesktopRestartSession.pending()
            )
            if session.directory.exists():
                dialogs = app.main_window.window_services
                try:
                    consumed_session = session.consume()
                except Exception as error:  # noqa: BLE001 - Qt presentation boundary
                    logging.exception("Failed to restore the saved update session")
                    session.present_restore_failure(dialogs, error)
                else:
                    try:
                        restore_outcome = consumed_session.restore(app.main_window)
                    except Exception as error:  # noqa: BLE001 - UI recovery boundary
                        logging.exception("Failed to restore the saved update session")
                        consumed_session.present_restore_failure(dialogs, error)
                    else:
                        restore_outcome.present(dialogs)
            if startup_progress is not None:
                startup_progress.ready()

        def _main_window_failed(error: BaseException) -> None:
            if startup_progress is not None:
                startup_progress.fail(
                    "OpenHCS could not build its main window.",
                    (f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"),
                )

        logging.info("Starting application event loop...")
        try:
            exit_code = app.run(
                on_main_window_ready=_main_window_ready,
                on_startup_failure=_main_window_failed,
            )
        finally:
            process_signal_relay.close()

        logging.info(f"Application exited with code: {exit_code}")
        return exit_code

    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        if startup_progress is not None:
            startup_progress.ready()
        return 130  # Standard exit code for Ctrl+C

    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
        if startup_progress is not None:
            startup_progress.fail(
                "OpenHCS could not start.",
                traceback.format_exc(),
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
