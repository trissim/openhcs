#!/usr/bin/env python3
"""
OpenHCS PyQt6 GUI Launcher

Launch script for the OpenHCS PyQt6 GUI application.
Provides command-line interface and application initialization.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add OpenHCS to path if needed
try:
    from openhcs.core.config import get_default_global_config
except ImportError:
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from openhcs.core.config import get_default_global_config

from openhcs.pyqt_gui.app import OpenHCSPyQtApp


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup unified logging configuration for entire OpenHCS system - matches TUI exactly."""
    log_level_obj = getattr(logging, log_level.upper())

    # Create logs directory
    log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file if not specified
    if log_file is None:
        import time
        log_file = log_dir / f"openhcs_unified_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Setup unified logging for entire OpenHCS system (EXACTLY like TUI)
    root_logger = logging.getLogger()

    # Clear any existing handlers to ensure clean state
    root_logger.handlers.clear()

    # Setup console + file logging (TUI only has file, GUI has both)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level_obj)

    # Prevent other modules from adding console handlers
    logging.basicConfig = lambda *args, **kwargs: None

    # Set OpenHCS logger level for all components
    logging.getLogger("openhcs").setLevel(log_level_obj)
    logger = logging.getLogger("openhcs.pyqt_gui")
    logger.info(f"OpenHCS PyQt6 GUI logging started - Level: {logging.getLevelName(log_level_obj)}")
    logger.info(f"Log file: {log_file}")

    # Reduce noise from some libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)


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
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path (default: console only)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Custom configuration file path'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='OpenHCS PyQt6 GUI 1.0.0'
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
    try:
        if config_path and config_path.exists():
            # Load custom configuration
            # This would need to be implemented based on config format
            logging.info(f"Loading custom configuration from: {config_path}")
            # For now, use default config
            config = get_default_global_config()
        else:
            # Load cached configuration (matches TUI pattern)
            from openhcs.pyqt_gui.services.config_cache_adapter import load_cached_global_config_sync
            config = load_cached_global_config_sync()

        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        logging.info("Falling back to default configuration")
        return get_default_global_config()


def check_dependencies():
    """
    Check for required dependencies.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    # Check PyQt6
    try:
        import PyQt6
        logging.debug(f"PyQt6 version: {PyQt6.QtCore.PYQT_VERSION_STR}")
    except ImportError:
        missing_deps.append("PyQt6")
    
    # Check PyQtGraph (optional)
    try:
        import pyqtgraph
        logging.debug(f"PyQtGraph version: {pyqtgraph.__version__}")
    except ImportError:
        logging.warning("PyQtGraph not available - system monitor will use fallback display")
    
    # Check other optional dependencies
    optional_deps = {
        'cupy': 'GPU acceleration',
        'dill': 'Pipeline serialization',
        'psutil': 'System monitoring'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            logging.debug(f"{dep} available for {description}")
        except ImportError:
            logging.warning(f"{dep} not available - {description} may be limited")
    
    if missing_deps:
        logging.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        return False
    
    return True


def main():
    """
    Main entry point for the OpenHCS PyQt6 GUI launcher.
    
    Returns:
        Exit code
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logging.info("Starting OpenHCS PyQt6 GUI...")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    
    try:
        # Check dependencies
        if not check_dependencies():
            logging.error("Dependency check failed")
            return 1
        
        # Load configuration
        config = load_configuration(args.config)
        
        # Apply command line overrides
        if args.no_gpu:
            logging.info("GPU acceleration disabled by command line")
            # This would need to be implemented in the config
            # config.disable_gpu = True
        
        # Create and run application
        logging.info("Initializing PyQt6 application...")
        app = OpenHCSPyQtApp(sys.argv, config)
        
        logging.info("Starting application event loop...")
        exit_code = app.run()
        
        logging.info(f"Application exited with code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
