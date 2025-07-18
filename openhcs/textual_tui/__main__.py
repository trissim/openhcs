"""
OpenHCS Textual TUI Entry Point

Entry point for the OpenHCS Textual TUI application.
Replicates the existing pattern from the prompt-toolkit TUI.
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
import subprocess
import tempfile
from pathlib import Path

from openhcs.core.config import get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry

from .app import OpenHCSTUIApp


def _parse_command_line_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenHCS Textual TUI - Modern Terminal User Interface"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace directory for outputs (optional)"
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the TUI via textual-web instead of running locally"
    )

    return parser.parse_args()


def _serve_web():
    """Serve the TUI via textual-serve."""
    try:
        # Check if textual-serve is installed
        import textual_serve
        print("✅ textual-serve found")
    except ImportError:
        print("❌ textual-serve is not installed!")
        print("📦 Install it with: pip install textual-serve")
        sys.exit(1)

    try:
        print("🌐 Starting OpenHCS web server...")
        print("🔗 Your TUI will be available at: http://localhost:8000")
        print("📝 Share this URL to give others access to your OpenHCS TUI")
        print("⚠️  Note: The TUI runs on YOUR machine, others just see it in their browser")
        print()

        # Use textual-serve to serve the TUI
        from textual_serve.server import Server

        server = Server(
            command="python -m openhcs.textual_tui",
            host="localhost",
            port=8000,
            title="OpenHCS - High-Content Screening Platform"
        )

        server.serve()

    except KeyboardInterrupt:
        print("\n🛑 Web server stopped by user")
    except Exception as e:
        print(f"❌ Error running textual-serve: {e}")
        sys.exit(1)


def _setup_logging(debug: bool = False):
    """Setup unified logging configuration for entire OpenHCS system."""
    log_level = logging.DEBUG if debug else logging.INFO

    # Create logs directory
    log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    import time
    log_file = log_dir / f"openhcs_unified_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Setup unified logging for entire OpenHCS system
    root_logger = logging.getLogger()

    # Clear any existing handlers to ensure clean state
    root_logger.handlers.clear()

    # Setup file-only logging (no console output for TUI)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # Prevent other modules from adding console handlers
    logging.basicConfig = lambda *args, **kwargs: None

    # Set OpenHCS logger level for all components
    logging.getLogger("openhcs").setLevel(log_level)
    logger = logging.getLogger("openhcs.main")
    logger.info(f"OpenHCS unified logging started - Level: {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {log_file}")
    return logger


def main():
    """Main entry point for OpenHCS Textual TUI."""
    args = _parse_command_line_arguments()

    # If web flag is set, serve via textual-serve instead
    if args.web:
        _serve_web()
        return

    # For TUI mode, run async
    asyncio.run(main_async(args))


def _setup_signal_handlers():
    """Setup signal handlers for clean shutdown."""
    import signal
    import threading
    import os

    def force_cleanup(signum, frame):
        """Force cleanup all threads on signal."""
        print("\nForcing immediate exit...")

        # Try to cleanup background threads first
        try:
            active_threads = [t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()]
            if active_threads:
                print(f"Cleaning up {len(active_threads)} background threads...")
                # Can't set daemon on running threads, just note them and force exit
        except:
            pass

        # Force immediate exit
        os._exit(0)

    signal.signal(signal.SIGINT, force_cleanup)
    signal.signal(signal.SIGTERM, force_cleanup)

async def main_async(args):
    """Async main function for TUI mode."""

    # Setup signal handlers for clean shutdown
    _setup_signal_handlers()

    # Set multiprocessing start method FIRST, before any other initialization
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError:
        # Already set, check if it's spawn
        current_method = multiprocessing.get_start_method()
        if current_method != 'spawn':
            print(f"Warning: Multiprocessing start method is '{current_method}', not 'spawn'. CUDA may not work in worker processes.")
        else:
            print("Multiprocessing start method already set to 'spawn'")

    logger = _setup_logging(args.debug)
    
    try:
        # Load global configuration with cache support
        from openhcs.textual_tui.services.global_config_cache import load_cached_global_config
        global_config = await load_cached_global_config()
        logger.info("Global configuration loaded")

        # Setup GPU registry
        setup_global_gpu_registry(global_config=global_config)
        logger.info("GPU registry setup completed")

        # Create and run the Textual app
        app = OpenHCSTUIApp(global_config=global_config)
        logger.info("Starting OpenHCS Textual TUI application...")

        # Run the app with a timeout wrapper to prevent hanging
        try:
            await asyncio.wait_for(app.run_async(), timeout=None)  # No timeout for normal operation
        except asyncio.TimeoutError:
            logger.warning("App run timed out, forcing exit")
            import os
            os._exit(0)

    except KeyboardInterrupt:
        logger.info("TUI terminated by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unhandled error in TUI: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("OpenHCS Textual TUI finished")

        # Final cleanup check - force exit if we're still hanging
        try:
            import threading
            import time
            time.sleep(0.1)
            active_threads = [t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()]
            if active_threads:
                logger.warning(f"Final cleanup: {len(active_threads)} threads still active, forcing exit")
                import os
                os._exit(0)
        except (ImportError, AttributeError, OSError) as cleanup_error:
            # If final cleanup fails, just log it - we're exiting anyway
            logger.debug(f"Final cleanup failed (non-critical): {cleanup_error}")


if __name__ == "__main__":
    main()
