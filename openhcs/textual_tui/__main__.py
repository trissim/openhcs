"""
OpenHCS Textual TUI Entry Point

Entry point for the OpenHCS Textual TUI application.
Replicates the existing pattern from the prompt-toolkit TUI.
"""

import argparse
import asyncio
import logging
import sys
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
    
    return parser.parse_args()


def _setup_logging(debug: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set OpenHCS logger level
    logging.getLogger("openhcs").setLevel(log_level)
    logger = logging.getLogger("openhcs.textual_tui.main")
    logger.info(f"OpenHCS Textual TUI starting with log level: {logging.getLevelName(log_level)}")
    return logger


async def main():
    """Main entry point for OpenHCS Textual TUI."""
    args = _parse_command_line_arguments()
    logger = _setup_logging(args.debug)
    
    try:
        # Load global configuration
        global_config = get_default_global_config()
        logger.info("Global configuration loaded")
        
        # Setup GPU registry
        setup_global_gpu_registry(global_config=global_config)
        logger.info("GPU registry setup completed")
        
        # Create and run the Textual app
        app = OpenHCSTUIApp(global_config=global_config)
        logger.info("Starting OpenHCS Textual TUI application...")
        
        await app.run_async()
        
    except KeyboardInterrupt:
        logger.info("TUI terminated by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unhandled error in TUI: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        logger.info("OpenHCS Textual TUI finished")


if __name__ == "__main__":
    asyncio.run(main())
