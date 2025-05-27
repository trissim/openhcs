#!/usr/bin/env python3
"""
Main entry point for OpenHCS TUI.

This module allows the TUI to be run with:
    python -m openhcs.tui

It provides a clean entry point that initializes and runs the hybrid TUI application.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openhcs_tui.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the TUI application."""
    try:
        logger.info("Starting OpenHCS TUI...")

        # Use the main entry point from main.py
        from .main import run_tui
        run_tui()

    except KeyboardInterrupt:
        logger.info("TUI interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start TUI: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
