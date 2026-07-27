#!/usr/bin/env python3
"""
OpenHCS PyQt6 GUI - Module Entry Point

Allows running the PyQt6 GUI directly with:
    python -m openhcs.pyqt_gui

This is a convenience wrapper around the launch script.
"""

import sys
import logging

# CRITICAL: Check for SILENT mode BEFORE any other imports
# This must be at MODULE LEVEL to run before launch.py is imported
if '--log-level' in sys.argv:
    log_level_idx = sys.argv.index('--log-level')
    if log_level_idx + 1 < len(sys.argv) and sys.argv[log_level_idx + 1] == 'SILENT':
        # Disable ALL logging before any imports
        logging.disable(logging.CRITICAL)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL + 1)

def main():
    """Main entry point with graceful error handling for missing GUI dependencies."""
    from openhcs.gui_startup import main as startup_main

    return startup_main()


if __name__ == "__main__":
    sys.exit(main())
