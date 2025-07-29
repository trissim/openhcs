#!/usr/bin/env python3
"""
OpenHCS PyQt6 GUI - Module Entry Point

Allows running the PyQt6 GUI directly with:
    python -m openhcs.pyqt_gui

This is a convenience wrapper around the launch script.
"""

import sys
from pathlib import Path

# Import the main function from launch script
try:
    from openhcs.pyqt_gui.launch import main
except ImportError:
    # Handle case where package isn't properly installed
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from openhcs.pyqt_gui.launch import main

if __name__ == "__main__":
    sys.exit(main())
