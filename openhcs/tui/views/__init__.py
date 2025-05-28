"""
TUI Views Package.

View components handle UI rendering and user input.
They delegate business logic to controllers.
"""

from .plate_manager_view import PlateManagerView
from .menu_view import MenuView

__all__ = ['PlateManagerView', 'MenuView']
