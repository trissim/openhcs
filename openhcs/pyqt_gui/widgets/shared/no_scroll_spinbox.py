"""
No-scroll spinbox widgets for PyQt6.

Prevents accidental value changes from mouse wheel events.
"""

from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox
from PyQt6.QtGui import QWheelEvent


class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()
