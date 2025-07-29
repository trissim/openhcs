"""
Status Bar Widget for PyQt6

Status display with system information and current operation status.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class StatusBarWidget(QWidget):
    """
    PyQt6 Status Bar Widget.
    
    Displays current status, progress, and system information.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    status_updated = pyqtSignal(str)  # status message
    
    def __init__(self, parent=None):
        """
        Initialize the status bar widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Business logic state
        self.current_status = "Ready"
        self.current_operation = ""
        self.progress_value = 0
        self.progress_visible = False
        
        # UI components
        self.status_label: Optional[QLabel] = None
        self.operation_label: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.time_label: Optional[QLabel] = None
        
        # Timer for time updates
        self.time_timer = QTimer()
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.start_time_updates()
        
        logger.debug("Status bar widget initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)
        
        # Status section
        status_frame = self.create_status_section()
        layout.addWidget(status_frame)
        
        # Progress section
        progress_frame = self.create_progress_section()
        layout.addWidget(progress_frame)
        
        # Spacer
        layout.addStretch()
        
        # Time section
        time_frame = self.create_time_section()
        layout.addWidget(time_frame)
        
        # Set styling
        self.setStyleSheet("""
            StatusBarWidget {
                background-color: #1e1e1e;
                border-top: 1px solid #555555;
                color: white;
            }
        """)
        
        # Set fixed height
        self.setFixedHeight(30)
    
    def create_status_section(self) -> QWidget:
        """
        Create the status message section.
        
        Returns:
            Widget containing status information
        """
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status icon
        status_icon = QLabel("●")
        status_icon.setStyleSheet("color: #00ff00; font-weight: bold;")
        layout.addWidget(status_icon)
        
        # Status label
        self.status_label = QLabel(self.current_status)
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(self.status_label)
        
        return frame
    
    def create_progress_section(self) -> QWidget:
        """
        Create the progress section.
        
        Returns:
            Widget containing progress information
        """
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Operation label
        self.operation_label = QLabel("")
        self.operation_label.setFont(QFont("Arial", 8))
        self.operation_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.operation_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 2px;
                background-color: #2b2b2b;
                color: white;
                text-align: center;
                font-size: 8px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 1px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return frame
    
    def create_time_section(self) -> QWidget:
        """
        Create the time display section.
        
        Returns:
            Widget containing time information
        """
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Time label
        self.time_label = QLabel("")
        self.time_label.setFont(QFont("Arial", 8))
        self.time_label.setStyleSheet("color: #888888;")
        self.time_label.setMinimumWidth(120)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.time_label)
        
        return frame
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        self.status_updated.connect(self.update_status_display)
        self.time_timer.timeout.connect(self.update_time_display)
    
    def start_time_updates(self):
        """Start the time update timer."""
        self.time_timer.start(1000)  # Update every second
        self.update_time_display()  # Initial update
    
    def stop_time_updates(self):
        """Stop the time update timer."""
        self.time_timer.stop()
    
    def update_time_display(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def set_status(self, message: str, status_type: str = "info"):
        """
        Set status message with type.
        
        Args:
            message: Status message
            status_type: Type of status (info, warning, error, success)
        """
        self.current_status = message
        self.status_updated.emit(message)
        
        # Update status icon color based on type
        if hasattr(self, 'status_label') and self.status_label:
            parent_frame = self.status_label.parent()
            if parent_frame:
                status_icon = parent_frame.findChild(QLabel)
                if status_icon and status_icon.text() == "●":
                    color_map = {
                        "info": "#00ff00",
                        "warning": "#ffaa00", 
                        "error": "#ff0000",
                        "success": "#00ff00"
                    }
                    color = color_map.get(status_type, "#00ff00")
                    status_icon.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        logger.debug(f"Status updated: {message} ({status_type})")
    
    def set_operation(self, operation: str):
        """
        Set current operation description.
        
        Args:
            operation: Operation description
        """
        self.current_operation = operation
        if self.operation_label:
            self.operation_label.setText(operation)
        
        logger.debug(f"Operation updated: {operation}")
    
    def show_progress(self, value: int = 0, maximum: int = 100):
        """
        Show progress bar with value.
        
        Args:
            value: Current progress value
            maximum: Maximum progress value
        """
        self.progress_visible = True
        self.progress_value = value
        
        if self.progress_bar:
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
            self.progress_bar.setVisible(True)
        
        logger.debug(f"Progress shown: {value}/{maximum}")
    
    def update_progress(self, value: int):
        """
        Update progress value.
        
        Args:
            value: New progress value
        """
        self.progress_value = value
        
        if self.progress_bar and self.progress_visible:
            self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_visible = False
        
        if self.progress_bar:
            self.progress_bar.setVisible(False)
        
        if self.operation_label:
            self.operation_label.setText("")
        
        logger.debug("Progress hidden")
    
    def update_status_display(self, message: str):
        """
        Update the status display.
        
        Args:
            message: Status message to display
        """
        if self.status_label:
            self.status_label.setText(message)
    
    def set_info_status(self, message: str):
        """Set info status message."""
        self.set_status(message, "info")
    
    def set_warning_status(self, message: str):
        """Set warning status message."""
        self.set_status(message, "warning")
    
    def set_error_status(self, message: str):
        """Set error status message."""
        self.set_status(message, "error")
    
    def set_success_status(self, message: str):
        """Set success status message."""
        self.set_status(message, "success")
    
    def clear_status(self):
        """Clear status message."""
        self.set_status("Ready", "info")
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_time_updates()
        event.accept()
