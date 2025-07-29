"""PyQt6 clickable help components - clean architecture without circular imports."""

import logging
from typing import Union, Callable, Optional
from PyQt6.QtWidgets import QLabel, QPushButton, QWidget, QHBoxLayout, QGroupBox, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

logger = logging.getLogger(__name__)


class ClickableHelpLabel(QLabel):
    """PyQt6 clickable label that shows help information - reuses Textual TUI help logic."""
    
    help_requested = pyqtSignal()
    
    def __init__(self, text: str, help_target: Union[Callable, type] = None, 
                 param_name: str = None, param_description: str = None, 
                 param_type: type = None, parent=None):
        """Initialize clickable help label.
        
        Args:
            text: Display text for the label
            help_target: Function or class to show help for (for function help)
            param_name: Parameter name (for parameter help)
            param_description: Parameter description (for parameter help)
            param_type: Parameter type (for parameter help)
        """
        # Add help indicator to text
        display_text = f"{text} (?)"
        super().__init__(display_text, parent)
        
        self.help_target = help_target
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        # Style as clickable
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("""
            QLabel {
                color: #0078d4;
                text-decoration: underline;
            }
            QLabel:hover {
                color: #106ebe;
            }
        """)
        
    def mousePressEvent(self, event):
        """Handle mouse press to show help - reuses Textual TUI help manager pattern."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                # Import inside method to avoid circular imports (same pattern as Textual TUI)
                from openhcs.pyqt_gui.windows.help_windows import HelpWindowManager

                if self.help_target:
                    # Show function/class help using unified manager
                    HelpWindowManager.show_docstring_help(self.help_target, parent=self)
                elif self.param_name and self.param_description:
                    # Show parameter help using unified manager
                    HelpWindowManager.show_parameter_help(
                        self.param_name, self.param_description, self.param_type, parent=self
                    )

                self.help_requested.emit()

            except Exception as e:
                logger.error(f"Failed to show help: {e}")
                raise

        super().mousePressEvent(event)


class ClickableFunctionTitle(ClickableHelpLabel):
    """PyQt6 clickable function title that shows function documentation - mirrors Textual TUI."""
    
    def __init__(self, func: Callable, index: int = None, parent=None):
        func_name = getattr(func, '__name__', 'Unknown Function')
        module_name = getattr(func, '__module__', '').split('.')[-1] if func else ''
        
        # Build title text
        title = f"{index + 1}: {func_name}" if index is not None else func_name
        if module_name:
            title += f" ({module_name})"
            
        super().__init__(
            text=title,
            help_target=func,
            parent=parent
        )
        
        # Make title bold
        font = QFont()
        font.setBold(True)
        self.setFont(font)


class ClickableParameterLabel(ClickableHelpLabel):
    """PyQt6 clickable parameter label that shows parameter documentation - mirrors Textual TUI."""
    
    def __init__(self, param_name: str, param_description: str = None, 
                 param_type: type = None, parent=None):
        # Format parameter name nicely
        display_name = param_name.replace('_', ' ').title()
        
        super().__init__(
            text=display_name,
            param_name=param_name,
            param_description=param_description or "No description available",
            param_type=param_type,
            parent=parent
        )


class HelpIndicator(QLabel):
    """PyQt6 simple help indicator that can be added next to any widget - mirrors Textual TUI."""
    
    help_requested = pyqtSignal()
    
    def __init__(self, help_target: Union[Callable, type] = None,
                 param_name: str = None, param_description: str = None,
                 param_type: type = None, parent=None):
        super().__init__("(?)", parent)
        
        self.help_target = help_target
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        # Style as clickable help indicator
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 10px;
                border: 1px solid #666666;
                border-radius: 8px;
                padding: 2px 4px;
                background-color: #f0f0f0;
            }
            QLabel:hover {
                color: #0078d4;
                border-color: #0078d4;
                background-color: #e6f3ff;
            }
        """)
        
        # Set fixed size for consistent appearance
        self.setFixedSize(20, 16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
    def mousePressEvent(self, event):
        """Handle mouse press to show help - reuses Textual TUI help manager pattern."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                # Import inside method to avoid circular imports (same pattern as Textual TUI)
                from openhcs.pyqt_gui.windows.help_windows import HelpWindowManager

                if self.help_target:
                    # Show function/class help using unified manager
                    HelpWindowManager.show_docstring_help(self.help_target, parent=self)
                elif self.param_name and self.param_description:
                    # Show parameter help using unified manager
                    HelpWindowManager.show_parameter_help(
                        self.param_name, self.param_description, self.param_type, parent=self
                    )

                self.help_requested.emit()

            except Exception as e:
                logger.error(f"Failed to show help: {e}")
                raise

        super().mousePressEvent(event)


class HelpButton(QPushButton):
    """PyQt6 help button for adding help functionality to any widget - mirrors Textual TUI."""
    
    def __init__(self, help_target: Union[Callable, type] = None,
                 param_name: str = None, param_description: str = None,
                 param_type: type = None, text: str = "Help", parent=None):
        super().__init__(text, parent)
        
        self.help_target = help_target
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        # Connect click to help display
        self.clicked.connect(self.show_help)
        
        # Style as help button
        self.setMaximumWidth(60)
        self.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
    def show_help(self):
        """Show help using the unified help manager - reuses Textual TUI logic."""
        try:
            # Import inside method to avoid circular imports (same pattern as Textual TUI)
            from openhcs.pyqt_gui.windows.help_windows import HelpWindowManager

            if self.help_target:
                # Show function/class help using unified manager
                HelpWindowManager.show_docstring_help(self.help_target, parent=self)
            elif self.param_name and self.param_description:
                # Show parameter help using unified manager
                HelpWindowManager.show_parameter_help(
                    self.param_name, self.param_description, self.param_type, parent=self
                )

        except Exception as e:
            logger.error(f"Failed to show help: {e}")
            raise


class LabelWithHelp(QWidget):
    """PyQt6 widget that combines a label with a help indicator - mirrors Textual TUI pattern."""
    
    def __init__(self, text: str, help_target: Union[Callable, type] = None,
                 param_name: str = None, param_description: str = None,
                 param_type: type = None, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Main label
        label = QLabel(text)
        layout.addWidget(label)
        
        # Help indicator
        help_indicator = HelpIndicator(
            help_target=help_target,
            param_name=param_name,
            param_description=param_description,
            param_type=param_type
        )
        layout.addWidget(help_indicator)
        
        layout.addStretch()


class FunctionTitleWithHelp(QWidget):
    """PyQt6 function title with integrated help - mirrors Textual TUI ClickableFunctionTitle."""
    
    def __init__(self, func: Callable, index: int = None, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Function title
        func_name = getattr(func, '__name__', 'Unknown Function')
        module_name = getattr(func, '__module__', '').split('.')[-1] if func else ''
        
        title = f"{index + 1}: {func_name}" if index is not None else func_name
        if module_name:
            title += f" ({module_name})"
            
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Help button
        help_btn = HelpButton(help_target=func, text="?")
        help_btn.setMaximumWidth(25)
        help_btn.setMaximumHeight(20)
        layout.addWidget(help_btn)
        
        layout.addStretch()


class GroupBoxWithHelp(QGroupBox):
    """PyQt6 group box with integrated help for dataclass titles - mirrors Textual TUI pattern."""

    def __init__(self, title: str, help_target: Union[Callable, type] = None, parent=None):
        super().__init__(parent)

        self.help_target = help_target

        # Create custom title widget with help
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        # Title label
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)

        # Help button for dataclass
        if help_target:
            help_btn = HelpButton(help_target=help_target, text="?")
            help_btn.setMaximumWidth(25)
            help_btn.setMaximumHeight(20)
            title_layout.addWidget(help_btn)

        title_layout.addStretch()

        # Set the custom title widget
        self.setTitle("")  # Clear default title

        # Create main layout and add title widget at top
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(title_widget)

        # Content area for child widgets
        self.content_layout = QVBoxLayout()
        main_layout.addLayout(self.content_layout)

    def addWidget(self, widget):
        """Add widget to the content area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Add layout to the content area."""
        self.content_layout.addLayout(layout)
