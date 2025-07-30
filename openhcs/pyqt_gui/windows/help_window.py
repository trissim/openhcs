"""
Help Window for PyQt6

Help display dialog with OpenHCS documentation and usage information.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QTextEdit, QFrame, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class HelpWindow(QDialog):
    """
    PyQt6 Help Window.
    
    Help display dialog with OpenHCS documentation and usage information.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Help content (extracted from Textual version)
    HELP_TEXT = """OpenHCS - Open High-Content Screening

🔬 Visual Programming for Cell Biology Research

Key Features:
• GPU-accelerated image processing
• Visual pipeline building
• Multi-backend storage support
• Real-time parameter editing

Workflow:
1. Add Plate → Select microscopy data
2. Edit Step → Visual function selection
3. Compile → Create execution plan
4. Run → Process images

Navigation:
• Use the Plate Manager to add and manage your microscopy plates
• Build processing pipelines in the Pipeline Editor
• Configure functions in the Function Library
• Monitor system performance in real-time

Keyboard Shortcuts:
• Ctrl+N: New pipeline
• Ctrl+O: Open pipeline
• Ctrl+S: Save pipeline
• F1: Show this help
• Esc: Close current dialog

Tips:
• Right-click on widgets for context menus
• Use drag-and-drop to reorder pipeline steps
• Parameters are validated in real-time
• GPU acceleration is automatic when available

For detailed documentation, see the Nature Methods publication
and the online documentation at https://openhcs.org

Version: PyQt6 GUI 1.0.0
"""
    
    def __init__(self, content: Optional[str] = None,
                 color_scheme: Optional[PyQt6ColorScheme] = None, parent=None):
        """
        Initialize the help window.

        Args:
            content: Custom help content (uses default if None)
            color_scheme: Color scheme for styling (optional, uses default if None)
            parent: Parent widget
        """
        super().__init__(parent)

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)

        # Business logic state
        self.content = content or self.HELP_TEXT
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        logger.debug("Help window initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("OpenHCS Help")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(800, 700)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header with OpenHCS logo/title
        header_frame = self.create_header()
        layout.addWidget(header_frame)
        
        # Scrollable content area
        content_area = self.create_content_area()
        layout.addWidget(content_area)
        
        # Button panel
        button_panel = self.create_button_panel()
        layout.addWidget(button_panel)
        
        # Apply centralized styling
        self.setStyleSheet(self.style_generator.generate_dialog_style() + f"""
            QTextEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }}
        """)
    
    def create_header(self) -> QWidget:
        """
        Create the header section with title and version.
        
        Returns:
            Widget containing header information
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 5px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        
        # ASCII art title
        title_label = QLabel(self.get_ascii_title())
        title_label.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Visual Programming for Cell Biology Research")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)
        
        return frame
    
    def create_content_area(self) -> QWidget:
        """
        Create the scrollable content area.
        
        Returns:
            Widget containing help content
        """
        # Text edit for help content
        self.content_text = QTextEdit()
        self.content_text.setPlainText(self.content)
        self.content_text.setReadOnly(True)
        
        # Enable rich text formatting
        self.content_text.setHtml(self.format_help_content(self.content))
        
        return self.content_text
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 10px;
            }}
        """)
        
        layout = QHBoxLayout(panel)
        layout.addStretch()
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setMinimumHeight(35)
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 1px solid #106ebe;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        layout.addWidget(close_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        pass  # No additional connections needed
    
    def get_ascii_title(self) -> str:
        """
        Get ASCII art title.
        
        Returns:
            ASCII art title string
        """
        return """
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗  ██╗ ██████╗███████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║  ██║██╔════╝██╔════╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║     ███████╗
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║     ╚════██║
╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║╚██████╗███████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝╚══════╝
        """
    
    def format_help_content(self, content: str) -> str:
        """
        Format help content with HTML for better display.
        
        Args:
            content: Raw help content
            
        Returns:
            HTML formatted content
        """
        # Convert plain text to HTML with basic formatting
        html_content = content.replace('\n', '<br>')
        
        # Format headers (lines starting with uppercase words followed by colon)
        import re
        accent_color = self.color_scheme.to_hex(self.color_scheme.text_accent)
        html_content = re.sub(
            r'^([A-Z][A-Za-z\s]+:)$',
            rf'<h3 style="color: {accent_color}; margin-top: 20px; margin-bottom: 10px;">\1</h3>',
            html_content,
            flags=re.MULTILINE
        )
        
        # Format bullet points
        secondary_color = self.color_scheme.to_hex(self.color_scheme.text_secondary)
        html_content = re.sub(
            r'^• (.+)$',
            rf'<li style="margin-left: 20px; color: {secondary_color};">\1</li>',
            html_content,
            flags=re.MULTILINE
        )
        
        # Format numbered lists
        html_content = re.sub(
            r'^(\d+\.) (.+)$',
            rf'<div style="margin-left: 20px; color: {secondary_color};"><strong style="color: {accent_color};">\1</strong> \2</div>',
            html_content,
            flags=re.MULTILINE
        )
        
        # Format keyboard shortcuts
        input_bg_color = self.color_scheme.to_hex(self.color_scheme.input_bg)
        text_accent_color = self.color_scheme.to_hex(self.color_scheme.text_accent)
        html_content = re.sub(
            r'(Ctrl\+[A-Z]|F\d+|Esc)',
            rf'<code style="background-color: {input_bg_color}; padding: 2px 4px; border-radius: 3px; color: {text_accent_color};">\1</code>',
            html_content
        )
        
        # Wrap in HTML structure
        primary_text_color = self.color_scheme.to_hex(self.color_scheme.text_primary)
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; font-size: 12px; line-height: 1.6; color: {primary_text_color};">
        {html_content}
        </body>
        </html>
        """
        
        return html_content
    
    def set_content(self, content: str):
        """
        Set new help content.
        
        Args:
            content: New help content
        """
        self.content = content
        if hasattr(self, 'content_text'):
            self.content_text.setHtml(self.format_help_content(content))
    
    def show_section(self, section_name: str):
        """
        Show specific help section.
        
        Args:
            section_name: Name of the section to show
        """
        # This could be extended to show specific sections
        # For now, just show the full help
        self.show()
    
    @staticmethod
    def show_help(parent=None, content: Optional[str] = None):
        """
        Static method to show help window.
        
        Args:
            parent: Parent widget
            content: Custom help content
            
        Returns:
            Dialog result
        """
        help_window = HelpWindow(content, parent)
        return help_window.exec()


# Convenience function for showing help
def show_help_dialog(parent=None, content: Optional[str] = None):
    """
    Show help dialog.
    
    Args:
        parent: Parent widget
        content: Custom help content
        
    Returns:
        Dialog result
    """
    return HelpWindow.show_help(parent, content)
