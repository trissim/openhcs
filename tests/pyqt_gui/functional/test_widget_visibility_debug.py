"""
Debug test to investigate widget visibility issues in PyQt6 parameter forms.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QWidget, QApplication
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager


class TestWidgetVisibilityDebug:
    """Debug widget visibility issues."""

    def test_form_manager_widget_hierarchy(self, qtbot):
        """Debug the widget hierarchy and visibility."""
        parameters = {"test_param": "initial_value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="debug_test"
        )
        
        qtbot.addWidget(manager)
        
        # Show the manager widget explicitly
        manager.show()
        
        # Debug widget hierarchy
        print(f"\nManager widget: {manager}")
        print(f"Manager visible: {manager.isVisible()}")
        print(f"Manager size: {manager.size()}")
        print(f"Manager layout: {manager.layout()}")
        
        # Find all child widgets
        all_widgets = manager.findChildren(QWidget)
        print(f"\nTotal child widgets: {len(all_widgets)}")
        
        # Find reset buttons specifically
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        print(f"Reset buttons found: {len(reset_buttons)}")
        
        for i, btn in enumerate(reset_buttons):
            print(f"  Button {i}: {btn}")
            print(f"    Text: '{btn.text()}'")
            print(f"    Visible: {btn.isVisible()}")
            print(f"    Enabled: {btn.isEnabled()}")
            print(f"    Size: {btn.size()}")
            print(f"    Parent: {btn.parent()}")
            print(f"    Parent visible: {btn.parent().isVisible() if btn.parent() else 'No parent'}")
        
        # Find line edit widgets
        line_edits = manager.findChildren(QLineEdit)
        print(f"\nLine edit widgets found: {len(line_edits)}")
        
        for i, edit in enumerate(line_edits):
            print(f"  LineEdit {i}: {edit}")
            print(f"    Text: '{edit.text()}'")
            print(f"    Visible: {edit.isVisible()}")
            print(f"    Size: {edit.size()}")
            print(f"    Parent: {edit.parent()}")
        
        # Check if manager needs to be shown
        if not manager.isVisible():
            print("\nManager is not visible - showing it...")
            manager.show()
            qtbot.wait(100)  # Wait for show to take effect
            
            print(f"After show() - Manager visible: {manager.isVisible()}")
            
            # Re-check reset buttons
            for i, btn in enumerate(reset_buttons):
                print(f"  Button {i} after show - Visible: {btn.isVisible()}")

    def test_manual_widget_creation_and_visibility(self, qtbot):
        """Test manual widget creation to isolate the issue."""
        # Create a simple form manually
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create a parameter row manually
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        
        # Add label
        label = QLabel("Test Parameter:")
        row_layout.addWidget(label)
        
        # Add input
        input_widget = QLineEdit("test_value")
        row_layout.addWidget(input_widget)
        
        # Add reset button
        reset_button = QPushButton("Reset")
        row_layout.addWidget(reset_button)
        
        # Add row to main layout
        main_layout.addWidget(row_widget)
        
        qtbot.addWidget(main_widget)
        main_widget.show()
        
        # Check visibility
        print(f"\nManual widget test:")
        print(f"Main widget visible: {main_widget.isVisible()}")
        print(f"Row widget visible: {row_widget.isVisible()}")
        print(f"Label visible: {label.isVisible()}")
        print(f"Input visible: {input_widget.isVisible()}")
        print(f"Reset button visible: {reset_button.isVisible()}")
        
        # All should be visible
        assert main_widget.isVisible()
        assert row_widget.isVisible()
        assert label.isVisible()
        assert input_widget.isVisible()
        assert reset_button.isVisible()

    def test_parameter_form_manager_show_method(self, qtbot):
        """Test if calling show() on the parameter form manager fixes visibility."""
        parameters = {"test_param": "initial_value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="show_test"
        )
        
        qtbot.addWidget(manager)
        
        # Explicitly show the manager
        manager.show()
        qtbot.wait(100)  # Wait for UI to update
        
        # Now check reset button visibility
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        
        assert len(reset_buttons) > 0, "Should have at least one reset button"
        
        reset_button = reset_buttons[0]
        print(f"\nAfter explicit show():")
        print(f"Manager visible: {manager.isVisible()}")
        print(f"Reset button visible: {reset_button.isVisible()}")
        
        # This should now pass
        assert manager.isVisible()
        assert reset_button.isVisible()
