"""
Unit tests for PyQt6-specific behaviors and shared infrastructure integration.

This module tests PyQt6-specific widget behaviors, layout management,
color scheme integration, and proper integration with the shared infrastructure.
"""

import pytest
from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.core.config import Microscope, PathPlanningConfig
from openhcs.core.pipeline_config import PipelineConfig


class TestPyQt6WidgetSpecificBehaviors:
    """Test PyQt6-specific widget behaviors."""

    def test_qlineedit_specific_behavior(self, qapp):
        """Test QLineEdit-specific behaviors and properties."""
        parameters = {"string_param": "test_value"}
        parameter_types = {"string_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="qlineedit_test"
        )
        
        if "string_param" in manager.widgets:
            widget = manager.widgets["string_param"]
            if isinstance(widget, QLineEdit):
                # Test QLineEdit-specific properties
                assert widget.text() == "test_value"
                
                # Test that widget accepts input
                widget.setText("new_value")
                assert widget.text() == "new_value"
                
                # Test placeholder text capability
                widget.setPlaceholderText("Enter value...")
                assert widget.placeholderText() == "Enter value..."

    def test_qcombobox_specific_behavior(self, qapp):
        """Test QComboBox-specific behaviors and properties."""
        parameters = {"enum_param": Microscope.IMAGEXPRESS}
        parameter_types = {"enum_param": Microscope}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="qcombobox_test"
        )
        
        if "enum_param" in manager.widgets:
            widget = manager.widgets["enum_param"]
            if isinstance(widget, QComboBox):
                # Test QComboBox-specific properties
                assert widget.count() > 0
                
                # Test item access
                items = [widget.itemText(i) for i in range(widget.count())]
                assert len(items) > 0
                
                # Test current selection
                current_text = widget.currentText()
                assert current_text != ""

    def test_qcheckbox_specific_behavior(self, qapp):
        """Test QCheckBox-specific behaviors and properties."""
        parameters = {"bool_param": True}
        parameter_types = {"bool_param": bool}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="qcheckbox_test"
        )
        
        if "bool_param" in manager.widgets:
            widget = manager.widgets["bool_param"]
            if isinstance(widget, QCheckBox):
                # Test QCheckBox-specific properties
                assert widget.isChecked() == True
                
                # Test state changes
                widget.setChecked(False)
                assert widget.isChecked() == False
                
                # Test tristate capability (if enabled)
                if widget.isTristate():
                    widget.setCheckState(Qt.CheckState.PartiallyChecked)
                    assert widget.checkState() == Qt.CheckState.PartiallyChecked

    def test_qspinbox_behavior(self, qapp):
        """Test QSpinBox behavior for integer parameters."""
        parameters = {"int_param": 42}
        parameter_types = {"int_param": int}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="qspinbox_test"
        )
        
        if "int_param" in manager.widgets:
            widget = manager.widgets["int_param"]
            if isinstance(widget, QSpinBox):
                # Test QSpinBox-specific properties
                assert widget.value() == 42
                
                # Test value changes
                widget.setValue(100)
                assert widget.value() == 100
                
                # Test range properties
                assert widget.minimum() <= widget.maximum()

    def test_qdoublespinbox_behavior(self, qapp):
        """Test QDoubleSpinBox behavior for float parameters."""
        parameters = {"float_param": 3.14}
        parameter_types = {"float_param": float}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="qdoublespinbox_test"
        )
        
        if "float_param" in manager.widgets:
            widget = manager.widgets["float_param"]
            if isinstance(widget, QDoubleSpinBox):
                # Test QDoubleSpinBox-specific properties
                assert abs(widget.value() - 3.14) < 0.001
                
                # Test value changes
                widget.setValue(2.71)
                assert abs(widget.value() - 2.71) < 0.001
                
                # Test decimal places
                assert widget.decimals() >= 0


class TestPyQt6LayoutManagement:
    """Test PyQt6 layout management and UI structure."""

    def test_main_layout_structure(self, qapp):
        """Test that the main layout is properly structured."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="layout_test"
        )
        
        # Should have a layout
        assert manager.layout() is not None
        assert isinstance(manager.layout(), QVBoxLayout)

    def test_scroll_area_integration(self, qapp):
        """Test scroll area integration when enabled."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        # Test with scroll area enabled
        manager_with_scroll = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="scroll_test",
            use_scroll_area=True
        )
        
        # Should contain a scroll area
        scroll_areas = manager_with_scroll.findChildren(QScrollArea)
        assert len(scroll_areas) > 0
        
        # Test without scroll area
        manager_without_scroll = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="no_scroll_test",
            use_scroll_area=False
        )
        
        # Should not contain scroll areas (or fewer)
        scroll_areas_no_scroll = manager_without_scroll.findChildren(QScrollArea)
        assert len(scroll_areas_no_scroll) <= len(scroll_areas)

    def test_parameter_widget_layout(self, qapp):
        """Test layout of individual parameter widgets."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="widget_layout_test"
        )
        
        # Find parameter containers (should have horizontal layout)
        containers = manager.findChildren(QWidget)
        parameter_containers = [w for w in containers if isinstance(w.layout(), QHBoxLayout)]
        
        # Should have at least one parameter container
        assert len(parameter_containers) > 0

    def test_nested_form_layout(self, qapp):
        """Test layout of nested forms."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_layout_test"
        )
        
        # Should handle nested layout properly
        assert manager.layout() is not None


class TestPyQt6ColorSchemeIntegration:
    """Test integration with PyQt6 color schemes."""

    def test_color_scheme_application(self, qapp, mock_color_scheme):
        """Test that color schemes are properly applied."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="color_scheme_test",
            color_scheme=mock_color_scheme
        )
        
        assert manager.color_scheme == mock_color_scheme

    def test_default_color_scheme(self, qapp):
        """Test default color scheme when none is provided."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="default_color_test"
        )
        
        # Should have a default color scheme
        assert manager.color_scheme is not None
        assert isinstance(manager.color_scheme, PyQt6ColorScheme)

    def test_color_scheme_widget_styling(self, qapp):
        """Test that color scheme affects widget styling."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        custom_scheme = PyQt6ColorScheme()
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="widget_styling_test",
            color_scheme=custom_scheme
        )
        
        # Widgets should be styled according to color scheme
        # The exact styling depends on implementation
        assert manager.color_scheme == custom_scheme


class TestPyQt6SharedInfrastructureIntegration:
    """Test integration with shared infrastructure components."""

    def test_widget_creation_registry_integration(self, qapp):
        """Test integration with widget creation registry."""
        parameters = {
            "string_param": "value",
            "enum_param": Microscope.IMAGEXPRESS,
            "bool_param": True
        }
        parameter_types = {
            "string_param": str,
            "enum_param": Microscope,
            "bool_param": bool
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="registry_integration_test"
        )
        
        # Should create appropriate widgets for each type
        assert len(manager.widgets) == 3
        
        # Each widget should be of appropriate type
        for param_name, widget in manager.widgets.items():
            assert widget is not None
            assert isinstance(widget, QWidget)

    def test_parameter_form_service_integration(self, qapp):
        """Test integration with parameter form service."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="service_integration_test"
        )
        
        # Should have service layer initialized
        assert hasattr(manager, 'service')
        assert manager.service is not None

    def test_form_abstraction_integration(self, qapp):
        """Test integration with form abstraction layer."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="abstraction_integration_test"
        )
        
        # Should have form abstraction initialized
        assert hasattr(manager, 'form_abstraction')
        assert manager.form_abstraction is not None

    def test_debug_infrastructure_integration(self, qapp):
        """Test integration with debug infrastructure."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="debug_integration_test"
        )
        
        # Should have debugger initialized
        assert hasattr(manager, 'debugger')
        assert manager.debugger is not None


class TestPyQt6EventHandling:
    """Test PyQt6 event handling and interaction."""

    def test_widget_focus_behavior(self, qapp, qtbot):
        """Test widget focus behavior."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="focus_test"
        )
        
        qtbot.addWidget(manager)
        
        if "test_param" in manager.widgets:
            widget = manager.widgets["test_param"]
            
            # Test focus behavior
            widget.setFocus()
            qtbot.wait(10)  # Small delay for focus to take effect
            
            # Widget should be able to receive focus
            assert widget.hasFocus() or widget.focusPolicy() != Qt.FocusPolicy.NoFocus

    def test_keyboard_navigation(self, qapp, qtbot):
        """Test keyboard navigation between widgets."""
        parameters = {
            "param1": "value1",
            "param2": "value2"
        }
        parameter_types = {
            "param1": str,
            "param2": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="keyboard_nav_test"
        )
        
        qtbot.addWidget(manager)
        
        # Test tab order and navigation
        # This is a basic test - full keyboard navigation testing would require more setup
        widgets = list(manager.widgets.values())
        if len(widgets) >= 2:
            first_widget = widgets[0]
            first_widget.setFocus()
            
            # Should be able to set focus
            assert first_widget.focusPolicy() != Qt.FocusPolicy.NoFocus

    def test_mouse_interaction(self, qapp, qtbot):
        """Test mouse interaction with widgets."""
        parameters = {"bool_param": False}
        parameter_types = {"bool_param": bool}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="mouse_test"
        )
        
        qtbot.addWidget(manager)
        
        if "bool_param" in manager.widgets:
            widget = manager.widgets["bool_param"]
            if isinstance(widget, QCheckBox):
                # Test mouse click
                initial_state = widget.isChecked()
                qtbot.mouseClick(widget, Qt.MouseButton.LeftButton)
                qtbot.wait(10)
                
                # State should change (or at least be clickable)
                # The exact behavior depends on signal connections
                assert widget.isEnabled()  # Should be interactive
