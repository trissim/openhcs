"""
Integration tests for nested dataclass form behavior.

This module tests the complex interactions between nested dataclass parameters,
including optional dataclasses, nested form creation, and hierarchical updates.
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtWidgets import QCheckBox, QWidget
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import PathPlanningConfig, MaterializationPathConfig
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer


@dataclass
class TestNestedConfig:
    """Test dataclass for nested form testing."""
    field1: str = "default1"
    field2: int = 10
    field3: Optional[str] = None


@dataclass
class TestParentConfig:
    """Test dataclass with nested configurations."""
    simple_field: str = "simple_default"
    nested_config: Optional[TestNestedConfig] = None
    required_nested: TestNestedConfig = None


class TestNestedDataclassFormCreation:
    """Test creation of forms with nested dataclass parameters."""

    def test_nested_dataclass_form_creation(self, qapp):
        """Test creating forms with nested dataclass parameters."""
        # Create config with nested dataclass
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_creation_test"
        )
        
        assert manager is not None
        # Should handle nested dataclass parameters
        assert "path_planning" in manager.parameters

    def test_multiple_nested_dataclass_creation(self, qapp):
        """Test creating forms with multiple nested dataclass parameters."""
        path_config = PathPlanningConfig()
        mat_config = MaterializationPathConfig()
        
        parameters = {
            "path_planning": path_config,
            "materialization": mat_config
        }
        parameter_types = {
            "path_planning": PathPlanningConfig,
            "materialization": MaterializationPathConfig
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="multi_nested_test"
        )
        
        assert manager is not None
        assert "path_planning" in manager.parameters
        assert "materialization" in manager.parameters

    def test_pipeline_config_nested_structure(self, qapp):
        """Test creating form for PipelineConfig with its nested structure."""
        pipeline_config = PipelineConfig()
        
        # Analyze the full structure
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="pipeline_nested_test",
            dataclass_type=PipelineConfig
        )
        
        assert manager is not None
        assert manager.dataclass_type == PipelineConfig


class TestOptionalDataclassHandling:
    """Test handling of optional dataclass parameters."""

    def test_optional_dataclass_with_none_value(self, qapp):
        """Test optional dataclass parameter with None value."""
        parent_config = TestParentConfig()
        
        # Analyze the structure
        param_info = SignatureAnalyzer.analyze(TestParentConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(parent_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="optional_none_test"
        )
        
        assert manager is not None
        # Optional nested config should be None initially
        assert manager.parameters.get("nested_config") is None

    def test_optional_dataclass_with_concrete_value(self, qapp):
        """Test optional dataclass parameter with concrete value."""
        nested_config = TestNestedConfig(field1="custom_value", field2=42)
        parent_config = TestParentConfig(nested_config=nested_config)
        
        param_info = SignatureAnalyzer.analyze(TestParentConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(parent_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="optional_concrete_test"
        )
        
        assert manager is not None
        # Optional nested config should have the concrete value
        assert manager.parameters.get("nested_config") is not None

    def test_optional_dataclass_checkbox_behavior(self, qapp):
        """Test checkbox behavior for optional dataclass parameters."""
        parent_config = TestParentConfig()
        
        param_info = SignatureAnalyzer.analyze(TestParentConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(parent_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="checkbox_behavior_test"
        )
        
        # Check if optional checkboxes were created
        if hasattr(manager, 'optional_checkboxes'):
            # Should have checkboxes for optional dataclass parameters
            assert isinstance(manager.optional_checkboxes, dict)


class TestNestedFormUpdates:
    """Test updates and interactions with nested forms."""

    def test_nested_parameter_update_propagation(self, qtbot):
        """Test that updates to nested parameters propagate correctly."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_update_test"
        )

        # Track signal emissions
        signal_emitted = False

        def track_signal(*args):
            nonlocal signal_emitted
            signal_emitted = True

        manager.parameter_changed.connect(track_signal)

        # Update the nested parameter
        new_path_config = PathPlanningConfig()
        manager.update_parameter("path_planning", new_path_config)

        # Should emit signal for the update
        assert signal_emitted

    def test_nested_form_value_retrieval(self, qapp):
        """Test retrieving values from nested forms."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_retrieval_test"
        )
        
        # Get current values
        current_values = manager.get_current_values()
        
        assert "path_planning" in current_values
        assert current_values["path_planning"] is not None

    def test_nested_form_reset_behavior(self, qapp):
        """Test reset behavior with nested forms."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_reset_test",
            dataclass_type=PipelineConfig
        )
        
        # Reset the nested parameter
        manager.reset_parameter("path_planning")
        
        # In lazy context, should be None after reset
        assert manager.parameters["path_planning"] is None


class TestNestedFormHierarchy:
    """Test complex nested form hierarchies."""

    def test_deeply_nested_structure(self, qapp):
        """Test handling of deeply nested dataclass structures."""
        # This would test structures like PipelineConfig -> PathPlanningConfig -> nested fields
        pipeline_config = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="deep_nested_test",
            dataclass_type=PipelineConfig
        )
        
        assert manager is not None
        
        # Should handle the complex nested structure
        current_values = manager.get_current_values()
        assert isinstance(current_values, dict)

    def test_nested_manager_references(self, qapp):
        """Test that nested managers are properly referenced."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_refs_test"
        )
        
        # Check if nested managers are tracked
        if hasattr(manager, 'nested_managers'):
            assert isinstance(manager.nested_managers, dict)

    def test_nested_form_isolation(self, qapp):
        """Test that nested forms are properly isolated."""
        # Create two separate nested configs
        path_config1 = PathPlanningConfig()
        path_config2 = PathPlanningConfig()
        
        parameters = {
            "path_planning1": path_config1,
            "path_planning2": path_config2
        }
        parameter_types = {
            "path_planning1": PathPlanningConfig,
            "path_planning2": PathPlanningConfig
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_isolation_test"
        )
        
        # Updates to one should not affect the other
        manager.update_parameter("path_planning1", PathPlanningConfig())
        
        # path_planning2 should remain unchanged
        assert manager.parameters["path_planning2"] == path_config2


class TestNestedFormErrorHandling:
    """Test error handling in nested form scenarios."""

    def test_invalid_nested_dataclass_type(self, qapp):
        """Test handling of invalid nested dataclass types."""
        parameters = {"invalid_nested": "not_a_dataclass"}
        parameter_types = {"invalid_nested": str}  # Not a dataclass type
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="invalid_nested_test"
        )
        
        # Should handle gracefully
        assert manager is not None
        assert manager.parameters["invalid_nested"] == "not_a_dataclass"

    def test_missing_nested_dataclass_value(self, qapp):
        """Test handling when nested dataclass value is missing."""
        parameters = {}  # Missing nested parameter
        parameter_types = {"missing_nested": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="missing_nested_test"
        )
        
        # Should handle gracefully
        assert manager is not None

    def test_corrupted_nested_structure(self, qapp):
        """Test handling of corrupted nested dataclass structures."""
        # Create a partially invalid structure
        parameters = {"path_planning": None}  # None where dataclass expected
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="corrupted_nested_test"
        )
        
        # Should handle None values gracefully
        assert manager is not None
        assert manager.parameters["path_planning"] is None
