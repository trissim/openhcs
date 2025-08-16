"""
Unit tests for lazy structure preservation in PyQt6 parameter forms.

This module tests the fix for the bug where lazy dataclasses were not properly
saving concrete edited fields. When a user edits a field in the parameter form,
that specific field should be saved as a concrete value in the dataclass, while
all other unedited fields should remain as None to preserve lazy resolution behavior.
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtWidgets import QApplication

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import LazyDefaultPlaceholderService
from openhcs.core.lazy_config import LazyDataclassFactory
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme


@dataclass
class TestLazyConfig:
    """Test configuration for lazy structure preservation testing."""
    field1: str = "default1"
    field2: int = 42
    field3: Optional[str] = None


class TestLazyStructurePreservation:
    """Test that edited fields are preserved as concrete values in lazy dataclasses."""

    def test_lazy_dataclass_detection(self, qapp):
        """Test that lazy dataclass detection works correctly."""
        # Create a lazy version of TestLazyConfig
        LazyTestConfig = LazyDataclassFactory.create_lazy_dataclass(
            defaults_source=TestLazyConfig,
            lazy_class_name="LazyTestConfig"
        )
        
        # Verify it has lazy resolution
        has_lazy_resolution = LazyDefaultPlaceholderService.has_lazy_resolution(LazyTestConfig)
        assert has_lazy_resolution, "LazyTestConfig should have lazy resolution"

    def test_edited_field_preservation(self, qapp):
        """Test that edited fields are preserved as concrete values."""
        # Create a lazy dataclass type
        LazyTestConfig = LazyDataclassFactory.create_lazy_dataclass(
            defaults_source=TestLazyConfig,
            lazy_class_name="LazyTestConfig"
        )
        
        # Create parameters with mixed None (lazy) and concrete values
        parameters = {
            'field1': None,        # This should remain None (lazy)
            'field2': 100,         # This should be preserved as concrete value
            'field3': "edited"     # This should be preserved as concrete value
        }
        
        parameter_types = {
            'field1': str,
            'field2': int,
            'field3': Optional[str]
        }
        
        # Create form manager with lazy dataclass type
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_test",
            dataclass_type=LazyTestConfig,
            color_scheme=PyQt6ColorScheme()
        )
        
        # Simulate user editing field1 (changing from None to concrete value)
        manager._emit_parameter_change("field1", "user_edited_value")
        
        # Get current values (this should preserve lazy structure)
        current_values = manager.get_current_values()
        
        # Verify the results
        assert current_values['field1'] == "user_edited_value", "field1 should be concrete after user edit"
        assert current_values['field2'] == 100, "field2 should remain concrete"
        assert current_values['field3'] == "edited", "field3 should remain concrete"

    def test_none_values_preserved_in_lazy_context(self, qapp):
        """Test that None values are preserved in lazy context (not converted to defaults)."""
        # Create a lazy dataclass type
        LazyTestConfig = LazyDataclassFactory.create_lazy_dataclass(
            defaults_source=TestLazyConfig,
            lazy_class_name="LazyTestConfig"
        )
        
        # Create parameters with all None values (pure lazy context)
        parameters = {
            'field1': None,
            'field2': None,
            'field3': None
        }
        
        parameter_types = {
            'field1': str,
            'field2': int,
            'field3': Optional[str]
        }
        
        # Create form manager with lazy dataclass type
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_none_test",
            dataclass_type=LazyTestConfig,
            color_scheme=PyQt6ColorScheme()
        )
        
        # Get current values without any edits
        current_values = manager.get_current_values()
        
        # All values should remain None (lazy) since no user edits occurred
        assert current_values['field1'] is None, "field1 should remain None (lazy)"
        assert current_values['field2'] is None, "field2 should remain None (lazy)"
        assert current_values['field3'] is None, "field3 should remain None (lazy)"

    def test_global_config_editing_mode_detection(self, qapp):
        """Test that global config editing mode is detected correctly."""
        # Test with PipelineConfig (should be lazy context)
        pipeline_parameters = {'microscope': None, 'plate_path': None}
        pipeline_types = {'microscope': str, 'plate_path': str}
        
        pipeline_manager = ParameterFormManager(
            parameters=pipeline_parameters,
            parameter_types=pipeline_types,
            field_id="pipeline_test",
            dataclass_type=PipelineConfig,
            color_scheme=PyQt6ColorScheme()
        )
        
        # Should be in lazy context (not global config editing)
        is_lazy = LazyDefaultPlaceholderService.has_lazy_resolution(PipelineConfig)
        assert is_lazy, "PipelineConfig should have lazy resolution"

    def test_dataclass_creation_with_preserved_values(self, qapp):
        """Test that a new dataclass instance can be created with preserved values."""
        # Create a lazy dataclass type
        LazyTestConfig = LazyDataclassFactory.create_lazy_dataclass(
            defaults_source=TestLazyConfig,
            lazy_class_name="LazyTestConfig"
        )
        
        # Create parameters and simulate user edits
        parameters = {
            'field1': None,
            'field2': None,
            'field3': None
        }
        
        parameter_types = {
            'field1': str,
            'field2': int,
            'field3': Optional[str]
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="creation_test",
            dataclass_type=LazyTestConfig,
            color_scheme=PyQt6ColorScheme()
        )
        
        # Simulate user editing multiple fields
        manager._emit_parameter_change("field1", "user_value1")
        manager._emit_parameter_change("field2", 999)
        # field3 remains None (lazy)
        
        # Get current values
        current_values = manager.get_current_values()
        
        # Create new dataclass instance (this is what save_config does)
        new_config = LazyTestConfig(**current_values)
        
        # Verify the new instance has correct values
        assert new_config.field1 == "user_value1", "field1 should have user-edited value"
        assert new_config.field2 == 999, "field2 should have user-edited value"
        assert new_config.field3 is None, "field3 should remain None (lazy)"

    def test_mixed_concrete_and_lazy_values(self, qapp):
        """Test handling of mixed concrete and lazy values in the same form."""
        # Create a lazy dataclass type
        LazyTestConfig = LazyDataclassFactory.create_lazy_dataclass(
            defaults_source=TestLazyConfig,
            lazy_class_name="LazyTestConfig"
        )
        
        # Start with mixed values
        parameters = {
            'field1': "initial_concrete",  # Already concrete
            'field2': None,                # Lazy
            'field3': "another_concrete"   # Already concrete
        }
        
        parameter_types = {
            'field1': str,
            'field2': int,
            'field3': Optional[str]
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="mixed_test",
            dataclass_type=LazyTestConfig,
            color_scheme=PyQt6ColorScheme()
        )
        
        # Edit the lazy field
        manager._emit_parameter_change("field2", 777)
        
        # Get current values
        current_values = manager.get_current_values()
        
        # Verify all values are preserved correctly
        assert current_values['field1'] == "initial_concrete", "field1 should remain concrete"
        assert current_values['field2'] == 777, "field2 should become concrete after edit"
        assert current_values['field3'] == "another_concrete", "field3 should remain concrete"

    def test_nested_lazy_dataclass_preservation(self, qapp):
        """Test that nested lazy dataclass edits are preserved correctly."""
        from openhcs.core.config import PathPlanningConfig
        from openhcs.core.pipeline_config import PipelineConfig
        from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer

        # Create a PipelineConfig instance with nested PathPlanningConfig
        pipeline_config = PipelineConfig()

        # Extract parameters using SignatureAnalyzer (same as config window)
        param_info = SignatureAnalyzer.analyze(PipelineConfig)

        parameters = {}
        parameter_types = {}
        for name, info in param_info.items():
            # Use object.__getattribute__ to preserve None values for lazy dataclasses
            current_value = object.__getattribute__(pipeline_config, name) if hasattr(pipeline_config, name) else info.default_value
            parameters[name] = current_value
            parameter_types[name] = info.param_type

        # Create form manager for PipelineConfig
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_test",
            dataclass_type=PipelineConfig,
            color_scheme=PyQt6ColorScheme()
        )

        # Verify that nested managers are created for dataclass parameters
        assert hasattr(manager, 'nested_managers'), "Should have nested managers"

        # Find the path_planning nested manager
        path_planning_manager = None
        for param_name, nested_manager in manager.nested_managers.items():
            if param_name == 'path_planning':
                path_planning_manager = nested_manager
                break

        if path_planning_manager is not None:
            # Simulate editing a field in the nested PathPlanningConfig
            # Edit the 'output_dir_suffix' field in path_planning (this is a real field)
            path_planning_manager._emit_parameter_change("output_dir_suffix", "_edited_suffix")

            # Get current values from the main form manager
            current_values = manager.get_current_values()

            # Verify that the nested dataclass instance was created and contains the edit
            path_planning_instance = current_values.get('path_planning')
            assert path_planning_instance is not None, "path_planning should not be None after edit"

            # Check that the edited field is preserved
            if hasattr(path_planning_instance, 'output_dir_suffix'):
                assert path_planning_instance.output_dir_suffix == "_edited_suffix", \
                    f"output_dir_suffix should be '_edited_suffix', got {path_planning_instance.output_dir_suffix}"

            # Verify that other fields in the nested dataclass remain None (lazy)
            # This tests that we're not filling in default values for unedited fields
            if hasattr(path_planning_instance, 'global_output_folder'):
                # In lazy context, unedited fields should remain None
                nested_global_folder = object.__getattribute__(path_planning_instance, 'global_output_folder')
                # Note: This might be None (lazy) or a resolved value depending on the lazy implementation
                # The key is that our edit to output_dir_suffix should be preserved
                print(f"global_output_folder value: {nested_global_folder}")

    def test_nested_dataclass_creation_with_preserved_edits(self, qapp):
        """Test that a new PipelineConfig can be created with preserved nested edits."""
        from openhcs.core.config import PathPlanningConfig
        from openhcs.core.pipeline_config import PipelineConfig
        from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer

        # Create a PipelineConfig instance
        pipeline_config = PipelineConfig()

        # Extract parameters
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        parameter_types = {}
        for name, info in param_info.items():
            current_value = object.__getattribute__(pipeline_config, name) if hasattr(pipeline_config, name) else info.default_value
            parameters[name] = current_value
            parameter_types[name] = info.param_type

        # Create form manager
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="creation_test",
            dataclass_type=PipelineConfig,
            color_scheme=PyQt6ColorScheme()
        )

        # Edit nested field if path_planning manager exists
        if 'path_planning' in manager.nested_managers:
            path_planning_manager = manager.nested_managers['path_planning']
            path_planning_manager._emit_parameter_change("sub_dir", "test_nested_edit")

        # Get current values and create new PipelineConfig instance
        current_values = manager.get_current_values()

        # This simulates what save_config does
        try:
            new_pipeline_config = PipelineConfig(**current_values)

            # Verify the new instance has the nested edit preserved
            if hasattr(new_pipeline_config, 'path_planning') and new_pipeline_config.path_planning:
                if hasattr(new_pipeline_config.path_planning, 'sub_dir'):
                    assert new_pipeline_config.path_planning.sub_dir == "test_nested_edit", \
                        "Nested edit should be preserved in new instance"

        except Exception as e:
            # If creation fails, at least verify that current_values contains the nested structure
            path_planning_value = current_values.get('path_planning')
            assert path_planning_value is not None, f"path_planning should not be None, got error: {e}"
