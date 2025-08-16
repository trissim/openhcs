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
from openhcs.core.config import (
    PathPlanningConfig, MaterializationPathConfig, VFSConfig, ZarrConfig,
    StepMaterializationConfig, ZarrCompressor, MaterializationBackend, WellFilterMode
)
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
from openhcs.constants import Backend, Microscope
from pathlib import Path
from PyQt6.QtWidgets import QComboBox


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

    def test_concrete_nested_values_preservation_in_lazy_context(self, qapp):
        """
        Test that concrete values in nested lazy dataclass fields are preserved during form creation.

        This test verifies the fix for the bug where concrete values entered by users in nested
        dataclass fields were being lost during PyQt6 form creation in lazy parent contexts.

        The bug was in the extract_nested_parameters method which was forcing all nested field
        values to None in lazy contexts, regardless of whether users had entered concrete values.
        """
        # Create custom nested configs with concrete values that should be preserved
        custom_path_planning = PathPlanningConfig(
            output_dir_suffix="_custom_outputs",
            global_output_folder=Path("/custom/output/path"),
            sub_dir="custom_images"
        )

        custom_vfs = VFSConfig(
            intermediate_backend=Backend.DISK,
            materialization_backend=MaterializationBackend.ZARR
        )

        custom_zarr = ZarrConfig(
            store_name="custom_store",
            compressor=ZarrCompressor.ZSTD,
            compression_level=5,
            shuffle=False
        )

        custom_materialization = StepMaterializationConfig(
            output_dir_suffix="_step_outputs",
            global_output_folder=Path("/step/output/path"),
            sub_dir="step_checkpoints",
            well_filter=["A01", "B02", "C03"],
            well_filter_mode=WellFilterMode.EXCLUDE
        )

        # Create a PipelineConfig instance with these custom nested values
        pipeline_config = PipelineConfig(
            path_planning=custom_path_planning,
            vfs=custom_vfs,
            zarr=custom_zarr,
            materialization_defaults=custom_materialization,
            microscope=Microscope.IMAGEXPRESS
        )

        # Analyze the structure for form creation
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value

        parameter_types = {name: info.param_type for name, info in param_info.items()}

        # Create form manager in lazy context (this is where the bug occurred)
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="concrete_value_preservation_test",
            dataclass_type=PipelineConfig  # This makes it a lazy context
        )

        # Get current values from the form - this should preserve concrete values
        current_values = manager.get_current_values()

        # Verify that all concrete nested values are preserved

        # Test path_planning nested values
        path_planning_value = current_values.get('path_planning')
        assert path_planning_value is not None, "path_planning should not be None - concrete values were lost!"
        assert path_planning_value.output_dir_suffix == "_custom_outputs"
        assert path_planning_value.global_output_folder == Path("/custom/output/path")
        assert path_planning_value.sub_dir == "custom_images"

        # Test vfs nested values
        vfs_value = current_values.get('vfs')
        assert vfs_value is not None, "vfs should not be None - concrete values were lost!"
        assert vfs_value.intermediate_backend == Backend.DISK
        assert vfs_value.materialization_backend == MaterializationBackend.ZARR

        # Test zarr nested values
        zarr_value = current_values.get('zarr')
        assert zarr_value is not None, "zarr should not be None - concrete values were lost!"
        assert zarr_value.store_name == "custom_store"
        assert zarr_value.compressor == ZarrCompressor.ZSTD
        assert zarr_value.compression_level == 5
        assert zarr_value.shuffle == False

        # Test materialization_defaults nested values
        mat_value = current_values.get('materialization_defaults')
        assert mat_value is not None, "materialization_defaults should not be None - concrete values were lost!"
        assert mat_value.output_dir_suffix == "_step_outputs"
        assert mat_value.global_output_folder == Path("/step/output/path")
        assert mat_value.sub_dir == "step_checkpoints"
        assert mat_value.well_filter == ["A01", "B02", "C03"]
        assert mat_value.well_filter_mode == WellFilterMode.EXCLUDE

        # Test simple enum value
        microscope_value = current_values.get('microscope')
        assert microscope_value == Microscope.IMAGEXPRESS

    def test_enum_combobox_behavior_after_nested_value_fix(self, qapp):
        """
        Test that enum combobox widgets work correctly after the nested value preservation fix.

        This test ensures that the fix to extract_nested_parameters() method did not break
        enum field handling in combobox widgets, including proper selection display,
        reset behavior, and value preservation.
        """
        # Create PipelineConfig with specific enum values
        custom_vfs = VFSConfig(
            intermediate_backend=Backend.DISK,
            materialization_backend=MaterializationBackend.ZARR
        )

        custom_zarr = ZarrConfig(
            compressor=ZarrCompressor.ZSTD,
            compression_level=5
        )

        custom_materialization = StepMaterializationConfig(
            well_filter_mode=WellFilterMode.EXCLUDE
        )

        pipeline_config = PipelineConfig(
            vfs=custom_vfs,
            zarr=custom_zarr,
            materialization_defaults=custom_materialization,
            microscope=Microscope.IMAGEXPRESS
        )

        # Create form manager in lazy context
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value

        parameter_types = {name: info.param_type for name, info in param_info.items()}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="enum_combobox_test",
            dataclass_type=PipelineConfig
        )

        # Test top-level enum combobox
        microscope_widget = manager.widgets.get('microscope')
        assert microscope_widget is not None, "microscope widget should exist"
        assert isinstance(microscope_widget, QComboBox), "microscope widget should be QComboBox"

        # Check that correct enum value is selected
        current_index = microscope_widget.currentIndex()
        assert current_index >= 0, "microscope combobox should have a selection"
        selected_value = microscope_widget.itemData(current_index)
        assert selected_value == Microscope.IMAGEXPRESS, f"Expected IMAGEXPRESS, got {selected_value}"

        # Test nested enum comboboxes
        assert hasattr(manager, 'nested_managers'), "Should have nested managers"

        # Test VFS nested enum
        vfs_manager = manager.nested_managers.get('vfs')
        assert vfs_manager is not None, "VFS nested manager should exist"

        intermediate_backend_widget = vfs_manager.widgets.get('intermediate_backend')
        assert intermediate_backend_widget is not None, "VFS intermediate_backend widget should exist"
        assert isinstance(intermediate_backend_widget, QComboBox), "VFS intermediate_backend should be QComboBox"

        current_index = intermediate_backend_widget.currentIndex()
        assert current_index >= 0, "VFS intermediate_backend should have a selection"
        selected_value = intermediate_backend_widget.itemData(current_index)
        assert selected_value == Backend.DISK, f"Expected DISK, got {selected_value}"

        # Test Zarr nested enum
        zarr_manager = manager.nested_managers.get('zarr')
        assert zarr_manager is not None, "Zarr nested manager should exist"

        compressor_widget = zarr_manager.widgets.get('compressor')
        assert compressor_widget is not None, "Zarr compressor widget should exist"
        assert isinstance(compressor_widget, QComboBox), "Zarr compressor should be QComboBox"

        current_index = compressor_widget.currentIndex()
        assert current_index >= 0, "Zarr compressor should have a selection"
        selected_value = compressor_widget.itemData(current_index)
        assert selected_value == ZarrCompressor.ZSTD, f"Expected ZSTD, got {selected_value}"

        # Test enum reset behavior
        original_microscope = manager.parameters.get('microscope')
        assert original_microscope == Microscope.IMAGEXPRESS

        manager.reset_parameter('microscope')
        reset_microscope = manager.parameters.get('microscope')
        assert reset_microscope is None, "Reset should set enum to None in lazy context"

        # Test value retrieval preserves nested enum values
        current_values = manager.get_current_values()

        # Nested enum values should be preserved
        assert hasattr(current_values.get('vfs'), 'intermediate_backend')
        vfs_backend = current_values['vfs'].intermediate_backend
        assert vfs_backend == Backend.DISK, f"VFS backend should be preserved, got {vfs_backend}"

        assert hasattr(current_values.get('zarr'), 'compressor')
        zarr_compressor = current_values['zarr'].compressor
        assert zarr_compressor == ZarrCompressor.ZSTD, f"Zarr compressor should be preserved, got {zarr_compressor}"

    def test_standalone_config_enum_reset_behavior(self, qapp):
        """
        Test that standalone config dataclass forms (like VFSConfig) show proper enum reset behavior.

        This test ensures that config dataclasses used as standalone forms (not nested within
        PipelineConfig) still behave lazily for enum reset operations, showing placeholder
        behavior instead of concrete default values.
        """
        # Create VFSConfig with specific enum value
        vfs_config = VFSConfig(intermediate_backend=Backend.DISK)

        # Create form manager for standalone VFSConfig
        param_info = SignatureAnalyzer.analyze(VFSConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(vfs_config, name, None)
            parameters[name] = value

        parameter_types = {name: info.param_type for name, info in param_info.items()}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="standalone_config_test",
            dataclass_type=VFSConfig,
            is_global_config_editing=False  # Explicitly set to behave lazily
        )

        # Verify initial enum combobox state
        backend_widget = manager.widgets.get('intermediate_backend')
        assert backend_widget is not None, "intermediate_backend widget should exist"
        assert isinstance(backend_widget, QComboBox), "intermediate_backend should be QComboBox"

        # Check initial selection shows the set value
        current_index = backend_widget.currentIndex()
        assert current_index >= 0, "Should have initial selection"
        selected_value = backend_widget.itemData(current_index)
        assert selected_value == Backend.DISK, f"Expected DISK, got {selected_value}"

        # Test enum reset behavior - this should reset to None and show placeholder state
        manager.reset_parameter('intermediate_backend')

        # Verify parameter was reset to None (lazy behavior)
        reset_param_value = manager.parameters.get('intermediate_backend')
        assert reset_param_value is None, f"Parameter should be None after reset, got {reset_param_value}"

        # Verify widget shows placeholder state (should show default value with placeholder styling)
        reset_widget_index = backend_widget.currentIndex()
        assert reset_widget_index >= 0, f"Widget should show placeholder value, got index {reset_widget_index}"

        # Verify the widget is showing the default value (Backend.MEMORY)
        reset_widget_value = backend_widget.itemData(reset_widget_index)
        assert reset_widget_value == Backend.MEMORY, f"Widget should show default value Backend.MEMORY, got {reset_widget_value}"

        # Verify the widget has placeholder styling (indicates it's in placeholder state)
        assert backend_widget.property("is_placeholder_state"), "Widget should have placeholder state property set"

        # Verify widget is still responsive after reset
        for i in range(backend_widget.count()):
            if backend_widget.itemData(i) == Backend.MEMORY:
                backend_widget.setCurrentIndex(i)
                break

        # Check that manual selection works
        assert backend_widget.currentIndex() >= 0, "Widget should be responsive after reset"
        selected_value = backend_widget.itemData(backend_widget.currentIndex())
        assert selected_value == Backend.MEMORY, f"Manual selection should work, got {selected_value}"


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
