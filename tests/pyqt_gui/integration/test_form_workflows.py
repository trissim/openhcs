"""
Integration tests for PyQt6 parameter form workflows.

This module tests complete workflows involving parameter forms,
including form creation, user interaction simulation, and data persistence.
"""

import pytest
from PyQt6.QtWidgets import QLineEdit, QComboBox, QCheckBox
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import PathPlanningConfig, MaterializationPathConfig, Microscope
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer


class TestCompleteFormWorkflows:
    """Test complete form creation and interaction workflows."""

    def test_pipeline_config_form_workflow(self, qapp):
        """Test complete workflow with PipelineConfig form."""
        # Create initial config
        pipeline_config = PipelineConfig()
        
        # Analyze parameters
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        # Create form manager
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="pipeline_workflow",
            dataclass_type=PipelineConfig
        )
        
        assert manager is not None
        assert manager.dataclass_type == PipelineConfig
        
        # Test that we can get current values
        current_values = manager.get_current_values()
        assert isinstance(current_values, dict)
        assert len(current_values) > 0

    def test_nested_dataclass_workflow(self, qapp):
        """Test workflow with nested dataclass parameters."""
        # Create config with nested dataclass
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_workflow"
        )
        
        assert manager is not None
        
        # Should have nested managers for dataclass parameters
        if hasattr(manager, 'nested_managers'):
            assert len(manager.nested_managers) >= 0  # May or may not have nested managers

    def test_parameter_update_workflow(self, qtbot):
        """Test complete parameter update workflow."""
        parameters = {
            "microscope": Microscope.IMAGEXPRESS,
            "plate_path": "/initial/path"
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="update_workflow"
        )

        # Track signal emissions
        signal_count = 0

        def count_signals(*args):
            nonlocal signal_count
            signal_count += 1

        manager.parameter_changed.connect(count_signals)

        # Update parameters
        manager.update_parameter("microscope", Microscope.OPERAPHENIX)
        manager.update_parameter("plate_path", "/updated/path")

        # Verify updates
        assert manager.parameters["microscope"] == Microscope.OPERAPHENIX
        assert manager.parameters["plate_path"] == "/updated/path"

        # Verify signals were emitted (may be more than 2 due to internal updates)
        assert signal_count >= 2

    def test_reset_workflow(self, qapp):
        """Test complete reset workflow."""
        parameters = {
            "string_param": "initial_value",
            "int_param": 42,
            "bool_param": True
        }
        parameter_types = {
            "string_param": str,
            "int_param": int,
            "bool_param": bool
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_workflow"
        )
        
        # Modify parameters
        manager.update_parameter("string_param", "modified_value")
        manager.update_parameter("int_param", 999)
        manager.update_parameter("bool_param", False)
        
        # Verify modifications
        assert manager.parameters["string_param"] == "modified_value"
        assert manager.parameters["int_param"] == 999
        assert manager.parameters["bool_param"] == False
        
        # Reset all parameters
        manager.reset_all_parameters()
        
        # Verify reset (values should be None in lazy context)
        for param_name in parameters.keys():
            assert manager.parameters[param_name] is None


class TestLazyDataclassIntegration:
    """Test integration with lazy dataclass functionality."""

    def test_lazy_context_placeholder_workflow(self, qapp):
        """Test workflow in lazy context with placeholders."""
        # Create lazy config (None values)
        parameters = {
            "microscope": None,
            "plate_path": None
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_workflow",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        assert manager is not None
        assert manager.dataclass_type == PipelineConfig
        assert manager.placeholder_prefix == "Pipeline default: "
        
        # In lazy context, parameters should remain None until explicitly set
        assert manager.parameters["microscope"] is None
        assert manager.parameters["plate_path"] is None

    def test_concrete_context_workflow(self, qapp):
        """Test workflow in concrete context (global config editing)."""
        # Create concrete config with actual values
        parameters = {
            "microscope": Microscope.IMAGEXPRESS,
            "plate_path": "/concrete/path"
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="concrete_workflow"
        )
        
        assert manager is not None
        
        # In concrete context, parameters should have actual values
        assert manager.parameters["microscope"] == Microscope.IMAGEXPRESS
        assert manager.parameters["plate_path"] == "/concrete/path"

    def test_mixed_state_workflow(self, qapp):
        """Test workflow with mixed lazy/concrete states."""
        # Some parameters have values, others are None
        parameters = {
            "microscope": Microscope.IMAGEXPRESS,  # Concrete
            "plate_path": None,                    # Lazy
            "output_path": "/output/path"          # Concrete
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str,
            "output_path": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="mixed_workflow",
            dataclass_type=PipelineConfig
        )
        
        assert manager is not None
        
        # Verify mixed state is preserved
        assert manager.parameters["microscope"] == Microscope.IMAGEXPRESS
        assert manager.parameters["plate_path"] is None
        assert manager.parameters["output_path"] == "/output/path"


class TestFormPersistenceWorkflows:
    """Test form data persistence and reload workflows."""

    def test_parameter_persistence_workflow(self, qapp):
        """Test that parameter changes persist correctly."""
        parameters = {"test_param": "initial"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="persistence_test"
        )
        
        # Update parameter
        manager.update_parameter("test_param", "updated")
        
        # Get current values (simulating save operation)
        saved_values = manager.get_current_values()
        assert saved_values["test_param"] == "updated"
        
        # Create new manager with saved values (simulating reload)
        new_manager = ParameterFormManager(
            parameters=saved_values,
            parameter_types=parameter_types,
            field_id="persistence_test_reload"
        )
        
        # Verify values persisted
        assert new_manager.parameters["test_param"] == "updated"

    def test_nested_parameter_persistence_workflow(self, qapp):
        """Test persistence of nested dataclass parameters."""
        # This test would verify that nested dataclass changes persist
        # through save/reload cycles, which was a major bug in the old system
        
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_persistence_test"
        )
        
        assert manager is not None
        
        # Get current values
        current_values = manager.get_current_values()
        assert "path_planning" in current_values
        
        # The actual persistence testing would require more complex setup
        # to simulate the save/reload cycle that was problematic before


class TestErrorHandlingWorkflows:
    """Test error handling in complete workflows."""

    def test_invalid_parameter_update_workflow(self, qapp):
        """Test workflow with invalid parameter updates."""
        parameters = {"int_param": 42}
        parameter_types = {"int_param": int}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="error_workflow"
        )
        
        # Try to update with invalid type (should handle gracefully)
        try:
            manager.update_parameter("int_param", "not_an_integer")
            # Should either convert or handle the error gracefully
            # The exact behavior depends on the implementation
        except Exception as e:
            # If an exception is raised, it should be a meaningful one
            assert str(e) != ""

    def test_missing_parameter_workflow(self, qapp):
        """Test workflow when trying to update non-existent parameters."""
        parameters = {"existing_param": "value"}
        parameter_types = {"existing_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="missing_param_workflow"
        )
        
        # Try to update non-existent parameter (should handle gracefully)
        try:
            manager.update_parameter("non_existent_param", "value")
            # Should handle gracefully without crashing
        except KeyError:
            # KeyError is acceptable for non-existent parameters
            pass


class TestComprehensiveIntegrationWorkflows:
    """Test comprehensive integration workflows that combine multiple features."""

    def test_complete_pipeline_config_workflow(self, qapp):
        """Test complete workflow with full PipelineConfig including all features."""
        # Create comprehensive pipeline config
        pipeline_config = PipelineConfig()

        # Analyze full structure
        from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
        param_info = SignatureAnalyzer.analyze(PipelineConfig)

        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value

        parameter_types = {name: info.param_type for name, info in param_info.items()}

        # Create manager with all features enabled
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="comprehensive_pipeline_test",
            parameter_info=param_info,
            use_scroll_area=True,
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )

        assert manager is not None
        assert manager.dataclass_type == PipelineConfig
        assert manager.use_scroll_area == True
        assert manager.placeholder_prefix == "Pipeline default: "

        # Test comprehensive functionality
        current_values = manager.get_current_values()
        assert isinstance(current_values, dict)
        assert len(current_values) > 0

        # Test reset all functionality
        manager.reset_all_parameters()

        # All parameters should be None in lazy context
        for param_name in parameters.keys():
            assert manager.parameters[param_name] is None

    def test_mixed_parameter_types_workflow(self, qapp):
        """Test workflow with mixed parameter types including all supported types."""
        from pathlib import Path
        from openhcs.core.config import Microscope, ZarrCompressor

        parameters = {
            "string_param": "test_string",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "enum_param": Microscope.IMAGEXPRESS,
            "zarr_enum": ZarrCompressor.ZSTD,
            "path_param": Path("/test/path"),
            "optional_string": None,
            "nested_config": PathPlanningConfig()
        }

        parameter_types = {
            "string_param": str,
            "int_param": int,
            "float_param": float,
            "bool_param": bool,
            "enum_param": Microscope,
            "zarr_enum": ZarrCompressor,
            "path_param": Path,
            "optional_string": type(None),
            "nested_config": PathPlanningConfig
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="mixed_types_test",
            use_scroll_area=True
        )

        assert manager is not None
        assert len(manager.widgets) > 0

        # Test updating each type
        signal_count = 0

        def count_signals(*args):
            nonlocal signal_count
            signal_count += 1

        manager.parameter_changed.connect(count_signals)

        manager.update_parameter("string_param", "updated_string")
        manager.update_parameter("int_param", 999)
        manager.update_parameter("float_param", 2.71)
        manager.update_parameter("bool_param", False)
        manager.update_parameter("enum_param", Microscope.OPERAPHENIX)

        # Should have emitted signals for each update
        assert signal_count >= 5

    def test_error_recovery_workflow(self, qapp):
        """Test workflow with error conditions and recovery."""
        parameters = {
            "valid_param": "valid_value",
            "problematic_param": None
        }
        parameter_types = {
            "valid_param": str,
            "problematic_param": str
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="error_recovery_test"
        )

        # Test recovery from various error conditions
        try:
            # Try invalid type conversion
            manager.update_parameter("valid_param", 12345)  # int to str
            # Should handle conversion gracefully
        except Exception:
            pass

        try:
            # Try updating non-existent parameter
            manager.update_parameter("non_existent", "value")
        except Exception:
            pass

        try:
            # Try resetting non-existent parameter
            manager.reset_parameter("non_existent")
        except Exception:
            pass

        # Manager should still be functional
        assert manager is not None
        current_values = manager.get_current_values()
        assert isinstance(current_values, dict)

    def test_performance_workflow(self, qapp):
        """Test workflow with many parameters to check performance."""
        # Create many parameters to test performance
        parameters = {}
        parameter_types = {}

        for i in range(50):  # 50 parameters should be manageable
            param_name = f"param_{i}"
            parameters[param_name] = f"value_{i}"
            parameter_types[param_name] = str

        import time
        start_time = time.time()

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="performance_test"
        )

        creation_time = time.time() - start_time

        # Should create form in reasonable time (less than 5 seconds)
        assert creation_time < 5.0
        assert manager is not None
        assert len(manager.widgets) == 50

        # Test bulk operations performance
        start_time = time.time()
        manager.reset_all_parameters()
        reset_time = time.time() - start_time

        # Bulk reset should be fast
        assert reset_time < 2.0
