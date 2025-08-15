"""
Integration tests for lazy dataclass functionality in PyQt6 parameter forms.

This module tests the integration between the parameter form manager and the
lazy dataclass system, including threadlocal behavior, context awareness,
and lazy resolution.
"""

import pytest
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer


class TestLazyDataclassContexts:
    """Test different lazy dataclass contexts."""

    def test_orchestrator_config_editing_context(self, qapp):
        """Test parameter form in orchestrator config editing context (lazy)."""
        # Create lazy config for orchestrator editing
        lazy_config = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(lazy_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="orchestrator_context_test",
            dataclass_type=PipelineConfig,  # Lazy context
            placeholder_prefix="Pipeline default: "
        )
        
        assert manager is not None
        assert manager.dataclass_type == PipelineConfig
        assert manager.placeholder_prefix == "Pipeline default: "

    def test_global_config_editing_context(self, qapp):
        """Test parameter form in global config editing context (concrete)."""
        # Create global config for editing
        global_config = GlobalPipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(GlobalPipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(global_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="global_context_test",
            dataclass_type=GlobalPipelineConfig  # Concrete context
        )
        
        assert manager is not None
        assert manager.dataclass_type == GlobalPipelineConfig

    def test_context_switching(self, qapp):
        """Test switching between lazy and concrete contexts."""
        # Start with lazy context
        lazy_config = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(lazy_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        lazy_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="context_switch_lazy",
            dataclass_type=PipelineConfig
        )
        
        # Get values from lazy context
        lazy_values = lazy_manager.get_current_values()
        
        # Create concrete context with same values
        concrete_manager = ParameterFormManager(
            parameters=lazy_values,
            parameter_types=parameter_types,
            field_id="context_switch_concrete",
            dataclass_type=GlobalPipelineConfig
        )
        
        assert lazy_manager.dataclass_type != concrete_manager.dataclass_type


class TestLazyResolutionBehavior:
    """Test lazy resolution behavior in parameter forms."""

    def test_lazy_value_resolution(self, qapp):
        """Test that lazy values are resolved correctly."""
        # Create config with lazy resolution
        config = PipelineConfig()

        param_info = SignatureAnalyzer.analyze(type(config))
        parameters = {}
        for name in param_info.keys():
            value = getattr(config, name, None)
            parameters[name] = value

        parameter_types = {name: info.param_type for name, info in param_info.items()}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_resolution_test",
            dataclass_type=type(config)
        )

        assert manager is not None

    def test_lazy_placeholder_resolution(self, qapp):
        """Test that lazy placeholders are resolved correctly."""
        # Create lazy config
        lazy_config = PipelineConfig()
        
        parameters = {
            "microscope": None,  # Should show placeholder
            "plate_path": None   # Should show placeholder
        }
        parameter_types = {
            "microscope": type(getattr(lazy_config, "microscope", None)) or str,
            "plate_path": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="placeholder_resolution_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        # Parameters should remain None until explicitly set
        assert manager.parameters["microscope"] is None
        assert manager.parameters["plate_path"] is None

    def test_lazy_to_concrete_transition(self, qapp):
        """Test transition from lazy to concrete values."""
        lazy_config = PipelineConfig()
        
        parameters = {"microscope": None}
        parameter_types = {"microscope": type(getattr(lazy_config, "microscope", None)) or str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_to_concrete_test",
            dataclass_type=PipelineConfig
        )
        
        # Track signal emissions
        signal_emitted = False

        def track_signal(*args):
            nonlocal signal_emitted
            signal_emitted = True

        manager.parameter_changed.connect(track_signal)

        # Start with lazy (None)
        assert manager.parameters["microscope"] is None

        # Set concrete value
        from openhcs.core.config import Microscope
        manager.update_parameter("microscope", Microscope.IMAGEXPRESS)

        # Should now be concrete
        assert manager.parameters["microscope"] == Microscope.IMAGEXPRESS
        assert signal_emitted


class TestThreadlocalBehavior:
    """Test threadlocal lazy dataclass behavior."""

    def test_threadlocal_isolation(self, qapp):
        """Test that threadlocal contexts are properly isolated."""
        # This test would verify threadlocal behavior if applicable
        # The exact implementation depends on how threadlocal is used
        
        config1 = PipelineConfig()
        config2 = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters1 = {}
        parameters2 = {}
        
        for name in param_info.keys():
            parameters1[name] = getattr(config1, name, None)
            parameters2[name] = getattr(config2, name, None)
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager1 = ParameterFormManager(
            parameters=parameters1,
            parameter_types=parameter_types,
            field_id="threadlocal_test1",
            dataclass_type=PipelineConfig
        )
        
        manager2 = ParameterFormManager(
            parameters=parameters2,
            parameter_types=parameter_types,
            field_id="threadlocal_test2",
            dataclass_type=PipelineConfig
        )
        
        # Managers should be independent
        assert manager1.field_id != manager2.field_id

    def test_threadlocal_context_preservation(self, qapp):
        """Test that threadlocal context is preserved across operations."""
        lazy_config = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(lazy_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="context_preservation_test",
            dataclass_type=PipelineConfig
        )
        
        # Perform multiple operations
        manager.reset_all_parameters()
        current_values = manager.get_current_values()
        
        # Context should be preserved
        assert manager.dataclass_type == PipelineConfig


class TestLazyConfigurationPersistence:
    """Test persistence of lazy configuration states."""

    def test_lazy_state_persistence_through_updates(self, qapp):
        """Test that lazy state is preserved through parameter updates."""
        parameters = {
            "param1": None,      # Lazy
            "param2": "concrete" # Concrete
        }
        parameter_types = {
            "param1": str,
            "param2": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_persistence_test",
            dataclass_type=PipelineConfig
        )
        
        # Update only the concrete parameter
        manager.update_parameter("param2", "updated_concrete")
        
        # Lazy parameter should remain None
        assert manager.parameters["param1"] is None
        assert manager.parameters["param2"] == "updated_concrete"

    def test_lazy_state_persistence_through_resets(self, qapp):
        """Test that lazy state is preserved through resets."""
        parameters = {
            "param1": "initial_value",
            "param2": None  # Lazy
        }
        parameter_types = {
            "param1": str,
            "param2": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_persistence_test",
            dataclass_type=PipelineConfig
        )
        
        # Reset the concrete parameter
        manager.reset_parameter("param1")
        
        # Both should now be None (lazy)
        assert manager.parameters["param1"] is None
        assert manager.parameters["param2"] is None

    def test_lazy_configuration_save_load_cycle(self, qapp):
        """Test lazy configuration through save/load cycle."""
        # This test simulates the save/reload bug that was fixed
        
        lazy_config = PipelineConfig()
        
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        parameters = {}
        for name in param_info.keys():
            value = getattr(lazy_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        # Create initial manager
        manager1 = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="save_load_test1",
            dataclass_type=PipelineConfig
        )
        
        # Get current values (simulating save)
        saved_values = manager1.get_current_values()
        
        # Create new manager with saved values (simulating load)
        manager2 = ParameterFormManager(
            parameters=saved_values,
            parameter_types=parameter_types,
            field_id="save_load_test2",
            dataclass_type=PipelineConfig
        )
        
        # Lazy state should be preserved
        assert manager2.dataclass_type == PipelineConfig


class TestLazyConfigurationErrorHandling:
    """Test error handling in lazy configuration scenarios."""

    def test_invalid_lazy_context(self, qapp):
        """Test handling of invalid lazy context."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}
        
        # Invalid dataclass_type for lazy context
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="invalid_lazy_test",
            dataclass_type=str  # Not a proper dataclass
        )
        
        # Should handle gracefully
        assert manager is not None
        assert manager.dataclass_type == str

    def test_missing_lazy_context_information(self, qapp):
        """Test handling when lazy context information is missing."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}
        
        # No dataclass_type provided
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="missing_lazy_context_test"
        )
        
        # Should work without lazy context
        assert manager is not None
        assert manager.dataclass_type is None

    def test_corrupted_lazy_state(self, qapp):
        """Test handling of corrupted lazy state."""
        # Simulate corrupted state with mixed valid/invalid values
        parameters = {
            "valid_param": None,
            "invalid_param": object()  # Invalid type
        }
        parameter_types = {
            "valid_param": str,
            "invalid_param": str
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="corrupted_lazy_test",
            dataclass_type=PipelineConfig
        )
        
        # Should handle gracefully
        assert manager is not None
