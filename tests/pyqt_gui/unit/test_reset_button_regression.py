"""
Automated regression test for reset button functionality issues.

This test captures the exact reset button problems demonstrated in the real PyQt6 
application demo without requiring human interaction. It validates the critical 
issues found in the actual application behavior.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox
from PyQt6.QtCore import Qt

from openhcs.core.config import GlobalPipelineConfig, Microscope
from openhcs.core.pipeline_config import PipelineConfig
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager


class TestResetButtonRegression:
    """
    Regression test that captures real-world reset button functionality failures.
    
    This test replicates the exact scenario from the real PyQt6 application demo
    and validates the critical issues found in actual usage.
    """

    def test_reset_button_lazy_context_regression(self, qtbot):
        """
        REGRESSION TEST: Reset buttons not working correctly in lazy context.
        
        This test replicates the exact scenario from the real application demo
        where reset buttons failed to properly reset parameters to None.
        """
        # === SETUP: Replicate Real Application Scenario ===
        
        # Use the same parameter configuration as the real demo
        parameters = {
            "num_workers": 8,  # Modified from default 16
            "microscope": Microscope.OPERAPHENIX,  # Modified from default AUTO
            "plate_path": "/custom/test/path",  # Custom path
            "use_threading": False,  # Modified from default True
            "output_dir_suffix": "custom_suffix"  # Custom suffix
        }
        
        parameter_types = {
            "num_workers": int,
            "microscope": Microscope,
            "plate_path": str,
            "use_threading": bool,
            "output_dir_suffix": str
        }
        
        # Create form manager with lazy context (same as demo)
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="regression_test_form",
            dataclass_type=PipelineConfig  # Lazy context - reset should go to None
        )
        
        qtbot.addWidget(manager)
        manager.show()
        qtbot.wait(10)
        
        # === VERIFY INITIAL STATE ===
        
        # Verify initial parameter values match demo setup
        assert manager.parameters["num_workers"] == 8
        assert manager.parameters["microscope"] == Microscope.OPERAPHENIX
        assert manager.parameters["plate_path"] == "/custom/test/path"
        assert manager.parameters["use_threading"] == False
        assert manager.parameters["output_dir_suffix"] == "custom_suffix"
        
        # === SIMULATE USER INTERACTIONS FROM DEMO ===
        
        # Track parameter change signals (like in the demo)
        signal_log = []
        
        def track_signals(param_name, value):
            signal_log.append((param_name, value))
        
        manager.parameter_changed.connect(track_signals)
        
        # Simulate user modifying values (like in demo where user changed num_workers to 0)
        manager.update_parameter("num_workers", 0)
        manager.update_parameter("plate_path", "")  # User cleared the path

        # Verify the modifications took effect
        assert manager.parameters["num_workers"] == 0
        # In lazy context, empty strings are converted to None for lazy resolution
        assert manager.parameters["plate_path"] is None
        
        # === TEST INDIVIDUAL RESET BUTTON FUNCTIONALITY ===
        
        # Find reset buttons (same as demo)
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        
        assert len(reset_buttons) > 0, "Should have reset buttons"
        
        # Test individual reset button click (simulate user clicking reset)
        if reset_buttons:
            # Click first reset button (should reset num_workers)
            reset_button = reset_buttons[0]
            qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
            
            # CRITICAL ISSUE: In lazy context, reset should set to None
            # But the real application showed it staying at current value
            current_value = manager.parameters["num_workers"]
            
            # Document the actual behavior vs expected behavior
            print(f"RESET REGRESSION: num_workers after reset = {current_value}")
            print(f"EXPECTED: None (lazy context)")
            print(f"ACTUAL: {current_value}")
            
            # This assertion will FAIL until the bug is fixed
            # In lazy context, reset should always set to None
            if current_value is not None:
                pytest.fail(
                    f"RESET BUTTON BUG: Parameter 'num_workers' was not reset to None in lazy context. "
                    f"Expected: None, Actual: {current_value}. "
                    f"This reproduces the bug found in the real application demo."
                )
        
        # === TEST RESET ALL FUNCTIONALITY ===
        
        # Clear signal log for reset all test
        signal_log.clear()
        
        # Record values before reset all (like in demo)
        values_before_reset = dict(manager.parameters)
        
        # Execute reset all (same as demo)
        manager.reset_all_parameters()
        
        # Record values after reset all
        values_after_reset = dict(manager.parameters)
        
        # === VALIDATE CRITICAL ISSUES FOUND IN DEMO ===
        
        print("\n=== RESET ALL REGRESSION TEST RESULTS ===")
        print("Values before reset:")
        for param, value in values_before_reset.items():
            print(f"  {param}: {value}")
        
        print("Values after reset:")
        for param, value in values_after_reset.items():
            print(f"  {param}: {value}")
        
        # Check each parameter for the specific issues found in demo
        issues_found = []
        
        for param_name in parameters.keys():
            before_value = values_before_reset[param_name]
            after_value = values_after_reset[param_name]
            
            # In lazy context, ALL parameters should reset to None
            if after_value is not None:
                issues_found.append(
                    f"Parameter '{param_name}': Expected None, got {after_value} "
                    f"(type: {type(after_value).__name__})"
                )
        
        # === VALIDATE WIDGET VS PARAMETER CONSISTENCY ===
        
        print("\nWidget vs Parameter consistency check:")
        widget_inconsistencies = []
        
        for param_name, widget in manager.widgets.items():
            param_value = manager.parameters[param_name]
            try:
                widget_value = manager.get_widget_value(widget)
                
                print(f"  {param_name}: param={param_value}, widget={widget_value}")
                
                # Check for inconsistencies (found in demo)
                if param_value != widget_value:
                    widget_inconsistencies.append(
                        f"Parameter '{param_name}': param={param_value}, widget={widget_value}"
                    )
            except Exception as e:
                print(f"  {param_name}: ERROR getting widget value - {e}")
        
        # === ASSERT THE CRITICAL ISSUES ===
        
        # This test documents the actual bugs found in the real application
        if issues_found:
            failure_message = (
                "RESET BUTTON REGRESSION: Reset All failed to set parameters to None in lazy context.\n"
                "This reproduces the exact issues found in the real PyQt6 application demo:\n\n"
                + "\n".join(f"  - {issue}" for issue in issues_found)
            )
            
            if widget_inconsistencies:
                failure_message += (
                    "\n\nAdditional widget inconsistencies found:\n"
                    + "\n".join(f"  - {inconsistency}" for inconsistency in widget_inconsistencies)
                )
            
            failure_message += (
                "\n\nExpected behavior: In lazy context (dataclass_type=PipelineConfig), "
                "ALL parameters should reset to None to allow lazy resolution of defaults."
            )
            
            pytest.fail(failure_message)
        
        # If we get here, the reset functionality is working correctly
        print("✅ Reset functionality working correctly - all parameters reset to None")

    def test_string_parameter_reset_empty_vs_none_regression(self, qtbot):
        """
        REGRESSION TEST: String parameters being set to empty string instead of None.
        
        The demo showed string parameters like 'plate_path' and 'output_dir_suffix' 
        being set to empty strings ("") instead of None after reset.
        """
        parameters = {"test_string": "initial_value"}
        parameter_types = {"test_string": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="string_reset_test",
            dataclass_type=PipelineConfig  # Lazy context
        )
        
        qtbot.addWidget(manager)
        manager.show()
        qtbot.wait(10)
        
        # Reset the string parameter
        manager.reset_parameter("test_string")
        
        # In lazy context, should be None, not empty string
        actual_value = manager.parameters["test_string"]
        
        if actual_value == "":
            pytest.fail(
                f"STRING RESET BUG: String parameter reset to empty string ('') instead of None. "
                f"This reproduces the bug found in the demo where plate_path and output_dir_suffix "
                f"were set to empty strings instead of None in lazy context."
            )
        elif actual_value is not None:
            pytest.fail(
                f"STRING RESET BUG: String parameter reset to {actual_value} instead of None. "
                f"Expected: None (lazy context), Actual: {actual_value}"
            )

    def test_widget_value_update_after_reset_regression(self, qtbot):
        """
        REGRESSION TEST: Widget values not updating correctly after reset.
        
        The demo showed widget values becoming inconsistent with parameter values
        after reset operations.
        """
        parameters = {"test_int": 42}
        parameter_types = {"test_int": int}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="widget_update_test",
            dataclass_type=PipelineConfig
        )
        
        qtbot.addWidget(manager)
        manager.show()
        qtbot.wait(10)
        
        # Get the widget
        widget = manager.widgets.get("test_int")
        assert widget is not None, "Should have widget for test_int"
        
        # Reset the parameter
        manager.reset_parameter("test_int")
        
        # Check parameter value
        param_value = manager.parameters["test_int"]
        
        # Check widget value
        widget_value = manager.get_widget_value(widget)
        
        # They should match
        if param_value != widget_value:
            pytest.fail(
                f"WIDGET UPDATE BUG: Widget value inconsistent with parameter value after reset. "
                f"Parameter: {param_value}, Widget: {widget_value}. "
                f"This reproduces the inconsistency found in the demo where num_workers "
                f"parameter was 0 but widget showed 2."
            )
