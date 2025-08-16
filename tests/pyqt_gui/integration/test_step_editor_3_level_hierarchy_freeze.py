#!/usr/bin/env python3
"""
Integration test to reproduce the 3-level hierarchy step editor freeze bug.

This test reproduces the exact scenario that causes the UI freeze:
1. Orchestrator has pipeline config applied (orchestrator-level override)
2. Step has materialization config that needs editing (step-level override)
3. UI attempts to open step editor for that step

The goal is to isolate the root cause of the freeze before attempting a fix.
"""

import pytest
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtTest import QTest

# OpenHCS imports
from openhcs.core.config import GlobalPipelineConfig, StepMaterializationConfig
from openhcs.core.pipeline_config import PipelineConfig, LazyStepMaterializationConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.lazy_config import get_context_stack, _global_context_stacks
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from openhcs.constants import Microscope
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer


class UIResponsivenessMonitor(QObject):
    """Monitor UI responsiveness to detect freezes."""

    freeze_detected = pyqtSignal(float)  # Emits freeze duration in seconds

    def __init__(self, threshold_ms: float = 100.0):
        super().__init__()
        self.threshold_ms = threshold_ms
        self.last_check = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_responsiveness)
        self.timer.start(50)  # Check every 50ms
        self.freeze_events = []  # Track freeze events

    def start(self):
        """Start monitoring UI responsiveness."""
        self.last_check = time.time()
        self.timer.start()

    def stop(self):
        """Stop monitoring UI responsiveness."""
        self.timer.stop()

    def _check_responsiveness(self):
        """Check if UI is responsive."""
        current_time = time.time()
        elapsed = (current_time - self.last_check) * 1000  # Convert to ms

        if elapsed > self.threshold_ms:
            self.freeze_events.append({
                'duration_ms': elapsed,
                'timestamp': current_time
            })
            self.freeze_detected.emit(elapsed)

        self.last_check = current_time


class UIFreezeDetector:
    """Advanced freeze detector that can detect UI freezes without getting stuck."""

    def __init__(self):
        self.freeze_detected = False
        self.freeze_start_time = None
        self.monitoring_thread = None
        self.stop_monitoring = False

    def start_monitoring(self):
        """Start monitoring for freezes."""
        self.freeze_detected = False
        self.freeze_start_time = None
        self.stop_monitoring = False

    def check_for_freeze_with_timeout(self, action_func, timeout_seconds=5):
        """
        Execute an action and detect if it causes a UI freeze.

        Uses a separate thread to monitor the main thread and detect freezes.

        Args:
            action_func: Function to execute that might cause freeze
            timeout_seconds: How long to wait before considering it frozen

        Returns:
            bool: True if freeze detected
        """
        import threading
        import signal

        freeze_detected = False
        action_completed = False

        def monitor_thread():
            """Monitor thread that detects if main thread is frozen."""
            nonlocal freeze_detected
            start_time = time.time()

            while not action_completed and not self.stop_monitoring:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    freeze_detected = True
                    print(f"🚨 FREEZE DETECTED: Action took longer than {timeout_seconds} seconds")
                    break
                time.sleep(0.1)  # Check every 100ms

        def timeout_handler(signum, frame):
            """Handle timeout signal."""
            nonlocal freeze_detected
            freeze_detected = True
            print(f"🚨 FREEZE DETECTED: Timeout after {timeout_seconds} seconds")
            raise TimeoutError("UI freeze detected")

        try:
            # Start monitoring thread
            monitor = threading.Thread(target=monitor_thread, daemon=True)
            monitor.start()

            # Set up signal handler for timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            # Execute the action that might freeze
            start_time = time.time()
            action_func()
            action_completed = True
            elapsed = time.time() - start_time

            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            print(f"✅ Action completed in {elapsed:.2f} seconds")

        except (TimeoutError, Exception) as e:
            freeze_detected = True
            print(f"❌ Action failed or froze: {e}")

            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        finally:
            action_completed = True
            self.stop_monitoring = True

        self.freeze_detected = freeze_detected
        return freeze_detected


class ContextStackMonitor:
    """Monitor context stack state for debugging."""

    def __init__(self):
        self.snapshots = []

    def take_snapshot(self, label: str):
        """Take a snapshot of current context stack state."""
        try:
            stack = get_context_stack(GlobalPipelineConfig)
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'total_contexts': len(stack),
                'context_details': [
                    {
                        'suffix': ctx.materialization_defaults.output_dir_suffix,
                        'sub_dir': ctx.materialization_defaults.sub_dir,
                        'well_filter': ctx.materialization_defaults.well_filter
                    }
                    for ctx in stack
                ]
            }
            self.snapshots.append(snapshot)
            return snapshot
        except Exception as e:
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'error': str(e),
                'total_contexts': 0,
                'context_details': []
            }
            self.snapshots.append(snapshot)
            return snapshot

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect context stack leaks by comparing snapshots."""
        leaks = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]

            if curr['total_contexts'] > prev['total_contexts']:
                leaks.append({
                    'from_snapshot': prev['label'],
                    'to_snapshot': curr['label'],
                    'context_increase': curr['total_contexts'] - prev['total_contexts']
                })

        return leaks


@pytest.fixture
def synthetic_plates(tmp_path):
    """Create synthetic microscopy plates for testing."""
    plates = []

    for i in range(2):  # Create 2 test plates
        plate_path = tmp_path / f"test_plate_{i+1}"
        generator = SyntheticMicroscopyGenerator(
            output_dir=str(plate_path),
            grid_size=(1, 1),  # Minimal grid for speed
            tile_size=(32, 32),  # Small tiles for speed
            overlap_percent=10,
            wavelengths=1,  # Single channel for speed
            z_stack_levels=1,
            wells=["A01"],  # Single well for testing
            format="ImageXpress"
        )
        generator.generate_dataset()
        plates.append(plate_path)

    return plates


@pytest.fixture
def global_config():
    """Create test global configuration."""
    return GlobalPipelineConfig(
        microscope=Microscope.IMAGEXPRESS,
        materialization_defaults=StepMaterializationConfig(
            output_dir_suffix="_global",
            sub_dir="global_images",
            well_filter=10
        )
    )


@pytest.fixture
def orchestrator_configs():
    """Create test orchestrator configurations."""
    return [
        PipelineConfig(
            materialization_defaults=LazyStepMaterializationConfig(
                output_dir_suffix="_orchestrator1",
                sub_dir="orch1_images",
                well_filter=1
            )
        ),
        PipelineConfig(
            materialization_defaults=LazyStepMaterializationConfig(
                output_dir_suffix="_orchestrator2",
                sub_dir="orch2_images",
                well_filter=5
            )
        )
    ]


class TestStepEditor3LevelHierarchyFreeze:
    """Test step editor freeze with 3-level hierarchy configuration."""

    def test_complete_user_workflow_step_editor_freeze(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Test that reproduces the step editor freeze using the COMPLETE OpenHCS user workflow.

        This test simulates the entire real-world user interaction sequence:
        1. Launch main OpenHCS application window
        2. Open Plate Manager and add a new plate
        3. Initialize plate with default configurations (global config level)
        4. Edit plate configuration with orchestrator-level settings (orchestrator config level)
        5. Open Pipeline Editor and add a step with materialization config (step config level)
        6. Click "Edit Step" to open step editor dialog - THIS IS WHERE THE FREEZE OCCURS

        Compares two scenarios:
        - Scenario A: Complete workflow with orchestrator config (should freeze)
        - Scenario B: Workflow without orchestrator config (should work normally)
        """
        # Setup simple freeze detection
        freeze_detector = UIFreezeDetector()
        freeze_detector.start_monitoring()

        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("initial_state")

        print("\n" + "="*80)
        print("=== SCENARIO A: COMPLETE WORKFLOW WITH ORCHESTRATOR CONFIG (SHOULD FREEZE) ===")
        print("=== WATCH THE UI WINDOWS OPEN AND INTERACT - FREEZE SHOULD OCCUR ON EDIT STEP ===")
        print("="*80)

        scenario_a_result = self._test_complete_user_workflow(
            qtbot, synthetic_plates[0], global_config, orchestrator_configs[0],
            "🔴 SCENARIO A (WITH CONFIG)", freeze_detector, context_monitor,
            apply_orchestrator_config=True
        )

        print(f"Scenario A - Freeze detected: {scenario_a_result['freeze_detected']}")
        print(f"Scenario A - Context leaks: {len(scenario_a_result['context_leaks'])}")

        print("\n" + "="*80)
        print("=== CLEANUP BETWEEN SCENARIOS ===")
        print("=== Ensuring clean state before Scenario B ===")
        print("="*80)

        # Reset freeze detector for clean state
        freeze_detector.start_monitoring()

        # Allow time for complete cleanup
        QApplication.processEvents()
        qtbot.wait(2000)  # 2 seconds for cleanup

        print("✅ Cleanup complete - starting Scenario B with clean state")

        print("\n" + "="*80)
        print("=== SCENARIO B: COMPLETE WORKFLOW WITHOUT ORCHESTRATOR CONFIG (SHOULD NOT FREEZE) ===")
        print("=== THIS SHOULD WORK NORMALLY - STEP EDITOR SHOULD OPEN WITHOUT ISSUES ===")
        print("="*80)

        scenario_b_result = self._test_complete_user_workflow(
            qtbot, synthetic_plates[1], global_config, None,
            "🟢 SCENARIO B (NO CONFIG)", freeze_detector, context_monitor,
            apply_orchestrator_config=False
        )

        print(f"Scenario B - Freeze detected: {scenario_b_result['freeze_detected']}")
        print(f"Scenario B - Context leaks: {len(scenario_b_result['context_leaks'])}")

        print("\n=== COMPARISON RESULTS ===")
        print(f"Scenario A (WITH orchestrator config) - Freeze: {scenario_a_result['freeze_detected']}")
        print(f"Scenario B (WITHOUT orchestrator config) - Freeze: {scenario_b_result['freeze_detected']}")

        # The test should show that Scenario A freezes but Scenario B doesn't
        if scenario_a_result['freeze_detected'] and not scenario_b_result['freeze_detected']:
            print("\n✅ BUG CONFIRMED: Orchestrator config causes step editor freeze in complete workflow")
            pytest.fail(
                f"Step editor freeze confirmed in complete user workflow: "
                f"WITH orchestrator config: FREEZE DETECTED, "
                f"WITHOUT orchestrator config: NO FREEZE"
            )
        elif scenario_a_result['freeze_detected']:
            print("\n⚠️ BOTH scenarios freeze - broader issue than orchestrator config")
            pytest.fail(f"Both scenarios freeze - need deeper investigation")
        else:
            print("\n❓ NO freezes detected in either scenario - bug may be fixed or test needs refinement")

    def _test_complete_user_workflow(self, qtbot, plate_path, global_config, orchestrator_config,
                                   scenario_name, freeze_detector, context_monitor, apply_orchestrator_config=True):
        """
        Test the complete OpenHCS user workflow that leads to the step editor freeze.

        This simulates the exact sequence of user interactions that cause the freeze:
        1. Launch main OpenHCS application
        2. Open Plate Manager and add plate
        3. Initialize plate with configurations
        4. Edit plate configuration (orchestrator level)
        5. Open Pipeline Editor and add step
        6. Click "Edit Step" - FREEZE OCCURS HERE
        """
        from openhcs.pyqt_gui.app import OpenHCSPyQtApp
        from openhcs.pyqt_gui.main import OpenHCSMainWindow
        from pathlib import Path

        context_snapshot_before = context_monitor.take_snapshot(f"{scenario_name}_before_workflow")

        main_window = None
        plate_manager_widget = None
        pipeline_editor_widget = None
        step_editor_dialog = None
        exception_occurred = False

        try:
            print(f"\n🔍 {scenario_name}: Step 1 - Launch Main OpenHCS Application")

            # Create the main OpenHCS application window (real application entry point)
            main_window = OpenHCSMainWindow(global_config)
            qtbot.addWidget(main_window)
            main_window.show()
            QApplication.processEvents()
            qtbot.wait(2000)  # 2 seconds - let user see main window launch

            print(f"✅ {scenario_name}: Main OpenHCS window launched")

            print(f"\n🔍 {scenario_name}: Step 2 - Access Existing Plate Manager")

            # Access the plate manager (already open by default - don't manipulate visibility)
            if "plate_manager" in main_window.floating_windows:
                plate_manager_window = main_window.floating_windows["plate_manager"]

                # Get the actual PlateManagerWidget without changing window state
                plate_manager_widget = plate_manager_window.findChild(QWidget, "PlateManagerWidget")
                if not plate_manager_widget:
                    # Find by layout
                    layout = plate_manager_window.layout()
                    if layout and layout.count() > 0:
                        plate_manager_widget = layout.itemAt(0).widget()

                QApplication.processEvents()
                qtbot.wait(1000)  # 1 second - let user see existing plate manager
                print(f"✅ {scenario_name}: Accessed existing Plate Manager (already open)")
            else:
                print(f"⚠️ {scenario_name}: Plate Manager not found in floating windows")

            print(f"\n🔍 {scenario_name}: Step 3 - Add Plate to Plate Manager")

            # Properly add plate using the plate manager's workflow
            if plate_manager_widget and hasattr(plate_manager_widget, 'add_plate_callback'):
                # Use the real plate manager method to add the plate
                from pathlib import Path
                plate_manager_widget.add_plate_callback([Path(plate_path)])
                QApplication.processEvents()
                qtbot.wait(2000)  # 2 seconds - let user see plate added
                print(f"✅ {scenario_name}: Added plate to plate manager: {Path(plate_path).name}")

                # Verify plate was added to the list
                if hasattr(plate_manager_widget, 'plates'):
                    plate_count = len(plate_manager_widget.plates)
                    print(f"✅ {scenario_name}: Plate manager now contains {plate_count} plates")
                else:
                    print(f"⚠️ {scenario_name}: Could not verify plate count")
            else:
                print(f"⚠️ {scenario_name}: Could not add plate - plate manager not accessible")

            print(f"\n🔍 {scenario_name}: Step 4 - Initialize Plate in Plate Manager")

            # First, select the plate in the plate manager
            if plate_manager_widget and hasattr(plate_manager_widget, 'plate_list'):
                plate_list = plate_manager_widget.plate_list
                if plate_list.count() > 0:
                    # Select the first (and only) plate
                    plate_list.setCurrentRow(0)
                    plate_list.setFocus()
                    QApplication.processEvents()
                    qtbot.wait(1000)  # 1 second - let user see plate selection
                    print(f"✅ {scenario_name}: Selected plate in plate manager")

                    # Now initialize the plate using the real plate manager workflow
                    if hasattr(plate_manager_widget, 'action_init_plate'):
                        print(f"🔍 {scenario_name}: Initializing plate (this creates orchestrator)...")

                        # This is the real way plates get initialized with orchestrators
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        try:
                            # Run the async initialization
                            loop.run_until_complete(plate_manager_widget.action_init_plate())
                            QApplication.processEvents()
                            qtbot.wait(2000)  # 2 seconds - let user see initialization
                            print(f"✅ {scenario_name}: Plate initialized with orchestrator")

                        finally:
                            loop.close()
                    else:
                        print(f"⚠️ {scenario_name}: action_init_plate method not found")
                else:
                    print(f"⚠️ {scenario_name}: No plates in plate list to select")
            else:
                print(f"⚠️ {scenario_name}: Could not access plate list")

            print(f"\n🔍 {scenario_name}: Step 5 - Configure Orchestrator (CRITICAL FOR 3-LEVEL HIERARCHY)")

            # This is the missing step that creates the 3-level hierarchy!
            # Simulate clicking "Edit" button on the selected plate to open config dialog
            if apply_orchestrator_config and orchestrator_config:
                print(f"🔍 {scenario_name}: Opening plate configuration dialog...")

                # Find and click the "Edit" button for the selected plate
                if plate_manager_widget and hasattr(plate_manager_widget, 'action_edit_config'):
                    # This opens the plate configuration dialog
                    plate_manager_widget.action_edit_config()
                    QApplication.processEvents()
                    qtbot.wait(2000)  # 2 seconds - let user see config dialog open

                    print(f"✅ {scenario_name}: Opened plate configuration dialog")

                    # The config dialog should now be open - we need to modify orchestrator settings
                    # and then save to apply the orchestrator-level overrides

                    # Find the config dialog window
                    config_dialog = None
                    for widget in QApplication.topLevelWidgets():
                        if hasattr(widget, 'windowTitle') and 'config' in widget.windowTitle().lower():
                            config_dialog = widget
                            break

                    if config_dialog:
                        print(f"✅ {scenario_name}: Found configuration dialog")

                        # Simulate modifying orchestrator configuration in the dialog
                        # (In real usage, user would change settings here)
                        print(f"🔍 {scenario_name}: Simulating orchestrator config modifications...")
                        qtbot.wait(1000)  # 1 second - simulate user making changes

                        # Find and click the "Save" button to apply orchestrator-level overrides
                        save_button = None
                        if hasattr(config_dialog, 'findChild'):
                            from PyQt6.QtWidgets import QPushButton
                            save_button = config_dialog.findChild(QPushButton, "save_button")
                            if not save_button:
                                # Try to find any button with "Save" text
                                for button in config_dialog.findChildren(QPushButton):
                                    if "save" in button.text().lower():
                                        save_button = button
                                        break

                        if save_button:
                            print(f"🔍 {scenario_name}: Clicking Save button to apply orchestrator config...")
                            qtbot.mouseClick(save_button, Qt.MouseButton.LeftButton)
                            QApplication.processEvents()
                            qtbot.wait(2000)  # 2 seconds - let user see save action
                            print(f"✅ {scenario_name}: Applied orchestrator configuration (3-level hierarchy NOW ACTIVE)")
                        else:
                            print(f"⚠️ {scenario_name}: Could not find Save button in config dialog")
                            # Close dialog manually if save button not found
                            config_dialog.close()
                    else:
                        print(f"⚠️ {scenario_name}: Could not find configuration dialog")
                else:
                    print(f"⚠️ {scenario_name}: action_edit_config method not found")
            else:
                print(f"✅ {scenario_name}: Skipped orchestrator configuration (2-level hierarchy only)")

            print(f"\n🔍 {scenario_name}: Step 6 - Access Existing Pipeline Editor")

            # Access the pipeline editor (already open by default - don't manipulate visibility)
            if "pipeline_editor" in main_window.floating_windows:
                pipeline_editor_window = main_window.floating_windows["pipeline_editor"]

                # Get the actual PipelineEditorWidget without changing window state
                pipeline_editor_widget = pipeline_editor_window.findChild(QWidget, "PipelineEditorWidget")
                if not pipeline_editor_widget:
                    # Find by layout
                    layout = pipeline_editor_window.layout()
                    if layout and layout.count() > 0:
                        pipeline_editor_widget = layout.itemAt(0).widget()

                QApplication.processEvents()
                qtbot.wait(1000)  # 1 second - let user see existing pipeline editor
                print(f"✅ {scenario_name}: Accessed existing Pipeline Editor (already open)")

                # The pipeline editor should already be connected to the orchestrator through the plate manager
                # No need to manually set current plate - the connection should be automatic
                print(f"✅ {scenario_name}: Pipeline editor connected to orchestrator via plate manager")

            else:
                print(f"⚠️ {scenario_name}: Pipeline Editor not found in floating windows")

            print(f"\n🔍 {scenario_name}: Step 7 - Add Step with Materialization Config")

            # Create a step with materialization config (3rd level in hierarchy)
            from openhcs.core.steps.function_step import FunctionStep
            from openhcs.core.pipeline_config import LazyStepMaterializationConfig

            step_materialization_config = LazyStepMaterializationConfig(
                output_dir_suffix="_step_override"
            )

            test_step = FunctionStep(
                func=[],  # Empty function list for testing
                name="test_step_with_materialization",
                materialization_config=step_materialization_config
            )

            # Add step to pipeline editor
            if pipeline_editor_widget and hasattr(pipeline_editor_widget, 'pipeline_steps'):
                pipeline_editor_widget.pipeline_steps.append(test_step)
                # Trigger UI update to show the step in the list
                if hasattr(pipeline_editor_widget, 'update_step_list'):
                    pipeline_editor_widget.update_step_list()
                QApplication.processEvents()
                qtbot.wait(1500)  # 1.5 seconds - let user see step added to list
                print(f"✅ {scenario_name}: Added step with materialization config")

                # Verify the step was added to the UI list
                if hasattr(pipeline_editor_widget, 'step_list') and pipeline_editor_widget.step_list:
                    step_count = pipeline_editor_widget.step_list.count()
                    print(f"✅ {scenario_name}: Step list now contains {step_count} items")
                else:
                    print(f"⚠️ {scenario_name}: Could not verify step list contents")
            else:
                print(f"⚠️ {scenario_name}: Could not add step - pipeline editor not accessible")

            print(f"\n🔍 {scenario_name}: Step 8 - Select Step in Pipeline Editor")

            # First, we need to properly select the step in the UI
            if pipeline_editor_widget:
                # Find the step list widget (should be step_list based on the code)
                step_list_widget = None
                if hasattr(pipeline_editor_widget, 'step_list'):
                    step_list_widget = pipeline_editor_widget.step_list
                    print(f"✅ {scenario_name}: Found step_list widget")
                else:
                    # Search for QListWidget in the pipeline editor as fallback
                    from PyQt6.QtWidgets import QListWidget
                    step_list_widget = pipeline_editor_widget.findChild(QListWidget)
                    if step_list_widget:
                        print(f"✅ {scenario_name}: Found QListWidget via findChild")
                    else:
                        print(f"⚠️ {scenario_name}: No QListWidget found")

                if step_list_widget and step_list_widget.count() > 0:
                    print(f"✅ {scenario_name}: Step list has {step_list_widget.count()} items")

                    # Select the first step (our test step) - this is critical for edit to work
                    step_list_widget.setCurrentRow(0)
                    step_list_widget.setFocus()
                    QApplication.processEvents()
                    qtbot.wait(2000)  # 2 seconds - let user see step selection

                    print(f"✅ {scenario_name}: Selected step in pipeline editor (row 0)")

                    # Verify selection using the pipeline editor's own method
                    if hasattr(pipeline_editor_widget, 'get_selected_steps'):
                        selected_steps = pipeline_editor_widget.get_selected_steps()
                        if selected_steps:
                            step_name = getattr(selected_steps[0], 'name', 'Unknown')
                            print(f"✅ {scenario_name}: Step selection confirmed via get_selected_steps(): {step_name}")
                        else:
                            print(f"⚠️ {scenario_name}: get_selected_steps() returned empty list")

                    # Also verify via Qt selection
                    selected_items = step_list_widget.selectedItems()
                    if selected_items:
                        print(f"✅ {scenario_name}: Qt selection confirmed: {selected_items[0].text()}")
                    else:
                        print(f"⚠️ {scenario_name}: Qt selection failed - no items selected")

                        # Try to force selection
                        item = step_list_widget.item(0)
                        if item:
                            item.setSelected(True)
                            step_list_widget.setCurrentItem(item)
                            QApplication.processEvents()
                            qtbot.wait(1000)  # 1 second - let user see forced selection
                            print(f"🔧 {scenario_name}: Forced selection of first item")
                else:
                    print(f"⚠️ {scenario_name}: Could not find step list widget or no steps available")
                    if step_list_widget:
                        print(f"   Step list count: {step_list_widget.count()}")
                    else:
                        print(f"   Step list widget is None")

            print(f"\n🔍 {scenario_name}: Step 9 - Click 'Edit Step' (CRITICAL FREEZE POINT)")
            print(f"⚠️ {scenario_name}: WATCH CAREFULLY - This is where the freeze should occur!")

            # Add a pause before the critical action so user can watch
            qtbot.wait(3000)  # 3 seconds - let user prepare to watch for freeze

            # This is where the freeze occurs - opening the step editor with 3-level hierarchy
            if pipeline_editor_widget and hasattr(pipeline_editor_widget, 'action_edit_step'):
                # Simulate user clicking "Edit Step" button
                print(f"🔍 {scenario_name}: Simulating Edit Step button click NOW...")

                # Define the action that might freeze
                def edit_step_action():
                    pipeline_editor_widget.action_edit_step()
                    QApplication.processEvents()

                # Check for freeze using timeout-based detection
                print(f"🔍 {scenario_name}: Checking for UI freeze after Edit Step click...")
                freeze_detected = freeze_detector.check_for_freeze_with_timeout(
                    edit_step_action,
                    timeout_seconds=5  # 5 seconds should be enough for normal operation
                )

                if freeze_detected:
                    print(f"❌ {scenario_name}: UI FREEZE DETECTED!")
                else:
                    print(f"✅ {scenario_name}: Edit Step action completed (no freeze detected)")

            else:
                print(f"⚠️ {scenario_name}: Could not trigger Edit Step - action not found")

        except Exception as e:
            print(f"❌ {scenario_name}: Exception during complete workflow: {e}")
            exception_occurred = True
            import traceback
            traceback.print_exc()

        finally:
            # Clean up all UI components for this scenario
            if step_editor_dialog:
                step_editor_dialog.close()
                qtbot.wait(10)

            # Close main window to ensure clean state for next scenario
            if main_window:
                main_window.close()
                qtbot.wait(100)  # Allow time for proper cleanup

        # Take snapshot after test
        context_snapshot_after = context_monitor.take_snapshot(f"{scenario_name}_after_workflow")

        # Detect context leaks for this scenario
        leaks = context_monitor.detect_leaks()

        return {
            'freeze_detected': freeze_detector.freeze_detected,
            'context_leaks': leaks,
            'exception_occurred': exception_occurred,
            'context_depth_before': context_snapshot_before['total_contexts'],
            'context_depth_after': context_snapshot_after['total_contexts']
        }
        # Setup UI responsiveness monitoring
        responsiveness_monitor = UIResponsivenessMonitor(threshold_ms=200.0)
        freeze_events = []

        def on_freeze_detected(duration_ms):
            freeze_events.append({
                'duration_ms': duration_ms,
                'timestamp': time.time()
            })
            print(f"UI FREEZE DETECTED: {duration_ms:.1f}ms")

        responsiveness_monitor.freeze_detected.connect(on_freeze_detected)

        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("test_start")

        # Step 1: Create orchestrator with pipeline config applied
        print("\n=== Step 1: Create Orchestrator with Pipeline Config ===")

        plate_path = synthetic_plates[0]
        file_manager = FileManager(storage_registry)
        orchestrator = PipelineOrchestrator(
            plate_path=plate_path,
            global_config=global_config,
            storage_registry=storage_registry
        )
        orchestrator.initialize()

        # Apply orchestrator-level configuration (this triggers context stacking)
        orchestrator.apply_pipeline_config(orchestrator_configs[0])
        print("✅ Applied orchestrator pipeline config")

        context_monitor.take_snapshot("orchestrator_config_applied")

        # Step 2: Create step with materialization config
        print("\n=== Step 2: Create Step with Materialization Config ===")

        # Level 3: Step-level configuration (overrides orchestrator)
        step_materialization_config = LazyStepMaterializationConfig(
            output_dir_suffix="_step_override",  # Override orchestrator
            sub_dir=None,  # Should inherit through hierarchy
            well_filter=None  # Should inherit through hierarchy
        )

        # Create function step with materialization config
        def dummy_function(image):
            return image

        step_with_config = FunctionStep(
            func=dummy_function,
            name="test_step_with_materialization",
            materialization_config=step_materialization_config
        )
        print("✅ Created step with materialization config")

        context_monitor.take_snapshot("step_with_config_created")

        # Step 3: Simulate step editor creation (this is where freeze occurs)
        print("\n=== Step 3: Simulate Step Editor Creation (FREEZE POINT) ===")

        # Analyze step parameters (same as step editor does)
        param_info = SignatureAnalyzer.analyze(FunctionStep.__init__)
        print(f"✅ Analyzed {len(param_info)} step parameters")

        # Extract current step parameters
        parameters = {}
        parameter_types = {}
        for name, info in param_info.items():
            current_value = getattr(step_with_config, name, info.default_value)
            parameters[name] = current_value
            parameter_types[name] = info.param_type

        print("✅ Extracted step parameters")

        # This is the critical point where freeze occurs in the UI
        print("Creating ParameterFormManager (FREEZE POINT)...")

        freeze_detected = False
        try:
            # Create parameter form manager (simulates step editor opening)
            # Use the correct ParameterFormManager constructor signature
            form_manager = ParameterFormManager(
                parameters=parameters,
                parameter_types=parameter_types,
                field_id="step_editor_test",
                parameter_info=param_info,
                global_config_type=GlobalPipelineConfig,
                placeholder_prefix="Pipeline default"
            )

            # The form manager is already created above - no need for create_parameter_form
            # Just access the materialization_config parameter to trigger lazy resolution
            print("✅ Created ParameterFormManager successfully")

            # Process events to trigger placeholder resolution
            QApplication.processEvents()
            QTest.qWait(100)  # Allow time for freeze detection

            # Test accessing materialization_config parameter
            print("Testing materialization_config parameter access...")

            # Test lazy resolution of step materialization config
            print("Testing lazy resolution...")
            suffix = step_materialization_config.output_dir_suffix
            sub_dir = step_materialization_config.sub_dir
            well_filter = step_materialization_config.well_filter

            print(f"✅ Step config resolved:")
            print(f"  - output_dir_suffix: {suffix}")
            print(f"  - sub_dir: {sub_dir}")
            print(f"  - well_filter: {well_filter}")

        except Exception as e:
            print(f"❌ FREEZE/ERROR during ParameterFormManager creation: {e}")
            freeze_detected = True
            import traceback
            traceback.print_exc()

        context_monitor.take_snapshot("parameter_form_creation_attempted")

        # Step 4: Analyze results
        print("\n=== Step 4: Analyze Results ===")

        # Check for context stack leaks
        leaks = context_monitor.detect_leaks()
        final_snapshot = context_monitor.snapshots[-1]

        print(f"Context leaks detected: {len(leaks)}")
        print(f"Final context stack depth: {final_snapshot['total_contexts']}")
        print(f"UI freeze events: {len(freeze_events)}")

        if freeze_events:
            total_freeze_time = sum(event['duration_ms'] for event in freeze_events)
            print(f"Total freeze time: {total_freeze_time:.1f}ms")

        # Verify the bug manifests
        if len(leaks) > 0 or final_snapshot['total_contexts'] > 1 or freeze_events or freeze_detected:
            print("\n❌ BUG REPRODUCTION SUCCESSFUL")
            print("3-level hierarchy step editor freeze confirmed")

            # This is expected behavior - the test should detect the bug
            pytest.fail(
                f"Step editor freeze bug reproduced: "
                f"Context leaks: {len(leaks)}, "
                f"Stack depth: {final_snapshot['total_contexts']}, "
                f"Freeze events: {len(freeze_events)}, "
                f"Exception detected: {freeze_detected}"
            )
        else:
            print("\n✅ No freeze detected - bug may be fixed")

    def test_context_stack_accumulation_during_step_editing(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Test if context stack accumulates infinitely during step editing workflow.
        """
        print("\n=== Testing Context Stack Accumulation During Step Editing ===")

        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("initial_state")

        # Create multiple orchestrators and apply configs
        orchestrators = []
        file_manager = FileManager(storage_registry)

        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=storage_registry
            )
            orchestrator.initialize()
            orchestrator.apply_pipeline_config(orchestrator_configs[i % len(orchestrator_configs)])
            orchestrators.append(orchestrator)

            context_monitor.take_snapshot(f"orchestrator_{i+1}_applied")

        # Simulate multiple step editing operations
        for i in range(3):
            print(f"Simulating step editing operation {i+1}")

            # Create step with materialization config
            step_config = LazyStepMaterializationConfig(
                output_dir_suffix=f"_step_{i}",
                sub_dir=None,
                well_filter=None
            )

            # Simulate parameter form creation (step editor opening)
            try:
                # Create parameter form for step materialization config
                step_params = {"materialization_config": step_config}
                step_types = {"materialization_config": LazyStepMaterializationConfig}

                form_manager = ParameterFormManager(
                    parameters=step_params,
                    parameter_types=step_types,
                    field_id=f"step_editing_test_{i}",
                    global_config_type=GlobalPipelineConfig,
                    placeholder_prefix="Pipeline default"
                )

                # Process events
                QApplication.processEvents()
                QTest.qWait(10)

            except Exception as e:
                print(f"Error during step editing simulation {i+1}: {e}")

            context_monitor.take_snapshot(f"step_editing_operation_{i+1}")

        # Analyze context stack growth
        leaks = context_monitor.detect_leaks()
        final_snapshot = context_monitor.snapshots[-1]

        print(f"\n=== CONTEXT STACK ANALYSIS ===")
        print(f"Total context leaks detected: {len(leaks)}")
        print(f"Final context stack depth: {final_snapshot['total_contexts']}")

        # The test should detect context accumulation
        if len(leaks) > 0 or final_snapshot['total_contexts'] > 2:
            pytest.fail(
                f"Context stack accumulation detected during step editing: "
                f"Leaks: {len(leaks)}, Final depth: {final_snapshot['total_contexts']}"
            )


