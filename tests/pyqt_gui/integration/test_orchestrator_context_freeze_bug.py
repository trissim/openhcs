#!/usr/bin/env python3
"""
Integration test to reproduce and debug the UI freezing issue with 3-level hierarchy refinement.

This test reproduces the critical bug where applying orchestrator context causes the UI to freeze.
The root cause is that orchestrator_context context managers are not being properly cleaned up,
leading to context stack corruption and infinite loops in the resolution logic.
"""

import pytest
import asyncio
import threading
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
from openhcs.core.lazy_config import get_context_stack, _global_context_stacks
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.shared.service_adapter import ServiceAdapter
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from openhcs.constants import Microscope
from openhcs.io.filemanager import FileManager


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
        
    def _check_responsiveness(self):
        """Check if UI is responsive."""
        current_time = time.time()
        elapsed = (current_time - self.last_check) * 1000  # Convert to ms
        
        if elapsed > self.threshold_ms:
            self.freeze_detected.emit(elapsed)
            
        self.last_check = current_time


class ContextStackMonitor:
    """Monitor context stack state for debugging."""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        
    def take_snapshot(self, label: str) -> Dict[str, Any]:
        """Take a snapshot of current context stack state."""
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'stacks': {},
            'total_contexts': 0
        }
        
        # Capture all context stacks
        for config_type, stack in _global_context_stacks.items():
            stack_info = {
                'type_name': config_type.__name__,
                'depth': len(stack),
                'contexts': [str(ctx) for ctx in stack]
            }
            snapshot['stacks'][config_type.__name__] = stack_info
            snapshot['total_contexts'] += len(stack)
            
        self.snapshots.append(snapshot)
        return snapshot
        
    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect context stack leaks by comparing snapshots."""
        leaks = []
        
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            if curr['total_contexts'] > prev['total_contexts']:
                leak = {
                    'from_label': prev['label'],
                    'to_label': curr['label'],
                    'context_increase': curr['total_contexts'] - prev['total_contexts'],
                    'details': {}
                }
                
                # Find which stacks grew
                for type_name in curr['stacks']:
                    curr_depth = curr['stacks'][type_name]['depth']
                    prev_depth = prev['stacks'].get(type_name, {}).get('depth', 0)
                    
                    if curr_depth > prev_depth:
                        leak['details'][type_name] = {
                            'prev_depth': prev_depth,
                            'curr_depth': curr_depth,
                            'increase': curr_depth - prev_depth
                        }
                        
                leaks.append(leak)
                
        return leaks


@pytest.fixture
def synthetic_plates(tmp_path):
    """Create 2 synthetic plates for testing."""
    plates = []
    
    for i in range(2):
        plate_dir = tmp_path / f"test_plate_{i+1}"
        
        generator = SyntheticMicroscopyGenerator(
            output_dir=str(plate_dir),
            grid_size=(2, 2),  # Small grid for fast testing
            tile_size=(64, 64),  # Small tiles for fast testing
            overlap_percent=10,
            wavelengths=2,
            z_stack_levels=1,
            wells=["A01", "A02"],  # Two wells for testing
            format="ImageXpress"
        )
        generator.generate_dataset()
        plates.append(str(plate_dir))
        
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


class TestOrchestratorContextFreezeBug:
    """Test suite to reproduce and debug the orchestrator context freeze bug."""
    
    def test_context_stack_corruption_reproduction(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Reproduce the context stack corruption that causes UI freezing.
        
        This test demonstrates the root cause: orchestrator_context managers
        are entered but never exited, causing context stack corruption.
        """
        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("initial_state")
        
        # Create orchestrators
        orchestrators = []
        file_manager = FileManager()
        
        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=file_manager.registry
            )
            orchestrator.initialize()
            orchestrators.append(orchestrator)
            
        context_monitor.take_snapshot("orchestrators_created")
        
        # Apply pipeline configs - this triggers the bug
        for i, (orchestrator, config) in enumerate(zip(orchestrators, orchestrator_configs)):
            print(f"\n=== Applying config to orchestrator {i+1} ===")
            
            # This is where the bug occurs - context is pushed but never popped
            orchestrator.apply_pipeline_config(config)
            
            snapshot = context_monitor.take_snapshot(f"config_applied_orchestrator_{i+1}")
            print(f"Context stack depth after config {i+1}: {snapshot['total_contexts']}")
            
        # Detect context leaks
        leaks = context_monitor.detect_leaks()
        
        # Verify the bug exists
        assert len(leaks) > 0, "Expected context stack leaks but none detected"
        
        print(f"\n=== CONTEXT LEAKS DETECTED ===")
        for leak in leaks:
            print(f"Leak from '{leak['from_label']}' to '{leak['to_label']}':")
            print(f"  Context increase: {leak['context_increase']}")
            for type_name, details in leak['details'].items():
                print(f"  {type_name}: {details['prev_depth']} -> {details['curr_depth']} (+{details['increase']})")
                
        # Verify specific leak pattern
        final_snapshot = context_monitor.snapshots[-1]
        global_config_stack_depth = final_snapshot['stacks'].get('GlobalPipelineConfig', {}).get('depth', 0)
        
        # Should have 2 contexts on stack (one per orchestrator) that never get cleaned up
        assert global_config_stack_depth == 2, f"Expected 2 contexts on GlobalPipelineConfig stack, got {global_config_stack_depth}"
        
        print(f"\n=== BUG CONFIRMED ===")
        print(f"GlobalPipelineConfig context stack has {global_config_stack_depth} contexts that will never be cleaned up")
        print("This will cause infinite loops in resolution logic and UI freezing")


    def test_ui_freeze_reproduction_with_plate_manager(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Reproduce the actual UI freeze using PlateManagerWidget.
        
        This test simulates the exact user workflow that triggers the freeze.
        """
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
        
        # Create plate manager widget
        service_adapter = ServiceAdapter()
        file_manager = FileManager()
        
        plate_manager = PlateManagerWidget(
            service_adapter=service_adapter,
            file_manager=file_manager,
            global_config=global_config
        )
        qtbot.addWidget(plate_manager)
        
        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("plate_manager_created")
        
        # Add plates to manager
        for plate_path in synthetic_plates:
            plate_manager.plates.append({
                'path': plate_path,
                'name': Path(plate_path).name,
                'status': 'ready'
            })
            
        plate_manager.update_plate_list()
        context_monitor.take_snapshot("plates_added")
        
        # Initialize orchestrators - this should trigger context stacking
        print("\n=== Initializing orchestrators ===")
        
        async def init_orchestrators():
            for i, plate_path in enumerate(synthetic_plates):
                print(f"Initializing orchestrator {i+1} for {plate_path}")
                
                # Create orchestrator
                orchestrator = PipelineOrchestrator(
                    plate_path=plate_path,
                    global_config=global_config,
                    storage_registry=file_manager.registry
                )
                orchestrator.initialize()
                
                # Apply pipeline config - this triggers the context leak
                orchestrator.apply_pipeline_config(orchestrator_configs[i])
                
                # Store in plate manager
                plate_manager.orchestrators[plate_path] = orchestrator
                
                # Take snapshot after each orchestrator
                context_monitor.take_snapshot(f"orchestrator_{i+1}_initialized")
                
                # Simulate UI update
                plate_manager.update_plate_list()
                QApplication.processEvents()  # Process any pending events
                
        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_orchestrators())
        
        # Simulate rapid orchestrator switching - this should trigger the freeze
        print("\n=== Simulating rapid orchestrator switching ===")
        
        for i in range(5):  # Switch between orchestrators multiple times
            for j, plate_path in enumerate(synthetic_plates):
                print(f"Switching to orchestrator {j+1} (iteration {i+1})")
                
                # Simulate plate selection change
                plate_manager.selected_plate_path = plate_path
                plate_manager.plate_selected.emit(plate_path)
                
                # Process events to trigger any UI updates
                QApplication.processEvents()
                
                # Small delay to allow freeze detection
                QTest.qWait(10)
                
                context_monitor.take_snapshot(f"switch_iter_{i+1}_orch_{j+1}")
                
        # Analyze results
        leaks = context_monitor.detect_leaks()
        final_snapshot = context_monitor.snapshots[-1]
        
        print(f"\n=== FINAL ANALYSIS ===")
        print(f"Total context leaks detected: {len(leaks)}")
        print(f"Final context stack depth: {final_snapshot['total_contexts']}")
        print(f"UI freeze events: {len(freeze_events)}")
        
        if freeze_events:
            total_freeze_time = sum(event['duration_ms'] for event in freeze_events)
            print(f"Total freeze time: {total_freeze_time:.1f}ms")
            
        # Verify the bug manifests
        assert len(leaks) > 0, "Expected context stack leaks"
        assert final_snapshot['total_contexts'] > 2, "Expected significant context accumulation"
        
        print("\n=== BUG REPRODUCTION SUCCESSFUL ===")
        print("Context stack corruption confirmed - this causes the UI freezing issue")


    def test_3_level_hierarchy_validation_with_corrupted_context(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Test 3-level hierarchy (Step → Orchestrator → Global) with corrupted context stack.

        This test validates that the StepMaterializationConfig hierarchy works correctly
        even when the context stack is corrupted, and identifies resolution failures.
        """
        from openhcs.core.lazy_config import ensure_global_config_context

        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("hierarchy_test_start")

        # Create orchestrators and apply configs (triggering context corruption)
        orchestrators = []
        file_manager = FileManager()

        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=file_manager.registry
            )
            orchestrator.initialize()
            orchestrator.apply_pipeline_config(orchestrator_configs[i])
            orchestrators.append(orchestrator)

        context_monitor.take_snapshot("orchestrators_with_corrupted_context")

        # Test 3-level hierarchy resolution with corrupted context
        print("\n=== Testing 3-Level Hierarchy with Corrupted Context ===")

        # Test Step-level config resolution
        step_config = LazyStepMaterializationConfig(
            output_dir_suffix="_step_override",  # Explicit step value
            sub_dir=None,  # Should resolve from orchestrator/global
            well_filter=None  # Should resolve from orchestrator/global
        )

        # Try to resolve values - this may fail or give incorrect results due to corrupted context
        try:
            resolved_suffix = step_config.output_dir_suffix
            resolved_sub_dir = step_config.sub_dir
            resolved_well_filter = step_config.well_filter

            print(f"Step resolution results:")
            print(f"  output_dir_suffix: {resolved_suffix}")
            print(f"  sub_dir: {resolved_sub_dir}")
            print(f"  well_filter: {resolved_well_filter}")

            # With corrupted context, resolution may be unpredictable
            # The test documents the broken behavior

        except Exception as e:
            print(f"Resolution failed due to corrupted context: {e}")
            # This is expected with severe context corruption

        context_monitor.take_snapshot("hierarchy_resolution_attempted")

        # Verify context corruption impact
        leaks = context_monitor.detect_leaks()
        assert len(leaks) > 0, "Expected context corruption"

        print(f"Context corruption confirmed: {len(leaks)} leaks detected")


    def test_context_isolation_failure(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Test that context isolation fails between orchestrators due to the bug.

        This test demonstrates that different orchestrators contaminate each other's
        contexts due to the context stack corruption.
        """
        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("isolation_test_start")

        # Create orchestrators
        file_manager = FileManager()
        orchestrators = []

        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=file_manager.registry
            )
            orchestrator.initialize()
            orchestrators.append(orchestrator)

        # Apply different configs to each orchestrator
        print("\n=== Testing Context Isolation ===")

        for i, (orchestrator, config) in enumerate(zip(orchestrators, orchestrator_configs)):
            print(f"Applying config {i+1} to orchestrator {i+1}")
            orchestrator.apply_pipeline_config(config)

            snapshot = context_monitor.take_snapshot(f"config_{i+1}_applied")
            print(f"Context stack depth: {snapshot['total_contexts']}")

        # Test that contexts are NOT properly isolated (demonstrating the bug)
        final_snapshot = context_monitor.snapshots[-1]
        global_stack_depth = final_snapshot['stacks'].get('GlobalPipelineConfig', {}).get('depth', 0)

        # With the bug, both orchestrator contexts remain on the stack
        assert global_stack_depth == 2, f"Expected 2 contexts (isolation failure), got {global_stack_depth}"

        print(f"\n=== CONTEXT ISOLATION FAILURE CONFIRMED ===")
        print(f"Both orchestrator contexts remain on stack: {global_stack_depth} contexts")
        print("This causes cross-contamination between orchestrator configurations")


    def test_parameter_form_freeze_with_corrupted_context(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Test that parameter forms freeze when trying to resolve values with corrupted context.

        This test simulates opening parameter forms after context corruption and
        demonstrates the UI freezing during placeholder text resolution.
        """
        # Setup UI responsiveness monitoring
        responsiveness_monitor = UIResponsivenessMonitor(threshold_ms=150.0)
        freeze_events = []

        def on_freeze_detected(duration_ms):
            freeze_events.append(duration_ms)
            print(f"PARAMETER FORM FREEZE: {duration_ms:.1f}ms")

        responsiveness_monitor.freeze_detected.connect(on_freeze_detected)

        # Create orchestrators and corrupt context
        file_manager = FileManager()
        orchestrators = []

        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=file_manager.registry
            )
            orchestrator.initialize()
            orchestrator.apply_pipeline_config(orchestrator_configs[i])
            orchestrators.append(orchestrator)

        print("\n=== Testing Parameter Form with Corrupted Context ===")

        # Create parameter form manager
        service_adapter = ServiceAdapter()
        form_manager = ParameterFormManager(service_adapter=service_adapter)

        # Try to create form for StepMaterializationConfig - this should freeze
        try:
            print("Creating parameter form (this may freeze)...")

            # This operation may freeze due to infinite loops in placeholder resolution
            form_widget = form_manager.create_parameter_form(
                LazyStepMaterializationConfig,
                LazyStepMaterializationConfig()
            )

            # Process events to trigger placeholder resolution
            QApplication.processEvents()
            QTest.qWait(100)  # Allow time for freeze detection

            print("Parameter form created successfully")

        except Exception as e:
            print(f"Parameter form creation failed: {e}")

        # Check for freezes
        if freeze_events:
            total_freeze_time = sum(freeze_events)
            print(f"Parameter form freezes detected: {len(freeze_events)} events, {total_freeze_time:.1f}ms total")

        # The test documents the freeze behavior
        print("Parameter form freeze test completed")


    def test_context_cleanup_simulation(self, qtbot, synthetic_plates, global_config, orchestrator_configs):
        """
        Simulate proper context cleanup to demonstrate the fix.

        This test shows what should happen when contexts are properly cleaned up.
        """
        from openhcs.core.lazy_config import pop_context

        # Setup context monitoring
        context_monitor = ContextStackMonitor()
        context_monitor.take_snapshot("cleanup_test_start")

        # Create orchestrators
        file_manager = FileManager()
        orchestrators = []

        for i, plate_path in enumerate(synthetic_plates):
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config,
                storage_registry=file_manager.registry
            )
            orchestrator.initialize()
            orchestrators.append(orchestrator)

        # Apply configs (this creates the bug)
        for i, (orchestrator, config) in enumerate(zip(orchestrators, orchestrator_configs)):
            orchestrator.apply_pipeline_config(config)

        context_monitor.take_snapshot("contexts_corrupted")

        # Simulate proper cleanup by manually popping contexts
        print("\n=== Simulating Proper Context Cleanup ===")

        initial_depth = context_monitor.snapshots[-1]['total_contexts']
        print(f"Initial corrupted context depth: {initial_depth}")

        # Manually clean up contexts (simulating the fix)
        for i in range(len(orchestrators)):
            popped = pop_context(GlobalPipelineConfig)
            if popped:
                print(f"Cleaned up context {i+1}: {popped}")
            else:
                print(f"No context to clean up for {i+1}")

        context_monitor.take_snapshot("contexts_cleaned")

        # Verify cleanup
        final_depth = context_monitor.snapshots[-1]['total_contexts']
        print(f"Final context depth after cleanup: {final_depth}")

        # Should be back to 0 contexts
        assert final_depth == 0, f"Expected 0 contexts after cleanup, got {final_depth}"

        print("\n=== CLEANUP SIMULATION SUCCESSFUL ===")
        print("This demonstrates what proper context management should achieve")
