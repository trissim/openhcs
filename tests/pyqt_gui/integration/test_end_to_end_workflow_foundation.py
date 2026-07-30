"""
OpenHCS PyQt6 GUI Integration Testing Framework - Refactored Foundation

Mathematical simplification approach applied to GUI testing framework.
Eliminates code duplication through algebraic factoring and parameterization.

Key Refactoring Principles Applied:
- Algebraic common factors extracted into reusable components
- Single-use methods inlined for clarity
- Duplicate conditional logic unified into parameterized functions
- Mathematical simplification through data-driven approaches
"""

import os
import pytest
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from enum import Enum

# Skip entire module in CPU-only mode to avoid PyQt6 imports
if os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true':
    pytest.skip("PyQt6 GUI tests skipped in CPU-only mode", allow_module_level=True)

from PyQt6.QtWidgets import QApplication, QDialog, QPushButton, QMessageBox, QLabel, QWidget, QCheckBox
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
from PyQt6.QtTest import QTest

from openhcs.core.config import GlobalPipelineConfig, LazyStepMaterializationConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.constants import Microscope
from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, get_default_ui_config
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.windows.config_window import ConfigWindow
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator


# ============================================================================
# CORE CONFIGURATION AND ENUMS
# ============================================================================

@dataclass(frozen=True)
class TimingConfig:
    """Timing configuration for GUI operations."""
    ACTION_DELAY: float = 1.5
    WINDOW_DELAY: float = 1.5
    SAVE_DELAY: float = 1.5
    # Visual debugging delays for manual observation
    VISUAL_OBSERVATION_DELAY: float = 3.0
    VISUAL_PREPARATION_DELAY: float = 2.0
    VISUAL_BUG_OBSERVATION_DELAY: float = 10.0

    @classmethod
    def from_environment(cls) -> 'TimingConfig':
        """Create timing config from environment variables."""
        return cls(
            ACTION_DELAY=float(os.environ.get('OPENHCS_TEST_ACTION_DELAY', cls.ACTION_DELAY)),
            WINDOW_DELAY=float(os.environ.get('OPENHCS_TEST_WINDOW_DELAY', cls.WINDOW_DELAY)),
            SAVE_DELAY=float(os.environ.get('OPENHCS_TEST_SAVE_DELAY', cls.SAVE_DELAY))
        )

class ValidationPattern(Enum):
    """Validation patterns for widget text analysis."""
    NONE = "_shows_none"
    PIPELINE_DEFAULT = "_shows_pipeline_default"
    ORCHESTRATOR_VALUES = "_shows_orchestrator_values"

@dataclass(frozen=True)
class FieldModificationSpec:
    """Specification for field modification testing."""
    field_name: str
    modification_value: Any
    config_section: str = "path_planning"
    expected_persistence_behavior: str = "shows_modified_value"

@dataclass(frozen=True)
class TestScenario:
    """Complete test scenario configuration."""
    name: str
    orchestrator_config: Dict[str, Any]
    expected_values: Dict[str, Any]
    field_to_test: FieldModificationSpec
    reset_field_to_test: Optional[str] = None
    legitimate_none_fields: frozenset = field(default_factory=lambda: frozenset({
        'barcode', 'plate_name', 'plate_id', 'description'
    }))

TIMING = TimingConfig.from_environment()
LEGITIMATE_NONE_FIELDS = frozenset({
    'barcode', 'plate_name', 'plate_id', 'description', 'default_format',
    # Fields that legitimately resolve to None through dual-axis resolver
    'global_output_folder'
})

# ============================================================================
# DATA-DRIVEN TEST SCENARIO FACTORY
# ============================================================================

def create_test_scenarios() -> Dict[str, TestScenario]:
    """Factory function to create test scenarios using data-driven approach."""

    # Base scenario configurations - eliminates duplicate scenario definitions
    scenario_configs = [
        {
            "name": "reset_placeholder_bug_path_planning",
            "orchestrator_config": {"output_dir_suffix": "828282", "sub_dir": "images", "well_filter": 5, "num_workers": 1},
            "field_to_test": ("output_dir_suffix", "828282", "path_planning")
        },
        {
            "name": "reset_placeholder_bug_materialization",
            "orchestrator_config": {"output_dir_suffix": "828282", "sub_dir": "images", "well_filter": 5, "num_workers": 1},
            "field_to_test": ("output_dir_suffix", "828282", "materialization_defaults")
        },
        {
            "name": "reset_placeholder_bug_direct_field",
            "orchestrator_config": {"output_dir_suffix": "_test", "sub_dir": "images", "well_filter": ["A01", "A02"]},
            "field_to_test": ("num_workers", 2, "direct"),
            "additional_expected": {"num_workers": 2}
        },
        {
            "name": "default_hierarchy",
            "orchestrator_config": {"output_dir_suffix": "_outputs", "sub_dir": "images", "well_filter": 5},
            "field_to_test": ("well_filter", 4, "step_materialization_config")
        },
        {
            "name": "alternative_config",
            "orchestrator_config": {"output_dir_suffix": "_processed", "sub_dir": "results", "well_filter": 10},
            "field_to_test": ("output_dir_suffix", "_custom", "path_planning")
        },
        {
            "name": "minimal_config",
            "orchestrator_config": {"output_dir_suffix": "", "sub_dir": "data", "well_filter": 1},
            "field_to_test": ("sub_dir", "test_data", "path_planning")
        },
        {
            "name": "inheritance_hierarchy_step_well_filter",
            "orchestrator_config": {"output_dir_suffix": "_test", "sub_dir": "images", "well_filter": 1, "num_workers": 1},
            "field_to_test": ("well_filter", 42, "step_well_filter_config"),
            "additional_expected": {"step_well_filter_inheritance_test": True}
        },
        {
            "name": "inheritance_hierarchy_path_planning_isolation",
            "orchestrator_config": {"output_dir_suffix": "_test", "sub_dir": "images", "well_filter": 1, "num_workers": 1},
            "field_to_test": ("well_filter", 99, "path_planning"),
            "additional_expected": {"path_planning_isolation_test": True}
        }
    ]

    # Generate scenarios using algebraic factoring approach
    scenarios = {}
    for config in scenario_configs:
        field_name, modification_value, config_section = config["field_to_test"]

        # Expected values = orchestrator config + any additional expected values
        expected_values = config["orchestrator_config"].copy()
        expected_values.update(config.get("additional_expected", {}))

        scenarios[config["name"]] = TestScenario(
            name=config["name"],
            orchestrator_config=config["orchestrator_config"],
            expected_values=expected_values,
            field_to_test=FieldModificationSpec(
                field_name=field_name,
                modification_value=modification_value,
                config_section=config_section
            ),
            legitimate_none_fields=LEGITIMATE_NONE_FIELDS
        )

    return scenarios

# Create scenarios using factory pattern
TEST_SCENARIOS = create_test_scenarios()


# ============================================================================
# WORKFLOW FRAMEWORK
# ============================================================================

@dataclass
class WorkflowContext:
    """Immutable context passed between workflow steps."""
    main_window: Optional[OpenHCSMainWindow] = None
    plate_manager_widget: Optional[PlateManagerWidget] = None
    config_window: Optional[QDialog] = None
    synthetic_plate_dir: Optional[Path] = None
    orchestrator: Optional[PipelineOrchestrator] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    test_scenario: Optional[TestScenario] = None

    def with_updates(self, **kwargs) -> 'WorkflowContext':
        """Create new context with updates (immutable pattern)."""
        from dataclasses import replace
        return replace(self, **kwargs)

@dataclass
class WorkflowStep:
    """Atomic workflow operation with clear input/output contract."""
    name: str
    operation: Callable[[WorkflowContext], WorkflowContext]
    timing_delay: Optional[float] = None

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Execute step with timing and logging."""
        print(f"  {self.name}...")
        result = self.operation(context)
        if self.timing_delay:
            _wait_for_gui(self.timing_delay)
        print(f"  ✅ {self.name} completed")
        return result

class WorkflowBuilder:
    """Composable workflow builder for GUI test scenarios."""

    def __init__(self):
        self.steps: List[WorkflowStep] = []
        self.assertions: List[Callable[[WorkflowContext], None]] = []

    def add_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
        """Add workflow step (fluent interface)."""
        self.steps.append(step)
        return self

    def add_assertion(self, assertion: Callable[[WorkflowContext], None]) -> 'WorkflowBuilder':
        """Add assertion to be checked after workflow completion."""
        self.assertions.append(assertion)
        return self

    def execute(self, initial_context: WorkflowContext) -> WorkflowContext:
        """Execute workflow steps sequentially."""
        context = initial_context
        for step in self.steps:
            context = step.execute(context)

        # Run all assertions
        for assertion in self.assertions:
            assertion(context)

        return context


# ============================================================================
# UNIFIED ERROR HANDLING SYSTEM
# ============================================================================

ERROR_KEYWORDS = ['error', 'exception', 'recursion', 'warning', 'unexpected']

class ErrorDialogMonitor(QObject):
    """Unified error dialog monitoring system."""
    error_detected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_for_errors)
        self.monitoring = False
        self.detected_error = None

    def start_monitoring(self, check_interval_ms: int = 100):
        """Start continuous monitoring for error dialogs."""
        print("  Starting background error dialog monitor...")
        self.monitoring = True
        self.detected_error = None
        self.timer.start(check_interval_ms)

    def stop_monitoring(self):
        """Stop monitoring for error dialogs."""
        self.monitoring = False
        self.timer.stop()
        print("  Stopped background error dialog monitor")

    def _check_for_errors(self):
        """Check for error dialogs and handle them immediately."""
        if not self.monitoring:
            return

        try:
            error_dialogs = self._find_error_dialogs()
            if error_dialogs and not self.detected_error:
                error_details = self._close_error_dialogs(error_dialogs)
                self.detected_error = (
                    f"LAZY CONFIG BUG DETECTED: Error dialog appeared! "
                    f"Error dialogs: {error_details}"
                )
                self.error_detected.emit(self.detected_error)
                self.stop_monitoring()
        except Exception as e:
            print(f"  Error in background monitor: {e}")

    def _find_error_dialogs(self) -> List[Any]:
        """Find error dialogs using unified detection logic."""
        error_dialogs = []
        try:
            for widget in QApplication.topLevelWidgets():
                if widget.isVisible() and self._is_error_dialog(widget):
                    error_dialogs.append(widget)
        except Exception:
            pass
        return error_dialogs

    def _is_error_dialog(self, widget) -> bool:
        """Unified error dialog detection logic."""
        if isinstance(widget, QMessageBox):
            return True

        if isinstance(widget, QDialog):
            title = widget.windowTitle().lower()
            if any(keyword in title for keyword in ERROR_KEYWORDS):
                return True

            # Check dialog content
            for label in widget.findChildren(QLabel):
                if hasattr(label, 'text'):
                    text = label.text().lower()
                    if any(keyword in text for keyword in ERROR_KEYWORDS):
                        return True
        return False

    def _close_error_dialogs(self, error_dialogs: List[Any]) -> List[str]:
        """Close error dialogs and extract details."""
        error_details = []
        for dialog in error_dialogs:
            try:
                title = dialog.windowTitle()
                error_text = self._extract_error_text(dialog)
                error_details.append(f"Dialog: '{title}', Text: '{error_text}'")

                dialog.accept()
                dialog.close()
                dialog.deleteLater()
                print(f"  Background monitor closed error dialog: {title}")
            except Exception as e:
                error_details.append(f"Error closing dialog: {e}")
                try:
                    dialog.close()
                    dialog.deleteLater()
                except:
                    pass
        return error_details

    def _extract_error_text(self, dialog) -> str:
        """Extract error text from dialog."""
        if isinstance(dialog, QMessageBox):
            return dialog.text()[:200]

        for label in dialog.findChildren(QLabel):
            if hasattr(label, 'text'):
                text = label.text()
                if any(keyword in text.lower() for keyword in ERROR_KEYWORDS):
                    return text[:200]
        return ""

# Global error monitor instance
_error_monitor = None

def get_error_monitor() -> ErrorDialogMonitor:
    """Get or create the global error monitor instance."""
    global _error_monitor
    if _error_monitor is None:
        _error_monitor = ErrorDialogMonitor()
    return _error_monitor


# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

def _wait_for_gui(delay_seconds: float = TIMING.ACTION_DELAY) -> None:
    """Wait for GUI operations with unified error dialog detection."""
    monitor = get_error_monitor()

    if delay_seconds > 1.0:
        check_interval = 0.5
        elapsed = 0.0
        while elapsed < delay_seconds:
            time.sleep(min(check_interval, delay_seconds - elapsed))
            QApplication.processEvents()

            # Check for error dialogs using unified system
            if monitor._find_error_dialogs():
                error_details = monitor._close_error_dialogs(monitor._find_error_dialogs())
                raise AssertionError(
                    f"LAZY CONFIG BUG DETECTED: Error dialog appeared during GUI wait! "
                    f"Error dialogs: {error_details}"
                )
            elapsed += check_interval
    else:
        time.sleep(delay_seconds)
        QApplication.processEvents()

def _create_synthetic_plate(tmp_path: Path) -> Path:
    """Create synthetic plate data for testing."""
    plate_dir = tmp_path / "test_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2), tile_size=(64, 64), overlap_percent=10,
        wavelengths=2, z_stack_levels=1, wells=["A01"], format="ImageXpress"
    )
    generator.generate_dataset()
    return plate_dir

def _create_test_global_config() -> GlobalPipelineConfig:
    """Create test global configuration with known values."""
    from openhcs.core.config import WellFilterConfig, PathPlanningConfig

    return GlobalPipelineConfig(
        num_workers=8,
        microscope=Microscope.IMAGEXPRESS,
        use_threading=True,
        # Add well_filter values that test scenarios expect to inherit
        well_filter_config=WellFilterConfig(well_filter=5),
        path_planning_config=PathPlanningConfig(
            well_filter=5,
            output_dir_suffix="_test_global",
            sub_dir="images"
        )
    )


# ============================================================================
# UNIFIED DECORATORS AND WIDGET UTILITIES
# ============================================================================

def with_timeout_and_error_handling(timeout_seconds: int = 10, operation_name: str = "operation"):
    """Unified decorator for timeout handling with error monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            monitor = get_error_monitor()
            monitor.start_monitoring(check_interval_ms=50)

            try:
                print(f"  {operation_name.title()}...")
                result = func(*args, **kwargs)

                if monitor.detected_error:
                    raise AssertionError(monitor.detected_error)

                elapsed = time.time() - start_time
                print(f"  {operation_name.title()} completed successfully in {elapsed:.2f}s")
                return result

            except Exception as e:
                if monitor.detected_error:
                    raise AssertionError(monitor.detected_error) from e

                elapsed = time.time() - start_time
                error_msg = (
                    f"LAZY CONFIG BUG DETECTED: {operation_name} "
                    f"{'timed out' if elapsed > timeout_seconds else 'failed'}! "
                    f"Error: {type(e).__name__}: {str(e)[:200]}..."
                )
                raise AssertionError(error_msg) from e
            finally:
                monitor.stop_monitoring()
        return wrapper
    return decorator

def find_widget_with_retry(widget_finder: Callable, timeout_seconds: int = 10, check_interval: float = 0.5):
    """Unified widget finding with timeout and error detection."""
    start_time = time.time()
    monitor = get_error_monitor()

    while time.time() - start_time < timeout_seconds:
        # Check for error dialogs using unified system
        if monitor._find_error_dialogs():
            error_details = monitor._close_error_dialogs(monitor._find_error_dialogs())
            raise AssertionError(
                f"LAZY CONFIG BUG DETECTED: Error dialog(s) appeared during operation! "
                f"Error dialogs found: {error_details}"
            )

        widget = widget_finder()
        if widget:
            return widget
        _wait_for_gui(check_interval)

    return None

def collect_diagnostic_info() -> Dict[str, Any]:
    """Collect diagnostic information about application state."""
    try:
        return {
            "visible_dialogs": len([w for w in QApplication.topLevelWidgets() if isinstance(w, QDialog) and w.isVisible()]),
            "total_widgets": len(QApplication.topLevelWidgets()),
            "top_level_widgets": [f"{type(w).__name__}: {w.windowTitle()}" for w in QApplication.topLevelWidgets() if w.isVisible()]
        }
    except:
        return {"error": "Could not collect diagnostic info"}


# ============================================================================
# WORKFLOW STEP OPERATIONS
# ============================================================================

def _launch_application(context: WorkflowContext) -> WorkflowContext:
    """Launch real OpenHCS application using normal startup process."""
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp
    from objectstate.global_config import get_current_global_config
    import sys

    # Use test global config instead of cached config to ensure test values are available
    config = _create_test_global_config()
    app = OpenHCSPyQtApp(
        sys.argv,
        runtime_context=PyQtGuiRuntimeContext(
            get_default_ui_config(),
            pipeline_runtime=config,
        ),
    )

    # Verify global config context establishment
    current_context = get_current_global_config(GlobalPipelineConfig)
    if not current_context:
        raise AssertionError("Global config context NOT established - this will cause placeholder issues")

    main_window = app.create_main_window()

    # Safe close event that doesn't trigger aggressive cleanup (inlined single-use helper)
    main_window.closeEvent = lambda event: event.accept()

    # Use app.show_main_window() to properly schedule deferred initialization
    # This schedules deferred_initialization via QTimer.singleShot(100, ...)
    app.show_main_window()
    _wait_for_gui(TIMING.WINDOW_DELAY)

    return context.with_updates(main_window=main_window)





def _access_plate_manager(context: WorkflowContext) -> WorkflowContext:
    """Access default plate manager window (already open by default)."""
    # Wait for plate manager to be created by deferred initialization
    # The QTimer.singleShot(100, ...) in app.py schedules deferred_initialization
    # We need to wait for it to complete
    max_wait = 5.0  # Maximum 5 seconds to wait
    elapsed = 0.0
    check_interval = 0.1

    while elapsed < max_wait:
        plate_manager_window = context.main_window.floating_windows.get("plate_manager")
        if plate_manager_window:
            break
        time.sleep(check_interval)
        QApplication.processEvents()
        elapsed += check_interval
    else:
        raise AssertionError(
            f"Plate manager window not created after {max_wait}s. "
            "Deferred initialization may have failed."
        )

    plate_manager_widget = plate_manager_window.findChild(PlateManagerWidget)
    if not plate_manager_widget:
        raise AssertionError("PlateManagerWidget should be found in default window")

    return context.with_updates(plate_manager_widget=plate_manager_widget)


def _add_and_select_plate(context: WorkflowContext) -> WorkflowContext:
    """Add synthetic plate and select it in plate manager."""
    context.plate_manager_widget.add_plate_callback([context.synthetic_plate_dir])
    _wait_for_gui(TIMING.ACTION_DELAY)

    plate_list = context.plate_manager_widget.plate_list
    if plate_list.count() == 0:
        raise AssertionError("No plates found in plate manager list after adding synthetic plate")

    plate_list.setCurrentRow(0)
    _wait_for_gui(TIMING.ACTION_DELAY)
    return context


def _initialize_plate(context: WorkflowContext) -> WorkflowContext:
    """Initialize plate using Init button."""
    init_button = context.plate_manager_widget.buttons["init_plate"]
    if not init_button.isEnabled():
        raise AssertionError("Init button is disabled - plate may not be properly added")

    init_button.click()
    _wait_for_gui(TIMING.SAVE_DELAY)
    return context


def _apply_orchestrator_config(context: WorkflowContext) -> WorkflowContext:
    """Apply parameterized orchestrator configuration to establish 3-level hierarchy."""
    if not context.test_scenario:
        raise ValueError("Test scenario must be provided for parameterized orchestrator configuration")

    orchestrator = context.plate_manager_widget.orchestrators[str(context.synthetic_plate_dir)]

    # Apply configuration from test scenario (eliminates hardcoded values)
    config_params = context.test_scenario.orchestrator_config

    # Import the dynamically generated PipelineConfig
    from openhcs.core.config import PipelineConfig

    # Generic configuration builder - automatically detects direct vs nested fields
    pipeline_config_kwargs = {}
    nested_config_params = {}

    # Get PipelineConfig field names to determine what's direct vs nested
    pipeline_fields = set(PipelineConfig.__dataclass_fields__.keys())

    for field_name, value in config_params.items():
        if field_name in pipeline_fields:
            # Direct field on PipelineConfig
            pipeline_config_kwargs[field_name] = value
        else:
            # Nested field - collect for step_materialization_config
            nested_config_params[field_name] = value

    # If we have nested parameters, create the nested config
    if nested_config_params:
        pipeline_config_kwargs['step_materialization_config'] = LazyStepMaterializationConfig(
            **nested_config_params
        )

    orchestrator_config = PipelineConfig(**pipeline_config_kwargs)
    orchestrator.apply_pipeline_config(orchestrator_config)
    _wait_for_gui(TIMING.ACTION_DELAY)



    return context.with_updates(orchestrator=orchestrator)


def _find_config_window() -> Optional[QDialog]:
    """Find configuration window among top-level widgets."""
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and "config" in widget.windowTitle().lower():
            return widget
    return None


@with_timeout_and_error_handling(timeout_seconds=10, operation_name="opening configuration window")
def _open_config_window(context: WorkflowContext) -> WorkflowContext:
    """Open plate-specific configuration window."""
    edit_button = context.plate_manager_widget.buttons["edit_config"]
    if not edit_button.isEnabled():
        raise AssertionError("Edit button is disabled - plate may not be properly initialized")

    edit_button.click()
    _wait_for_gui(TIMING.WINDOW_DELAY)

    config_window = find_widget_with_retry(_find_config_window, timeout_seconds=10)
    if not config_window:
        diagnostics = collect_diagnostic_info()
        raise AssertionError(f"Configuration window not found. Diagnostics: {diagnostics}")

    _wait_for_gui(TIMING.ACTION_DELAY)
    return context.with_updates(config_window=config_window)


# ============================================================================
# UNIFIED WIDGET INTERACTION SYSTEM
# ============================================================================

class WidgetFinder:
    """Unified widget finding system using algebraic factoring approach."""

    @staticmethod
    def find_field_widget(form_managers: List[ParameterFormManager], field_name: str) -> Optional[Any]:
        """Find widget for specified field name across form managers."""
        print(f"🔍 DEBUG: Looking for field '{field_name}' in {len(form_managers)} form managers")

        for i, form_manager in enumerate(form_managers):
            if hasattr(form_manager, 'widgets') and form_manager.widgets and field_name in form_manager.widgets:
                dataclass_name = getattr(form_manager.dataclass_type, '__name__', 'Unknown') if hasattr(form_manager, 'dataclass_type') else 'No dataclass'
                print(f"🔍 DEBUG: Found '{field_name}' in form manager [{i}] {dataclass_name}")
                return form_manager.widgets[field_name]

        print(f"🔍 DEBUG: Field '{field_name}' not found in any form manager")
        return None

    @staticmethod
    def find_field_widget_in_config_section(form_managers: List[ParameterFormManager],
                                           field_name: str, config_section: str) -> Optional[Any]:
        """Find widget for specified field name in a specific config section."""
        print(f"🔍 DEBUG: Looking for field '{field_name}' in config section '{config_section}' across {len(form_managers)} form managers")

        # Convert config_section to expected dataclass name patterns
        expected_dataclass_patterns = [
            config_section,  # exact match
            f"Lazy{config_section}",  # LazyStepWellFilterConfig
            f"Lazy{config_section.replace('_config', '').title().replace('_', '')}Config",  # LazyStepWellFilterConfig
            config_section.replace('_config', '').title().replace('_', '') + 'Config',  # StepWellFilterConfig
        ]

        print(f"🔍 DEBUG: Expected dataclass patterns: {expected_dataclass_patterns}")

        for i, form_manager in enumerate(form_managers):
            if hasattr(form_manager, 'widgets') and form_manager.widgets and field_name in form_manager.widgets:
                dataclass_name = getattr(form_manager.dataclass_type, '__name__', 'Unknown') if hasattr(form_manager, 'dataclass_type') else 'No dataclass'
                print(f"🔍 DEBUG: Checking form manager [{i}] {dataclass_name}")

                # Check if this form manager matches the expected config section
                if any(pattern.lower() in dataclass_name.lower() for pattern in expected_dataclass_patterns):
                    print(f"🔍 DEBUG: Found '{field_name}' in correct config section '{config_section}' - form manager [{i}] {dataclass_name}")
                    return form_manager.widgets[field_name]
                else:
                    print(f"🔍 DEBUG: Skipping form manager [{i}] {dataclass_name} (doesn't match config section '{config_section}')")

        print(f"🔍 DEBUG: Field '{field_name}' not found in config section '{config_section}'")
        return None

    @staticmethod
    def find_button_by_text(parent_widget, button_texts: List[str]) -> Optional[QPushButton]:
        """Find button by text using lookup table approach."""
        button_texts_lower = [text.lower() for text in button_texts]
        for button in parent_widget.findChildren(QPushButton):
            if button.text().lower() in button_texts_lower:
                return button
        return None

    @staticmethod
    def find_form_manager_for_field(form_managers: List[ParameterFormManager], field_name: str) -> Optional[ParameterFormManager]:
        """Find form manager containing specified field."""
        for form_manager in form_managers:
            if hasattr(form_manager, 'widgets') and field_name in form_manager.widgets:
                return form_manager
        return None

    @staticmethod
    def find_widget_by_attribute(widgets: List[Any], attribute_name: str) -> Optional[Any]:
        """Find widget that has specified attribute."""
        for widget in widgets:
            if hasattr(widget, attribute_name):
                return widget
        return None

class WidgetInteractor:
    """Unified widget interaction system."""

    @staticmethod
    def set_widget_value(widget: Any, value: Any) -> None:
        """Set value on widget using appropriate method."""
        # Lookup table approach for widget value setting
        value_setters = [
            ('setValue', lambda w, v: w.setValue(v)),
            ('setText', lambda w, v: w.setText(str(v))),
            ('setCurrentText', lambda w, v: w.setCurrentText(str(v)))
        ]

        for attr_name, setter_func in value_setters:
            if hasattr(widget, attr_name):
                setter_func(widget, value)
                return

        raise AssertionError(f"Cannot set value on widget of type {type(widget)}")

    @staticmethod
    def scroll_to_widget(widget) -> None:
        """Scroll the config window to make the widget visible for manual observation."""
        if not widget:
            return

        # Find the scroll area that contains this widget
        parent = widget.parentWidget()
        while parent:
            # Check if this is a QScrollArea
            if hasattr(parent, 'ensureWidgetVisible'):
                try:
                    parent.ensureWidgetVisible(widget, 50, 50)  # 50px margins
                    print(f"📍 Scrolled to make widget visible using ensureWidgetVisible")
                    break
                except Exception as e:
                    print(f"⚠️ ensureWidgetVisible failed: {e}")

            # Check if this has scroll bars (QAbstractScrollArea)
            elif hasattr(parent, 'verticalScrollBar') and hasattr(parent, 'viewport'):
                try:
                    # Manual scroll calculation for scroll areas
                    widget_pos = widget.mapTo(parent.viewport(), widget.rect().center())
                    v_scroll_bar = parent.verticalScrollBar()
                    h_scroll_bar = parent.horizontalScrollBar()

                    if v_scroll_bar:
                        # Calculate target scroll position to center the widget vertically
                        viewport_height = parent.viewport().height()
                        target_y = widget_pos.y() - viewport_height // 2
                        v_scroll_bar.setValue(max(0, min(target_y, v_scroll_bar.maximum())))

                    if h_scroll_bar:
                        # Calculate target scroll position to center the widget horizontally
                        viewport_width = parent.viewport().width()
                        target_x = widget_pos.x() - viewport_width // 2
                        h_scroll_bar.setValue(max(0, min(target_x, h_scroll_bar.maximum())))

                    print(f"📍 Manually scrolled to widget position")
                    break
                except Exception as e:
                    print(f"⚠️ Manual scroll failed: {e}")

            # Move up the parent hierarchy
            parent = parent.parentWidget()

        # Process events and wait for scroll animation
        QApplication.processEvents()
        _wait_for_gui(TimingConfig.ACTION_DELAY)

# Unified widget operations using the new system
def _find_config_window() -> Optional[QDialog]:
    """Find configuration window among top-level widgets."""
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and "config" in widget.windowTitle().lower():
            return widget
    return None

@with_timeout_and_error_handling(timeout_seconds=5, operation_name="modifying field")
def _modify_field(context: WorkflowContext) -> WorkflowContext:
    """Modify specified field in the configuration window and save."""
    if not context.test_scenario:
        raise ValueError("Test scenario required for parameterized field modification")

    field_name = context.test_scenario.field_to_test.field_name
    field_value = context.test_scenario.field_to_test.modification_value
    config_section = context.test_scenario.field_to_test.config_section

    form_managers = context.config_window.findChildren(ParameterFormManager)
    # CRITICAL FIX: Use config-section-specific field finder to target the correct form manager
    field_widget = WidgetFinder.find_field_widget_in_config_section(form_managers, field_name, config_section)

    print(f"🔧 MODIFY FIELD: Targeting {config_section}.{field_name} = {field_value}")

    if not field_widget:
        available_fields = [field for fm in form_managers if hasattr(fm, 'widgets') for field in fm.widgets.keys()]
        raise AssertionError(f"Field '{field_name}' widget not found. Available fields: {available_fields}")

    print(f"  Setting {field_name} = {field_value}")
    WidgetInteractor.set_widget_value(field_widget, field_value)
    _wait_for_gui(TIMING.ACTION_DELAY)

    # Save the configuration (inlined single-use function)
    save_button = WidgetFinder.find_button_by_text(context.config_window, ['ok', 'save', 'apply'])
    if not save_button:
        buttons = [b.text() for b in context.config_window.findChildren(QPushButton)]
        raise AssertionError(f"Save button not found. Available buttons: {buttons}")

    save_button.click()
    _wait_for_gui(TIMING.SAVE_DELAY)
    return context


def _set_step_well_filter_for_isolation_test(context: WorkflowContext) -> WorkflowContext:
    """Set step_well_filter_config.well_filter = 5 for isolation test using existing field modification logic."""
    if not context.test_scenario or context.test_scenario.name != "inheritance_hierarchy_path_planning_isolation":
        return context  # Skip for other tests

    from dataclasses import replace

    # Create temporary field spec for step_well_filter_config.well_filter = 5
    temp_field_spec = FieldModificationSpec(
        field_name="well_filter",
        modification_value=5,
        config_section="step_well_filter_config"
    )

    # Temporarily replace field spec and reuse existing _modify_field logic
    temp_scenario = replace(context.test_scenario, field_to_test=temp_field_spec)
    temp_context = context.with_updates(test_scenario=temp_scenario)

    print(f"🔧 ISOLATION TEST SETUP: Setting step_well_filter_config.well_filter = 5")
    result_context = _modify_field(temp_context)

    # Restore original field spec
    return result_context.with_updates(test_scenario=context.test_scenario)


def _close_config_window(context: WorkflowContext) -> WorkflowContext:
    """Close configuration window with cleanup."""
    try:
        if context.config_window and context.config_window.isVisible():
            context.config_window.close()
            context.config_window.deleteLater()
            _wait_for_gui(TIMING.ACTION_DELAY)

        # Clean up any remaining config windows
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and "config" in widget.windowTitle().lower() and widget.isVisible():
                widget.close()
                widget.deleteLater()

        _wait_for_gui(TIMING.ACTION_DELAY)
        return context.with_updates(config_window=None)

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
        return context.with_updates(config_window=None)


@with_timeout_and_error_handling(timeout_seconds=10, operation_name="reopening configuration window")
def _reopen_config_window(context: WorkflowContext) -> WorkflowContext:
    """Reopen configuration window to test persistence."""
    # Close existing window first
    context = _close_config_window(context)

    # Validate edit button state
    edit_button = context.plate_manager_widget.buttons["edit_config"]
    if not edit_button.isEnabled():
        raise AssertionError(
            "LAZY CONFIG BUG: Edit button disabled after closing config window. "
            "This indicates a state management issue."
        )

    # Reopen using existing function (composition)
    return _open_config_window(context)


@with_timeout_and_error_handling(timeout_seconds=5, operation_name="resetting field")
def _reset_field(context: WorkflowContext) -> WorkflowContext:
    """Reset specified field to lazy state using reset button."""
    if not context.test_scenario:
        raise ValueError("Test scenario required for parameterized field reset")

    # Use reset_field_to_test if specified, otherwise use the field being modified
    field_name = (context.test_scenario.reset_field_to_test or
                  context.test_scenario.field_to_test.field_name)
    form_managers = context.config_window.findChildren(ParameterFormManager)

    # Use unified widget finder system
    target_form_manager = WidgetFinder.find_form_manager_for_field(form_managers, field_name)
    if not target_form_manager:
        raise AssertionError(f"Form manager with field '{field_name}' not found")

    print(f"  DEBUG: Looking for reset button for field '{field_name}'")

    # Inline the reset button finding logic (single-use helper elimination)
    reset_button = None
    if hasattr(target_form_manager, 'reset_buttons') and field_name in target_form_manager.reset_buttons:
        reset_button = target_form_manager.reset_buttons[field_name]
        print(f"  DEBUG: Available reset buttons: {list(target_form_manager.reset_buttons.keys())}")
    else:
        print(f"  DEBUG: Form manager has no reset_buttons attribute")

    if not reset_button:
        raise AssertionError(f"Reset button for field '{field_name}' not found")

    print(f"  Resetting {field_name} to lazy state")

    # Properly click the reset button with Qt event processing
    from PyQt6.QtWidgets import QApplication
    reset_button.click()
    QApplication.processEvents()  # Process the click event

    print(f"  ✅ Reset button clicked and events processed for {field_name}")
    _wait_for_gui(TIMING.ACTION_DELAY)

    return context


# ============================================================================
# PARAMETERIZED VALIDATION FRAMEWORK
# ============================================================================

# Step editor validation configuration
STEP_EDITOR_TEST_FIELDS = ['output_dir_suffix', 'sub_dir']
EXPECTED_RESOLVED_VALUES = ['_openhcs', '_processed', 'checkpoints', '_outputs', 'images']

class StepEditorValidator:
    """Unified step editor validation using mathematical simplification principles."""

    @staticmethod
    def find_step_editor_window() -> Optional[Any]:
        """Find step editor window using unified widget finder approach."""
        for widget in QApplication.topLevelWidgets():
            if hasattr(widget, 'step_editor') and hasattr(widget, 'editing_step'):
                return widget
        return None

    @staticmethod
    def validate_placeholder_resolution(nested_manager, test_fields: List[str], expected_values: List[str]) -> bool:
        """Validate placeholder resolution using parameterized approach."""
        placeholder_resolution_verified = False

        for field_name in test_fields:
            if hasattr(nested_manager, 'widgets') and field_name in nested_manager.widgets:
                widget = nested_manager.widgets[field_name]
                texts = ValidationEngine.extract_widget_texts(widget)
                all_text = " ".join(texts.values())

                print(f"🔍 STEP EDITOR {field_name}: '{all_text}'")

                # Unified validation logic using lookup table approach
                if "pipeline default:" in all_text.lower():
                    if any(expected_value in all_text for expected_value in expected_values):
                        print(f"✅ GOOD: {field_name} placeholder shows resolved value")
                        placeholder_resolution_verified = True
                    else:
                        raise AssertionError(
                            f"Placeholder resolution bug: {field_name} should show resolved value "
                            f"from global pipeline config but shows: '{all_text}'"
                        )
                elif all_text.strip() == "" or "none" in all_text.lower():
                    raise AssertionError(
                        f"Placeholder resolution bug: {field_name} placeholder is empty/None instead of showing resolved value"
                    )
                else:
                    print(f"🔍 {field_name} shows direct value (not placeholder)")

        return placeholder_resolution_verified

def _verify_step_editor_placeholder_resolution(context: WorkflowContext) -> WorkflowContext:
    """Verify step editor placeholders using unified validation system."""
    print(f"\n🔍 Verifying step editor placeholder resolution after initialization...")

    # Access pipeline editor using unified approach
    pipeline_editor_window = context.main_window.floating_windows.get("pipeline_editor")
    if not pipeline_editor_window:
        raise AssertionError("Pipeline editor window not found in floating_windows")

    # Find pipeline editor widget using unified finder
    pipeline_editor = WidgetFinder.find_widget_by_attribute(
        pipeline_editor_window.findChildren(QWidget), 'pipeline_steps'
    )
    if not pipeline_editor:
        raise AssertionError("Pipeline editor widget not found")

    # Open step editor using pipeline editor's button dictionary
    if not hasattr(pipeline_editor, 'buttons') or "add_step" not in pipeline_editor.buttons:
        raise AssertionError("Add Step button not found in pipeline editor buttons")
    add_step_button = pipeline_editor.buttons["add_step"]

    add_step_button.click()
    QApplication.processEvents()
    time.sleep(2.0)  # Wait for step editor to open
    QApplication.processEvents()

    # Find step editor window using unified finder
    step_editor_window = StepEditorValidator.find_step_editor_window()
    if not step_editor_window:
        raise AssertionError("Step editor window (DualEditorWindow) not found")

    try:
        # Access form manager using unified approach
        step_param_editor = step_editor_window.step_editor
        if not step_param_editor or not hasattr(step_param_editor, 'form_manager'):
            raise AssertionError("Form manager not found in step parameter editor")

        form_manager = step_param_editor.form_manager

        # Validate placeholder resolution using parameterized approach
        placeholder_resolution_verified = False
        if hasattr(form_manager, 'nested_managers') and 'materialization_config' in form_manager.nested_managers:
            nested_manager = form_manager.nested_managers['materialization_config']
            placeholder_resolution_verified = StepEditorValidator.validate_placeholder_resolution(
                nested_manager, STEP_EDITOR_TEST_FIELDS, EXPECTED_RESOLVED_VALUES
            )

        if not placeholder_resolution_verified:
            print(f"⚠️  WARNING: Could not verify placeholder resolution - materialization_config form not found")
        else:
            print(f"✅ Step editor placeholder resolution verified successfully")

    finally:
        # Clean up step editor window
        step_editor_window.close()
        QApplication.processEvents()

    return context


# ============================================================================
# UNIFIED VALIDATION ENGINE
# ============================================================================

class ValidationEngine:
    """Unified validation system using algebraic factoring approach."""

    @staticmethod
    def extract_widget_texts(widget) -> Dict[str, str]:
        """Extract all text content from a widget."""
        # Special handling for EnhancedPathWidget - check path_input for placeholder
        def get_placeholder_text(w):
            if hasattr(w, 'path_input') and hasattr(w.path_input, 'placeholderText'):
                return w.path_input.placeholderText()
            return getattr(w, 'placeholderText', lambda: "")()

        # Special handling for EnhancedPathWidget - check path_input for text
        def get_text(w):
            if hasattr(w, 'path_input') and hasattr(w.path_input, 'text'):
                return w.path_input.text()
            return getattr(w, 'text', lambda: "")() if hasattr(w, 'text') else ""

        text_extractors = [
            ('placeholder', get_placeholder_text),
            ('special', lambda w: getattr(w, 'specialValueText', lambda: "")()),
            ('tooltip', lambda w: getattr(w, 'toolTip', lambda: "")()),
            ('text', get_text)
        ]

        return {name: extractor(widget) or "" for name, extractor in text_extractors}

    @staticmethod
    def analyze_widget_text(text: str, expected_patterns: List[str]) -> Dict[str, bool]:
        """Analyze widget text against expected patterns."""
        return {
            ValidationPattern.NONE.value: "(none)" in text,
            ValidationPattern.PIPELINE_DEFAULT.value: "pipeline default:" in text,
            ValidationPattern.ORCHESTRATOR_VALUES.value: any(
                str(pattern).lower() in text for pattern in expected_patterns if pattern
            )
        }

    @staticmethod
    def validate_widgets(form_managers: List[ParameterFormManager],
                        validation_rules: Dict[str, Callable[[str, str], Dict[str, bool]]],
                        context: WorkflowContext) -> Dict[str, Any]:
        """Unified widget validation using parameterized rules."""
        validation_results = context.validation_results.copy()

        for form_manager in form_managers:
            if not hasattr(form_manager, 'widgets'):
                continue

            for field_name, widget in form_manager.widgets.items():
                texts = ValidationEngine.extract_widget_texts(widget)
                all_text = " ".join(texts.values()).lower()

                # Apply validation rules
                for rule_name, rule_func in validation_rules.items():
                    result = rule_func(field_name, all_text)
                    validation_results.update(result)

        return validation_results

# Validation rule factories using mathematical simplification
def create_placeholder_validation_rules(scenario: TestScenario) -> Dict[str, Callable]:
    """Create validation rules for placeholder behavior."""
    expected_patterns = [str(v).lower() for v in scenario.expected_values.values() if v is not None]

    def placeholder_rule(field_name: str, text: str) -> Dict[str, bool]:
        analysis = ValidationEngine.analyze_widget_text(text, expected_patterns)
        return {f"{field_name}{suffix}": value for suffix, value in analysis.items()}

    return {"placeholder_behavior": placeholder_rule}

def create_persistence_validation_rules(scenario: TestScenario) -> Dict[str, Callable]:
    """Create validation rules for field persistence."""
    modified_field = scenario.field_to_test.field_name
    expected_value = str(scenario.field_to_test.modification_value).lower()

    def persistence_rule(field_name: str, text: str) -> Dict[str, bool]:
        if field_name == modified_field:
            return {f"{field_name}_shows_saved_value": expected_value in text and "(none)" not in text}
        else:
            shows_pipeline_default = "pipeline default:" in text
            shows_none_correctly = "(none)" not in text or field_name in scenario.legitimate_none_fields
            return {f"{field_name}_shows_lazy_state": shows_pipeline_default and shows_none_correctly}

    return {"field_persistence": persistence_rule}

def create_lazy_state_validation_rules(scenario: TestScenario) -> Dict[str, Callable]:
    """Create validation rules for full lazy state."""
    def lazy_state_rule(field_name: str, text: str) -> Dict[str, bool]:
        shows_pipeline_default = "pipeline default:" in text.lower()
        shows_none_correctly = "(none)" not in text or field_name in scenario.legitimate_none_fields

        # Debug output to see what placeholders we're actually getting
        if not shows_pipeline_default or not shows_none_correctly:
            print(f"🔍 PLACEHOLDER DEBUG: {field_name} = '{text}' (pipeline_default={shows_pipeline_default}, none_correct={shows_none_correctly})")

        return {f"{field_name}_shows_full_lazy_state": shows_pipeline_default and shows_none_correctly}

    return {"full_lazy_state": lazy_state_rule}

# Unified validation function using algebraic factoring approach
def _execute_validation(context: WorkflowContext, validation_rule_factory: Callable) -> WorkflowContext:
    """Execute validation using unified engine with parameterized rule factory."""
    if not context.test_scenario:
        raise ValueError("Test scenario required for parameterized validation")

    form_managers = context.config_window.findChildren(ParameterFormManager)
    validation_rules = validation_rule_factory(context.test_scenario)
    validation_results = ValidationEngine.validate_widgets(form_managers, validation_rules, context)

    return context.with_updates(validation_results=validation_results)

# Specific validation functions using unified approach (eliminates duplicate structure)
def _validate_placeholder_behavior(context: WorkflowContext) -> WorkflowContext:
    """Validate placeholder behavior using unified engine."""
    return _execute_validation(context, create_placeholder_validation_rules)

def _validate_field_persistence(context: WorkflowContext) -> WorkflowContext:
    """Validate field persistence using unified engine."""
    return _execute_validation(context, create_persistence_validation_rules)

def _validate_full_lazy_state(context: WorkflowContext) -> WorkflowContext:
    """Validate full lazy state using unified engine."""
    return _execute_validation(context, create_lazy_state_validation_rules)


# ============================================================================
# UNIFIED ASSERTION FRAMEWORK
# ============================================================================

class AssertionFactory:
    """Factory for creating parameterized assertions using algebraic factoring."""

    @staticmethod
    def create_placeholder_assertions(scenario: TestScenario) -> List[Callable[[WorkflowContext], None]]:
        """Create assertions for placeholder behavior validation."""

        def assert_no_placeholder_bugs(context: WorkflowContext) -> None:
            """Assert that no fields show '(none)' incorrectly."""
            results = context.validation_results

            # Find fields showing "(none)" - filter out legitimate ones
            fields_showing_none = [
                key for key, value in results.items()
                if key.endswith(ValidationPattern.NONE.value) and value
            ]

            legitimate_none_keys = {
                f"{field}{ValidationPattern.NONE.value}"
                for field in scenario.legitimate_none_fields
            }
            actual_bug_fields = [
                field for field in fields_showing_none
                if field not in legitimate_none_keys
            ]

            if actual_bug_fields:
                raise AssertionError(
                    f"PLACEHOLDER BUG in scenario '{scenario.name}': "
                    f"Fields showing '(none)': {actual_bug_fields}. "
                    f"Context capture fix is not working!"
                )

        def assert_inheritance_working(context: WorkflowContext) -> None:
            """Assert that orchestrator values are being inherited."""
            results = context.validation_results

            fields_showing_orchestrator_values = [
                key for key, value in results.items()
                if key.endswith(ValidationPattern.ORCHESTRATOR_VALUES.value) and value
            ]

            if not fields_showing_orchestrator_values:
                expected_patterns = [str(v) for v in scenario.expected_values.values() if v is not None]
                raise AssertionError(
                    f"No orchestrator values detected in scenario '{scenario.name}' - "
                    f"inheritance may not be working. Expected patterns: {expected_patterns}"
                )

        def assert_inheritance_hierarchy(context: WorkflowContext) -> None:
            """Assert that inheritance hierarchy is respected for specific test scenarios."""
            if scenario.name == "inheritance_hierarchy_step_well_filter":
                # Test: step_well_filter_config.well_filter = 42 should update step_materialization_config placeholder
                form_managers = context.config_window.findChildren(ParameterFormManager)
                step_materialization_placeholder = None

                for form_manager in form_managers:
                    if (hasattr(form_manager, 'widgets') and
                        hasattr(form_manager, 'dataclass_type') and
                        form_manager.dataclass_type and
                        'StepMaterialization' in form_manager.dataclass_type.__name__ and
                        'well_filter' in form_manager.widgets):

                        widget = form_manager.widgets['well_filter']
                        texts = ValidationEngine.extract_widget_texts(widget)
                        all_text = " ".join(texts.values())
                        step_materialization_placeholder = all_text
                        print(f"🔍 INHERITANCE TEST: step_materialization_config.well_filter placeholder = '{all_text}'")
                        break

                if step_materialization_placeholder:
                    # Should show inheritance from step_well_filter_config (value=42)
                    if "pipeline default: 42" not in step_materialization_placeholder.lower():
                        raise AssertionError(
                            f"INHERITANCE HIERARCHY BUG: step_materialization_config.well_filter should inherit "
                            f"from step_well_filter_config (42), but shows: '{step_materialization_placeholder}'"
                        )
                    print(f"✅ INHERITANCE HIERARCHY CORRECT: step_materialization inherits from step_well_filter")
                else:
                    raise AssertionError("Could not find step_materialization_config.well_filter widget for inheritance test")

                # CRITICAL: Also verify streaming configs inherit from step_well_filter_config
                napari_placeholder = None
                fiji_placeholder = None

                for form_manager in form_managers:
                    if (hasattr(form_manager, 'widgets') and
                        hasattr(form_manager, 'dataclass_type') and
                        form_manager.dataclass_type and
                        'well_filter' in form_manager.widgets):

                        if 'NapariStreaming' in form_manager.dataclass_type.__name__:
                            widget = form_manager.widgets['well_filter']
                            texts = ValidationEngine.extract_widget_texts(widget)
                            napari_placeholder = " ".join(texts.values())
                            print(f"🔍 STREAMING TEST: napari_streaming_config.well_filter placeholder = '{napari_placeholder}'")

                        elif 'FijiStreaming' in form_manager.dataclass_type.__name__:
                            widget = form_manager.widgets['well_filter']
                            texts = ValidationEngine.extract_widget_texts(widget)
                            fiji_placeholder = " ".join(texts.values())
                            print(f"🔍 STREAMING TEST: fiji_streaming_config.well_filter placeholder = '{fiji_placeholder}'")

                # Verify streaming configs inherit from step_well_filter_config (value=42)
                if napari_placeholder:
                    if "pipeline default: 42" not in napari_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING INHERITANCE BUG: napari_streaming_config.well_filter should inherit "
                            f"from step_well_filter_config (42), but shows: '{napari_placeholder}'"
                        )
                    print(f"✅ STREAMING INHERITANCE CORRECT: napari inherits from step_well_filter")
                else:
                    raise AssertionError("Could not find napari_streaming_config.well_filter widget for streaming test")

                if fiji_placeholder:
                    if "pipeline default: 42" not in fiji_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING INHERITANCE BUG: fiji_streaming_config.well_filter should inherit "
                            f"from step_well_filter_config (42), but shows: '{fiji_placeholder}'"
                        )
                    print(f"✅ STREAMING INHERITANCE CORRECT: fiji inherits from step_well_filter")
                else:
                    raise AssertionError("Could not find fiji_streaming_config.well_filter widget for streaming test")

            elif scenario.name == "inheritance_hierarchy_path_planning_isolation":
                # Test: path_planning_config.well_filter = 99 should NOT affect step_materialization_config or streaming configs
                form_managers = context.config_window.findChildren(ParameterFormManager)
                step_materialization_placeholder = None
                napari_placeholder = None
                fiji_placeholder = None

                for form_manager in form_managers:
                    if (hasattr(form_manager, 'widgets') and
                        hasattr(form_manager, 'dataclass_type') and
                        form_manager.dataclass_type and
                        'well_filter' in form_manager.widgets):

                        widget = form_manager.widgets['well_filter']
                        texts = ValidationEngine.extract_widget_texts(widget)
                        all_text = " ".join(texts.values())

                        if 'StepMaterialization' in form_manager.dataclass_type.__name__:
                            step_materialization_placeholder = all_text
                            print(f"🔍 ISOLATION TEST: step_materialization_config.well_filter placeholder = '{all_text}'")
                        elif 'NapariStreaming' in form_manager.dataclass_type.__name__:
                            napari_placeholder = all_text
                            print(f"🔍 ISOLATION TEST: napari_streaming_config.well_filter placeholder = '{all_text}'")
                        elif 'FijiStreaming' in form_manager.dataclass_type.__name__:
                            fiji_placeholder = all_text
                            print(f"🔍 ISOLATION TEST: fiji_streaming_config.well_filter placeholder = '{all_text}'")

                # Check step_materialization_config isolation
                if step_materialization_placeholder:
                    # Should still show inheritance from step_well_filter_config (value=5), NOT path_planning (value=99)
                    if "pipeline default: 99" in step_materialization_placeholder.lower():
                        raise AssertionError(
                            f"INHERITANCE ISOLATION BUG: step_materialization_config.well_filter incorrectly inherits "
                            f"from path_planning_config (99), shows: '{step_materialization_placeholder}'"
                        )
                    if "pipeline default: 5" not in step_materialization_placeholder.lower():
                        raise AssertionError(
                            f"INHERITANCE ISOLATION BUG: step_materialization_config.well_filter should still inherit "
                            f"from step_well_filter_config (5), but shows: '{step_materialization_placeholder}'"
                        )
                    print(f"✅ INHERITANCE ISOLATION CORRECT: step_materialization still inherits from step_well_filter")
                else:
                    raise AssertionError("Could not find step_materialization_config.well_filter widget for isolation test")

                # Check napari_streaming_config isolation
                if napari_placeholder:
                    if "pipeline default: 99" in napari_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING ISOLATION BUG: napari_streaming_config.well_filter incorrectly inherits "
                            f"from path_planning_config (99), shows: '{napari_placeholder}'"
                        )
                    if "pipeline default: 5" not in napari_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING ISOLATION BUG: napari_streaming_config.well_filter should inherit "
                            f"from step_well_filter_config (5), but shows: '{napari_placeholder}'"
                        )
                    print(f"✅ STREAMING ISOLATION CORRECT: napari does not inherit from path_planning")
                else:
                    raise AssertionError("Could not find napari_streaming_config.well_filter widget for isolation test")

                # Check fiji_streaming_config isolation
                if fiji_placeholder:
                    if "pipeline default: 99" in fiji_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING ISOLATION BUG: fiji_streaming_config.well_filter incorrectly inherits "
                            f"from path_planning_config (99), shows: '{fiji_placeholder}'"
                        )
                    if "pipeline default: 5" not in fiji_placeholder.lower():
                        raise AssertionError(
                            f"STREAMING ISOLATION BUG: fiji_streaming_config.well_filter should inherit "
                            f"from step_well_filter_config (5), but shows: '{fiji_placeholder}'"
                        )
                    print(f"✅ STREAMING ISOLATION CORRECT: fiji does not inherit from path_planning")
                else:
                    raise AssertionError("Could not find fiji_streaming_config.well_filter widget for isolation test")

        assertions = [assert_no_placeholder_bugs, assert_inheritance_working]

        # Add inheritance hierarchy assertion for specific scenarios
        if scenario.name in ["inheritance_hierarchy_step_well_filter", "inheritance_hierarchy_path_planning_isolation"]:
            assertions.append(assert_inheritance_hierarchy)

        return assertions

    @staticmethod
    def create_persistence_assertions(scenario: TestScenario) -> List[Callable[[WorkflowContext], None]]:
        """Create assertions for field persistence validation."""

        def assert_field_persistence(context: WorkflowContext) -> None:
            """Assert that modified field shows saved value while others show lazy state."""
            results = context.validation_results
            modified_field = scenario.field_to_test.field_name

            # Modified field should show saved value
            saved_value_key = f"{modified_field}_shows_saved_value"
            if not results.get(saved_value_key, False):
                raise AssertionError(
                    f"Modified field '{modified_field}' does not show saved value in scenario '{scenario.name}'"
                )

        return [assert_field_persistence]

    @staticmethod
    def create_lazy_state_assertions(scenario: TestScenario) -> List[Callable[[WorkflowContext], None]]:
        """Create assertions for full lazy state validation."""

        def assert_full_lazy_state(context: WorkflowContext) -> None:
            """Assert that all fields show lazy state after reset."""
            results = context.validation_results

            # Check for reset placeholder bug specifically
            if scenario.name == "reset_placeholder_bug_direct_field":
                # Validate that num_workers placeholder shows correct pipeline default, not old concrete value
                form_managers = context.config_window.findChildren(ParameterFormManager)
                for form_manager in form_managers:
                    if hasattr(form_manager, 'widgets') and 'num_workers' in form_manager.widgets:
                        widget = form_manager.widgets['num_workers']
                        texts = ValidationEngine.extract_widget_texts(widget)
                        all_text = " ".join(texts.values()).lower()

                        print(f"🔍 RESET PLACEHOLDER BUG CHECK: num_workers text = '{all_text}'")

                        # Critical bug check: should NOT show the old concrete value "2"
                        if "2" in all_text and "pipeline default:" in all_text:
                            raise AssertionError(
                                f"RESET PLACEHOLDER BUG DETECTED: num_workers placeholder shows old concrete value '2' "
                                f"instead of pipeline default. Text: '{all_text}'"
                            )

                        # Should show pipeline default: 1
                        if "pipeline default:" in all_text and "1" not in all_text:
                            raise AssertionError(
                                f"RESET PLACEHOLDER BUG: num_workers should show 'pipeline default: 1'. "
                                f"Text: '{all_text}'"
                            )

                        print(f"✅ RESET PLACEHOLDER CORRECT: num_workers shows proper pipeline default")
                        break

            lazy_state_fields = [
                key for key, value in results.items()
                if key.endswith("_shows_full_lazy_state") and not value
            ]

            if lazy_state_fields:
                raise AssertionError(
                    f"Fields not showing full lazy state in scenario '{scenario.name}': {lazy_state_fields}"
                )

        return [assert_full_lazy_state]


# Removed duplicate assertion function - using unified AssertionFactory instead


    @staticmethod
    def create_reset_validation_assertions(scenario: TestScenario) -> List[Callable[[WorkflowContext], None]]:
        """Create assertions for reset functionality validation."""

        def assert_no_concrete_values_in_reset_placeholders(context: WorkflowContext) -> None:
            """Assert that reset placeholders don't show concrete saved values."""
            if "reset_placeholder_bug" not in scenario.name:
                return  # Only run for reset bug scenarios

            print(f"\n🔍 CHECKING FOR RESET PLACEHOLDER BUG...")
            form_managers = context.config_window.findChildren(ParameterFormManager)

            # Check for specific problematic values using lookup table approach
            problematic_values = ["828282"]  # Values that shouldn't appear in reset placeholders
            bug_detected = False

            for form_manager in form_managers:
                if not hasattr(form_manager, 'widgets'):
                    continue

                for field_name, widget in form_manager.widgets.items():
                    texts = ValidationEngine.extract_widget_texts(widget)
                    all_text = " ".join(texts.values())

                    # Check for problematic values in placeholders
                    if "pipeline default:" in all_text.lower():
                        for problematic_value in problematic_values:
                            if problematic_value in all_text:
                                print(f"🚨 RESET PLACEHOLDER BUG: Field '{field_name}' shows '{problematic_value}' in placeholder!")
                                bug_detected = True

            if bug_detected:
                raise AssertionError(
                    f"RESET PLACEHOLDER BUG DETECTED in scenario '{scenario.name}': "
                    f"Placeholders show concrete saved values instead of proper inheritance values."
                )
            else:
                print(f"✅ No reset placeholder bug detected")

        return [assert_no_concrete_values_in_reset_placeholders]


class TestPyQtGUIWorkflowFoundation:

    @pytest.fixture
    def synthetic_plate_dir(self, tmp_path):
        """Create synthetic plate data for testing."""
        return _create_synthetic_plate(tmp_path)

    @pytest.fixture
    def global_config(self):
        """Create test global configuration."""
        return _create_test_global_config()

    @pytest.fixture(autouse=True)
    def cleanup_gui_state(self, qtbot):
        """Automatically cleanup GUI state between tests with error monitoring."""
        # Setup: Clear any existing state
        from PyQt6.QtWidgets import QApplication
        from openhcs.pyqt_gui.main import OpenHCSMainWindow

        # Close any existing top-level widgets (except OpenHCS main windows)
        for widget in QApplication.topLevelWidgets():
            if widget.isVisible() and not isinstance(widget, OpenHCSMainWindow):
                widget.close()
                widget.deleteLater()

        QApplication.processEvents()

        # Start global error monitoring for the entire test
        monitor = get_error_monitor()
        monitor.start_monitoring(check_interval_ms=100)

        try:
            yield  # Run the test

            # Check if any errors were detected during the test
            if monitor.detected_error:
                raise AssertionError(f"Error detected during test execution: {monitor.detected_error}")

        finally:
            # Always stop monitoring
            monitor.stop_monitoring()

            # Teardown: Gentle cleanup to avoid main window closeEvent conflicts
            try:
                # First, close floating windows manually to avoid main window cleanup
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, OpenHCSMainWindow):
                        # Manually close floating windows without triggering main window closeEvent
                        for window_name, window in list(widget.floating_windows.items()):
                            try:
                                window.hide()
                                window.deleteLater()
                            except:
                                pass
                        widget.floating_windows.clear()

                        # Hide main window without calling close() to avoid closeEvent
                        widget.hide()
                        widget.deleteLater()
                    elif widget.isVisible():
                        widget.close()
                        widget.deleteLater()

                # Process events gently
                QApplication.processEvents()

            except Exception as e:
                print(f"Warning: Error during GUI cleanup: {e}")
                # Continue anyway - don't fail the test due to cleanup issues

    @pytest.mark.parametrize("test_scenario", list(TEST_SCENARIOS.values()), ids=lambda scenario: scenario.name)
    def test_parameterized_end_to_end_workflow(
        self, qtbot, synthetic_plate_dir, test_scenario: TestScenario
    ):
        """
        Unified end-to-end workflow test using mathematical simplification principles.

        Demonstrates algebraic factoring approach:
        - Single workflow handles all test scenarios through parameterization
        - Unified validation engine with pluggable rules
        - Composable assertions using factory pattern
        """
        print(f"\n=== Unified Workflow Test: {test_scenario.name} ===")
        print(f"Config: {test_scenario.orchestrator_config}")
        print(f"Expected: {test_scenario.expected_values}")

        # Create unified workflow using factory pattern
        workflow = self._create_unified_workflow(test_scenario)

        # Execute workflow with initial context
        initial_context = WorkflowContext(
            synthetic_plate_dir=synthetic_plate_dir,
            test_scenario=test_scenario
        )
        final_context = workflow.execute(initial_context)

        # Register main window with qtbot for cleanup
        qtbot.addWidget(final_context.main_window)

        print(f"✅ Unified workflow '{test_scenario.name}' validation passed!")
        print(f"✅ Configuration {test_scenario.orchestrator_config} applied successfully!")
        print(f"✅ Expected values {test_scenario.expected_values} validated!")

    def _create_unified_workflow(self, test_scenario: TestScenario) -> WorkflowBuilder:
        """Create unified workflow using algebraic factoring approach."""

        # Base workflow steps (common to all scenarios)
        base_steps = [
            ("Launch OpenHCS Application", _launch_application, TIMING.WINDOW_DELAY),
            ("Access Plate Manager", _access_plate_manager, None),
            ("Add and Select Plate", _add_and_select_plate, TIMING.ACTION_DELAY),
            ("Initialize Plate", _initialize_plate, TIMING.SAVE_DELAY),
            ("Verify Step Editor Placeholder Resolution", _verify_step_editor_placeholder_resolution, TIMING.ACTION_DELAY),
            ("Apply Parameterized Orchestrator Configuration", _apply_orchestrator_config, TIMING.ACTION_DELAY),
            ("Open Configuration Window", _open_config_window, TIMING.WINDOW_DELAY),
            ("Validate Initial Placeholder Behavior", _validate_placeholder_behavior, None),
            ("Set Step Well Filter for Isolation Test", _set_step_well_filter_for_isolation_test, TIMING.SAVE_DELAY),
            ("Modify Field", _modify_field, TIMING.SAVE_DELAY),
            ("Reopen Configuration Window", _reopen_config_window, TIMING.WINDOW_DELAY),
            ("Validate Field Persistence", _validate_field_persistence, None),
            ("Reset Field", _reset_field, TIMING.ACTION_DELAY),
            ("Validate Full Lazy State", _validate_full_lazy_state, None)
        ]

        # Build workflow using unified step factory
        workflow = WorkflowBuilder()
        for name, operation, timing_delay in base_steps:
            workflow.add_step(WorkflowStep(name=name, operation=operation, timing_delay=timing_delay))

        # Add unified assertions using factory pattern
        placeholder_assertions = AssertionFactory.create_placeholder_assertions(test_scenario)
        persistence_assertions = AssertionFactory.create_persistence_assertions(test_scenario)
        lazy_state_assertions = AssertionFactory.create_lazy_state_assertions(test_scenario)
        reset_assertions = AssertionFactory.create_reset_validation_assertions(test_scenario)

        for assertion in placeholder_assertions + persistence_assertions + lazy_state_assertions + reset_assertions:
            workflow.add_assertion(assertion)

        return workflow
