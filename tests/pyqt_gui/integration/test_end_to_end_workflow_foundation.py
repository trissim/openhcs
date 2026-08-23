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
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
from zmqruntime import EndpointShutdownMode
from zmqruntime.shutdown import EndpointShutdownService
from zmqruntime.transport import DataControlPortPairAuthority

# Skip entire module in CPU-only mode to avoid PyQt6 imports
if os.getenv("OPENHCS_CPU_ONLY", "false").lower() == "true":
    pytest.skip("PyQt6 GUI tests skipped in CPU-only mode", allow_module_level=True)

from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
)
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager

from openhcs.constants import Microscope
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyStepMaterializationConfig,
    PipelineConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.demo.synthetic_data import SyntheticMicroscopyGenerator
from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, get_default_ui_config
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.windows.config_window import ConfigWindow

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
    def from_environment(cls) -> "TimingConfig":
        """Create timing config from environment variables."""
        return cls(
            ACTION_DELAY=float(
                os.environ.get("OPENHCS_TEST_ACTION_DELAY", cls.ACTION_DELAY)
            ),
            WINDOW_DELAY=float(
                os.environ.get("OPENHCS_TEST_WINDOW_DELAY", cls.WINDOW_DELAY)
            ),
            SAVE_DELAY=float(os.environ.get("OPENHCS_TEST_SAVE_DELAY", cls.SAVE_DELAY)),
        )


@dataclass(frozen=True)
class FieldModificationSpec:
    """Specification for field modification testing."""

    field_path: tuple[str, ...]
    modification_value: Any


@dataclass(frozen=True)
class InheritanceExpectation:
    """One declared source path and the lazy fields that must resolve from it."""

    source_path: tuple[str, ...]
    target_paths: tuple[tuple[str, ...], ...]
    excluded_source_paths: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TestScenario:
    """Complete test scenario configuration."""

    name: str
    pipeline_config: PipelineConfig
    field_to_test: FieldModificationSpec
    reset_field_to_test: Optional[tuple[str, ...]] = None
    setup_modifications: tuple[FieldModificationSpec, ...] = ()
    inheritance_expectations: tuple[InheritanceExpectation, ...] = ()


TIMING = TimingConfig.from_environment()

# ============================================================================
# DATA-DRIVEN TEST SCENARIO FACTORY
# ============================================================================


def create_test_scenarios() -> Dict[str, TestScenario]:
    """Declare GUI scenarios with the same typed configs used by production."""

    def materialization_config(
        *, output_dir_suffix: str, sub_dir: str, well_filter: Any, num_workers=None
    ) -> PipelineConfig:
        return PipelineConfig(
            num_workers=num_workers,
            step_materialization_config=LazyStepMaterializationConfig(
                output_dir_suffix=output_dir_suffix,
                sub_dir=sub_dir,
                well_filter=well_filter,
            ),
        )

    scenarios = (
        TestScenario(
            name="reset_placeholder_bug_path_planning",
            pipeline_config=materialization_config(
                output_dir_suffix="828282",
                sub_dir="images",
                well_filter=5,
                num_workers=1,
            ),
            field_to_test=FieldModificationSpec(
                ("path_planning_config", "output_dir_suffix"), "828282"
            ),
        ),
        TestScenario(
            name="reset_placeholder_bug_materialization",
            pipeline_config=materialization_config(
                output_dir_suffix="828282",
                sub_dir="images",
                well_filter=5,
                num_workers=1,
            ),
            field_to_test=FieldModificationSpec(
                ("step_materialization_config", "output_dir_suffix"), "828282"
            ),
        ),
        TestScenario(
            name="reset_placeholder_bug_direct_field",
            pipeline_config=materialization_config(
                output_dir_suffix="_test",
                sub_dir="images",
                well_filter=["A01", "A02"],
            ),
            field_to_test=FieldModificationSpec(("num_workers",), 2),
        ),
        TestScenario(
            name="default_hierarchy",
            pipeline_config=materialization_config(
                output_dir_suffix="_outputs", sub_dir="images", well_filter=5
            ),
            field_to_test=FieldModificationSpec(
                ("step_materialization_config", "well_filter"), 4
            ),
        ),
        TestScenario(
            name="alternative_config",
            pipeline_config=materialization_config(
                output_dir_suffix="_processed", sub_dir="results", well_filter=10
            ),
            field_to_test=FieldModificationSpec(
                ("path_planning_config", "output_dir_suffix"), "_custom"
            ),
        ),
        TestScenario(
            name="minimal_config",
            pipeline_config=materialization_config(
                output_dir_suffix="", sub_dir="data", well_filter=1
            ),
            field_to_test=FieldModificationSpec(
                ("path_planning_config", "sub_dir"), "test_data"
            ),
        ),
        TestScenario(
            name="inheritance_hierarchy_step_well_filter",
            pipeline_config=PipelineConfig(
                num_workers=1,
                step_materialization_config=LazyStepMaterializationConfig(
                    output_dir_suffix="_test",
                    sub_dir="images",
                ),
            ),
            field_to_test=FieldModificationSpec(
                ("step_well_filter_config", "well_filter"), 42
            ),
            inheritance_expectations=(
                InheritanceExpectation(
                    source_path=("step_well_filter_config", "well_filter"),
                    target_paths=(
                        ("step_materialization_config", "well_filter"),
                        ("napari_streaming_config", "well_filter"),
                        ("fiji_streaming_config", "well_filter"),
                    ),
                ),
            ),
        ),
        TestScenario(
            name="inheritance_hierarchy_path_planning_isolation",
            pipeline_config=PipelineConfig(
                num_workers=1,
                path_planning_config=LazyPathPlanningConfig(
                    output_dir_suffix="_test",
                    sub_dir="images",
                ),
            ),
            setup_modifications=(
                FieldModificationSpec(("step_well_filter_config", "well_filter"), 5),
            ),
            field_to_test=FieldModificationSpec(
                ("path_planning_config", "well_filter"), 99
            ),
            inheritance_expectations=(
                InheritanceExpectation(
                    source_path=("step_well_filter_config", "well_filter"),
                    target_paths=(
                        ("step_materialization_config", "well_filter"),
                        ("napari_streaming_config", "well_filter"),
                        ("fiji_streaming_config", "well_filter"),
                    ),
                    excluded_source_paths=(("path_planning_config", "well_filter"),),
                ),
            ),
        ),
    )
    return {scenario.name: scenario for scenario in scenarios}


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
    test_scenario: Optional[TestScenario] = None

    def with_updates(self, **kwargs) -> "WorkflowContext":
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

    def add_step(self, step: WorkflowStep) -> "WorkflowBuilder":
        """Add workflow step (fluent interface)."""
        self.steps.append(step)
        return self

    def execute(self, initial_context: WorkflowContext) -> WorkflowContext:
        """Execute workflow steps sequentially."""
        context = initial_context
        for step in self.steps:
            context = step.execute(context)

        return context


# ============================================================================
# UNIFIED ERROR HANDLING SYSTEM
# ============================================================================

ERROR_KEYWORDS = ["error", "exception", "recursion", "warning", "unexpected"]


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
                if hasattr(label, "text"):
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
                except Exception:
                    pass
        return error_details

    def _extract_error_text(self, dialog) -> str:
        """Extract error text from dialog."""
        if isinstance(dialog, QMessageBox):
            return dialog.text()[:200]

        for label in dialog.findChildren(QLabel):
            if hasattr(label, "text"):
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
                error_details = monitor._close_error_dialogs(
                    monitor._find_error_dialogs()
                )
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
        grid_size=(2, 2),
        tile_size=(64, 64),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=1,
        wells=["A01"],
        format="ImageXpress",
    )
    generator.generate_dataset()
    return plate_dir


def _create_test_global_config() -> GlobalPipelineConfig:
    """Create test global configuration with known values."""
    from openhcs.core.config import PathPlanningConfig, WellFilterConfig

    return GlobalPipelineConfig(
        num_workers=8,
        microscope=Microscope.IMAGEXPRESS,
        use_threading=True,
        # Add well_filter values that test scenarios expect to inherit
        well_filter_config=WellFilterConfig(well_filter=5),
        path_planning_config=PathPlanningConfig(
            well_filter=5, output_dir_suffix="_test_global", sub_dir="images"
        ),
    )


# ============================================================================
# UNIFIED DECORATORS AND WIDGET UTILITIES
# ============================================================================


def with_timeout_and_error_handling(
    timeout_seconds: int = 10, operation_name: str = "operation"
):
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
                print(
                    f"  {operation_name.title()} completed successfully in {elapsed:.2f}s"
                )
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


def find_widget_with_retry(
    widget_finder: Callable, timeout_seconds: int = 10, check_interval: float = 0.5
):
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
            "visible_dialogs": len(
                [
                    w
                    for w in QApplication.topLevelWidgets()
                    if isinstance(w, QDialog) and w.isVisible()
                ]
            ),
            "total_widgets": len(QApplication.topLevelWidgets()),
            "top_level_widgets": [
                f"{type(w).__name__}: {w.windowTitle()}"
                for w in QApplication.topLevelWidgets()
                if w.isVisible()
            ],
        }
    except Exception:
        return {"error": "Could not collect diagnostic info"}


# ============================================================================
# WORKFLOW STEP OPERATIONS
# ============================================================================


def _launch_application(context: WorkflowContext) -> WorkflowContext:
    """Launch real OpenHCS application using normal startup process."""
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    existing_application = QApplication.instance()
    if isinstance(existing_application, OpenHCSPyQtApp):
        main_window = existing_application.create_main_window()
        if not main_window.isVisible():
            existing_application.show_main_window()
            _wait_for_gui(TIMING.WINDOW_DELAY)
        return context.with_updates(main_window=main_window)

    raise AssertionError(
        "The OpenHCS application lifecycle fixture did not establish the real "
        "application before the workflow started."
    )


def _access_plate_manager(context: WorkflowContext) -> WorkflowContext:
    """Access the declaration-owned Plate Manager dock pane."""
    plate_manager_widget = context.main_window.embedded_widgets.require_plate_manager()
    if not isinstance(plate_manager_widget, PlateManagerWidget):
        raise AssertionError("The Plate Manager pane does not own PlateManagerWidget")

    return context.with_updates(plate_manager_widget=plate_manager_widget)


def _add_and_select_plate(context: WorkflowContext) -> WorkflowContext:
    """Add synthetic plate and select it in plate manager."""
    context.plate_manager_widget.add_plate_callback([context.synthetic_plate_dir])
    _wait_for_gui(TIMING.ACTION_DELAY)

    item_list = context.plate_manager_widget.item_list
    if item_list.count() == 0:
        raise AssertionError(
            "No plates found in plate manager list after adding synthetic plate"
        )

    item_list.setCurrentRow(item_list.count() - 1)
    _wait_for_gui(TIMING.ACTION_DELAY)
    return context


def _initialize_plate(context: WorkflowContext) -> WorkflowContext:
    """Initialize plate using Init button."""
    init_button = context.plate_manager_widget.buttons["init_plate"]
    if not init_button.isEnabled():
        raise AssertionError(
            "Init button is disabled - plate may not be properly added"
        )

    init_button.click()
    _wait_for_gui(TIMING.SAVE_DELAY)
    return context


def _apply_orchestrator_config(context: WorkflowContext) -> WorkflowContext:
    """Apply parameterized orchestrator configuration to establish 3-level hierarchy."""
    if not context.test_scenario:
        raise ValueError(
            "Test scenario must be provided for parameterized orchestrator configuration"
        )

    orchestrator = context.plate_manager_widget.get_selected_orchestrator()
    if orchestrator is None:
        raise AssertionError("The selected plate has no initialized orchestrator")

    orchestrator.apply_pipeline_config(context.test_scenario.pipeline_config)
    _wait_for_gui(TIMING.ACTION_DELAY)

    return context.with_updates(orchestrator=orchestrator)


def _find_config_window(scope_id: str) -> Optional[ConfigWindow]:
    """Find the visible configuration window owned by one ObjectState scope."""
    for widget in QApplication.topLevelWidgets():
        if (
            isinstance(widget, ConfigWindow)
            and widget.scope_id == scope_id
            and widget.isVisible()
        ):
            return widget
    return None


def _wait_for_form_build(
    form_manager: ParameterFormManager,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not form_manager.form_build_complete:
        failure = form_manager.form_build_failure
        if failure is not None:
            raise AssertionError("Configuration form construction failed.") from failure
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Configuration form did not finish building within {timeout_seconds}s."
            )
        QApplication.processEvents()
        QTest.qWait(10)


@with_timeout_and_error_handling(
    timeout_seconds=10, operation_name="opening configuration window"
)
def _open_config_window(context: WorkflowContext) -> WorkflowContext:
    """Open plate-specific configuration window."""
    edit_button = context.plate_manager_widget.buttons["edit_config"]
    if not edit_button.isEnabled():
        raise AssertionError(
            "Edit button is disabled - plate may not be properly initialized"
        )

    edit_button.click()
    _wait_for_gui(TIMING.WINDOW_DELAY)

    scope_id = context.plate_manager_widget.selected_plate_path
    if scope_id is None:
        raise AssertionError("Plate Manager has no selected configuration scope.")
    config_window = find_widget_with_retry(
        lambda: _find_config_window(scope_id),
        timeout_seconds=10,
    )
    if not config_window:
        diagnostics = collect_diagnostic_info()
        raise AssertionError(
            f"Configuration window not found. Diagnostics: {diagnostics}"
        )

    _wait_for_form_build(config_window.form_manager)
    _wait_for_gui(TIMING.ACTION_DELAY)
    return context.with_updates(config_window=config_window)


# ============================================================================
# UNIFIED WIDGET INTERACTION SYSTEM
# ============================================================================


class WidgetFinder:
    """Resolve widgets through the form manager's declared field hierarchy."""

    @staticmethod
    def form_manager(
        root: ParameterFormManager,
        field_path: tuple[str, ...],
    ) -> ParameterFormManager:
        if not field_path:
            raise ValueError("A form field path cannot be empty.")

        manager = root
        traversed: list[str] = []
        for field_name in field_path[:-1]:
            traversed.append(field_name)
            try:
                manager = manager.nested_managers[field_name]
            except KeyError as error:
                prefix = ".".join(traversed)
                raise AssertionError(
                    f"Nested form manager '{prefix}' is not materialized. "
                    f"Available children: {tuple(manager.nested_managers)}"
                ) from error
        return manager

    @classmethod
    def field_widget(
        cls,
        root: ParameterFormManager,
        field_path: tuple[str, ...],
    ) -> Any:
        manager = cls.form_manager(root, field_path)
        field_name = field_path[-1]
        try:
            return manager.widgets[field_name]
        except KeyError as error:
            raise AssertionError(
                f"Form field '{'.'.join(field_path)}' is not materialized. "
                f"Available fields: {tuple(manager.widgets)}"
            ) from error

    @staticmethod
    def find_button_by_text(
        parent_widget, button_texts: List[str]
    ) -> Optional[QPushButton]:
        """Find button by text using lookup table approach."""
        button_texts_lower = [text.lower() for text in button_texts]
        for button in parent_widget.findChildren(QPushButton):
            if button.text().lower() in button_texts_lower:
                return button
        return None


class WidgetInteractor:
    """Unified widget interaction system."""

    @staticmethod
    def set_widget_value(widget: Any, value: Any) -> None:
        """Set a value through the form system's widget protocol."""
        from pyqt_reactive.forms.widget_operations import WidgetOperations

        WidgetOperations.set_value(widget, value)


@with_timeout_and_error_handling(timeout_seconds=5, operation_name="modifying field")
def _modify_field(context: WorkflowContext) -> WorkflowContext:
    """Modify specified field in the configuration window and save."""
    if not context.test_scenario:
        raise ValueError("Test scenario required for parameterized field modification")

    return _modify_field_from_spec(context, context.test_scenario.field_to_test)


def _modify_field_from_spec(
    context: WorkflowContext, field_spec: FieldModificationSpec
) -> WorkflowContext:
    """Apply and save one concrete config-field edit through its canonical path."""
    field_path = field_spec.field_path
    field_name = field_path[-1]
    field_value = field_spec.modification_value
    field_widget = WidgetFinder.field_widget(
        context.config_window.form_manager,
        field_path,
    )

    print(f"🔧 MODIFY FIELD: Targeting {'.'.join(field_path)} = {field_value}")

    print(f"  Setting {field_name} = {field_value}")
    WidgetInteractor.set_widget_value(field_widget, field_value)
    _wait_for_gui(TIMING.ACTION_DELAY)

    # Save the configuration (inlined single-use function)
    save_button = WidgetFinder.find_button_by_text(
        context.config_window, ["ok", "save", "apply"]
    )
    if not save_button:
        buttons = [b.text() for b in context.config_window.findChildren(QPushButton)]
        raise AssertionError(f"Save button not found. Available buttons: {buttons}")

    save_button.click()
    _wait_for_gui(TIMING.SAVE_DELAY)
    return context


def _apply_setup_modifications(context: WorkflowContext) -> WorkflowContext:
    """Apply scenario-declared prerequisite edits without scenario-name dispatch."""
    if context.test_scenario is None:
        raise ValueError("Test scenario required for setup modifications")

    for field_spec in context.test_scenario.setup_modifications:
        context = _modify_field_from_spec(context, field_spec)
        context = _reopen_config_window(context)
    return context


def _close_config_window(context: WorkflowContext) -> WorkflowContext:
    """Close configuration window with cleanup."""
    try:
        if context.config_window and context.config_window.isVisible():
            context.config_window.close()
            context.config_window.deleteLater()
            _wait_for_gui(TIMING.ACTION_DELAY)

        # Clean up any remaining config windows
        for widget in QApplication.topLevelWidgets():
            if (
                isinstance(widget, QDialog)
                and "config" in widget.windowTitle().lower()
                and widget.isVisible()
            ):
                widget.close()
                widget.deleteLater()

        _wait_for_gui(TIMING.ACTION_DELAY)
        return context.with_updates(config_window=None)

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
        return context.with_updates(config_window=None)


@with_timeout_and_error_handling(
    timeout_seconds=10, operation_name="reopening configuration window"
)
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
    field_path = (
        context.test_scenario.reset_field_to_test
        or context.test_scenario.field_to_test.field_path
    )
    field_name = field_path[-1]
    target_form_manager = WidgetFinder.form_manager(
        context.config_window.form_manager,
        field_path,
    )

    print(f"  DEBUG: Looking for reset button for field '{field_name}'")

    # Inline the reset button finding logic (single-use helper elimination)
    try:
        reset_button = target_form_manager.reset_buttons[field_name]
    except KeyError as error:
        raise AssertionError(
            f"Reset button for '{'.'.join(field_path)}' is not materialized. "
            f"Available reset fields: {tuple(target_form_manager.reset_buttons)}"
        ) from error

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
STEP_EDITOR_TEST_FIELDS = ["output_dir_suffix", "sub_dir"]


class StepEditorValidator:
    """Unified step editor validation using mathematical simplification principles."""

    @staticmethod
    def find_step_editor_window() -> Optional[Any]:
        """Find the visible declaration-owned step editor window."""
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, DualEditorWindow) and widget.isVisible():
                return widget
        return None

    @staticmethod
    def validate_placeholder_resolution(
        nested_manager: ParameterFormManager,
        test_fields: List[str],
    ) -> bool:
        """Validate placeholders against their ObjectState-resolved values."""
        from pyqt_reactive.protocols import PlaceholderStateTrackable

        placeholder_resolution_verified = False

        for field_name in test_fields:
            widget = nested_manager.widgets[field_name]
            dotted_path = f"{nested_manager.field_id}.{field_name}"
            if not isinstance(widget, PlaceholderStateTrackable):
                raise AssertionError(
                    f"Step field '{dotted_path}' does not expose placeholder state."
                )
            if not widget.has_placeholder_state():
                raise AssertionError(
                    f"Step field '{dotted_path}' is concrete before any step override."
                )

            resolved_value = nested_manager.state.get_resolved_value(dotted_path)
            all_text = " ".join(_extract_widget_texts(widget).values())
            if resolved_value is None or str(resolved_value) not in all_text:
                raise AssertionError(
                    f"Step field '{dotted_path}' presents '{all_text}' instead of "
                    f"its resolved value {resolved_value!r}."
                )
            placeholder_resolution_verified = True

        return placeholder_resolution_verified


def _verify_step_editor_placeholder_resolution(
    context: WorkflowContext,
) -> WorkflowContext:
    """Verify step editor placeholders using unified validation system."""
    print("\n🔍 Verifying step editor placeholder resolution after initialization...")

    pipeline_editor = context.main_window.embedded_widgets.require_pipeline_editor()

    # Open step editor using pipeline editor's button dictionary
    if (
        not hasattr(pipeline_editor, "buttons")
        or "add_step" not in pipeline_editor.buttons
    ):
        raise AssertionError("Add Step button not found in pipeline editor buttons")
    add_step_button = pipeline_editor.buttons["add_step"]
    ready_button = find_widget_with_retry(
        lambda: add_step_button if add_step_button.isEnabled() else None,
        timeout_seconds=10,
        check_interval=0.05,
    )
    if ready_button is None:
        raise AssertionError(
            "Add Step is disabled after the selected plate reports initialization."
        )

    ready_button.click()
    QApplication.processEvents()

    step_editor_window = find_widget_with_retry(
        StepEditorValidator.find_step_editor_window,
        timeout_seconds=10,
        check_interval=0.05,
    )
    if not step_editor_window:
        raise AssertionError("Step editor window (DualEditorWindow) not found")

    try:
        # Access form manager using unified approach
        step_param_editor = step_editor_window.step_editor
        if not step_param_editor or not hasattr(step_param_editor, "form_manager"):
            raise AssertionError("Form manager not found in step parameter editor")

        form_manager = step_param_editor.form_manager
        _wait_for_form_build(form_manager)

        nested_manager = form_manager.nested_managers["step_materialization_config"]
        placeholder_resolution_verified = (
            StepEditorValidator.validate_placeholder_resolution(
                nested_manager,
                STEP_EDITOR_TEST_FIELDS,
            )
        )

        if not placeholder_resolution_verified:
            raise AssertionError(
                "Step materialization fields did not expose resolved placeholders."
            )

    finally:
        # Clean up step editor window
        step_editor_window.close()
        step_editor_window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    return context


def _extract_widget_texts(widget) -> Dict[str, str]:
    """Read presentation text through capabilities exposed by a concrete widget."""

    path_input = getattr(widget, "path_input", None)
    placeholder_owner = path_input or widget
    text_owner = path_input or widget
    text_extractors = (
        ("placeholder", getattr(placeholder_owner, "placeholderText", lambda: "")),
        ("special", getattr(widget, "specialValueText", lambda: "")),
        ("tooltip", getattr(widget, "toolTip", lambda: "")),
        ("text", getattr(text_owner, "text", lambda: "")),
    )
    return {name: extractor() or "" for name, extractor in text_extractors}


def _form_managers(root_manager: ParameterFormManager):
    """Yield one form tree from its declaration-owned nested-manager graph."""
    managers = [root_manager]
    for manager in managers:
        yield manager
        managers.extend(manager.nested_managers.values())


def _validate_placeholder_behavior(context: WorkflowContext) -> WorkflowContext:
    """Reject rendered None placeholders whose authoritative value is resolved."""
    invalid_fields = []
    for manager in _form_managers(context.config_window.form_manager):
        for field_name, widget in manager.widgets.items():
            dotted_path = ".".join(
                part for part in (manager.field_id, field_name) if part
            )
            resolved_value = manager.state.get_resolved_value(dotted_path)
            presented_text = " ".join(_extract_widget_texts(widget).values()).lower()
            if resolved_value is not None and "(none)" in presented_text:
                invalid_fields.append(dotted_path)

    if invalid_fields:
        raise AssertionError(
            "Resolved fields rendered '(none)' placeholders: "
            f"{tuple(invalid_fields)}"
        )
    return context


def _validate_field_persistence(context: WorkflowContext) -> WorkflowContext:
    """Validate authoritative state and concrete widget value after reopening."""
    scenario = context.test_scenario
    if scenario is None:
        raise AssertionError("Persistence validation requires a test scenario.")

    from pyqt_reactive.forms.widget_operations import WidgetOperations
    from pyqt_reactive.protocols import PlaceholderStateTrackable

    field_path = scenario.field_to_test.field_path
    dotted_path = ".".join(field_path)
    expected_value = scenario.field_to_test.modification_value
    root_manager = context.config_window.form_manager
    target_manager = WidgetFinder.form_manager(root_manager, field_path)
    resolved_value = target_manager.state.get_resolved_value(dotted_path)
    if str(resolved_value) != str(expected_value):
        raise AssertionError(
            f"Saved field '{dotted_path}' resolved to {resolved_value!r}, "
            f"not {expected_value!r}."
        )

    widget = WidgetFinder.field_widget(root_manager, field_path)
    if isinstance(widget, PlaceholderStateTrackable) and widget.has_placeholder_state():
        raise AssertionError(f"Saved field '{dotted_path}' remained a placeholder.")
    presented_value = WidgetOperations.get_value(widget)
    if str(presented_value) != str(expected_value):
        raise AssertionError(
            f"Saved field '{dotted_path}' presents {presented_value!r}, "
            f"not {expected_value!r}."
        )

    return context


def _validate_inheritance_relationships(
    context: WorkflowContext,
) -> WorkflowContext:
    """Validate scenario-declared lazy inheritance through canonical field paths."""
    scenario = context.test_scenario
    if scenario is None or not scenario.inheritance_expectations:
        return context

    from pyqt_reactive.protocols import PlaceholderStateTrackable

    root_manager = context.config_window.form_manager
    for expectation in scenario.inheritance_expectations:
        source_manager = WidgetFinder.form_manager(
            root_manager,
            expectation.source_path,
        )
        source_path = ".".join(expectation.source_path)
        source_value = source_manager.state.get_resolved_value(source_path)

        excluded_values = {
            ".".join(path): WidgetFinder.form_manager(
                root_manager,
                path,
            ).state.get_resolved_value(".".join(path))
            for path in expectation.excluded_source_paths
        }

        for target_path_parts in expectation.target_paths:
            target_path = ".".join(target_path_parts)
            target_manager = WidgetFinder.form_manager(
                root_manager,
                target_path_parts,
            )
            target_value = target_manager.state.get_resolved_value(target_path)
            if target_value != source_value:
                raise AssertionError(
                    f"'{target_path}' resolved to {target_value!r}, not its declared "
                    f"test source '{source_path}' value {source_value!r}."
                )

            conflicting_sources = [
                path for path, value in excluded_values.items() if value == target_value
            ]
            if conflicting_sources:
                raise AssertionError(
                    f"'{target_path}' resolved from excluded test source(s) "
                    f"{conflicting_sources}."
                )

            widget = WidgetFinder.field_widget(root_manager, target_path_parts)
            if not isinstance(widget, PlaceholderStateTrackable):
                raise AssertionError(
                    f"Inherited field '{target_path}' does not expose placeholder state."
                )
            if not widget.has_placeholder_state():
                raise AssertionError(
                    f"Inherited field '{target_path}' became concrete in the UI."
                )
            presented_text = " ".join(_extract_widget_texts(widget).values()).lower()
            if str(source_value).lower() not in presented_text:
                raise AssertionError(
                    f"Inherited field '{target_path}' presents {presented_text!r}, "
                    f"not resolved value {source_value!r}."
                )

    return context


def _validate_full_lazy_state(context: WorkflowContext) -> WorkflowContext:
    """Validate reset through authoritative state and placeholder capability."""
    scenario = context.test_scenario
    if scenario is None:
        raise AssertionError("Reset validation requires a test scenario.")

    from pyqt_reactive.protocols import PlaceholderStateTrackable

    field_path = scenario.reset_field_to_test or scenario.field_to_test.field_path
    dotted_path = ".".join(field_path)
    root_manager = context.config_window.form_manager
    target_manager = WidgetFinder.form_manager(root_manager, field_path)
    widget = WidgetFinder.field_widget(root_manager, field_path)
    if target_manager.state.parameters[dotted_path] is not None:
        raise AssertionError(f"Reset did not clear concrete state for '{dotted_path}'.")
    if not isinstance(widget, PlaceholderStateTrackable):
        raise AssertionError(
            f"Reset target '{dotted_path}' does not expose placeholder state."
        )
    if not widget.has_placeholder_state():
        raise AssertionError(
            f"Reset target '{dotted_path}' remained concrete in the UI."
        )

    resolved_value = target_manager.state.get_resolved_value(dotted_path)
    modified_value = scenario.field_to_test.modification_value
    displayed_text = " ".join(_extract_widget_texts(widget).values())
    if resolved_value != modified_value and str(modified_value) in displayed_text:
        raise AssertionError(
            f"Reset target '{dotted_path}' still presents stale value "
            f"{modified_value!r}; resolved value is {resolved_value!r}."
        )
    return context


class TestPyQtGUIWorkflowFoundation:

    @pytest.fixture(scope="class", autouse=True)
    def application_lifecycle(self):
        """Own one real OpenHCS application for the complete workflow matrix."""
        import sys

        from objectstate.global_config import get_current_global_config

        from openhcs.pyqt_gui.app import OpenHCSPyQtApp

        existing_application = QApplication.instance()
        if existing_application is not None:
            raise AssertionError(
                "The workflow suite requires ownership of the QApplication lifecycle."
            )

        base_ui_config = get_default_ui_config()
        candidate_zmq_config = replace(
            base_ui_config.zmq,
            default_port=20_000 + os.getpid() % 20_000,
        )
        endpoint_pair = DataControlPortPairAuthority.acquire(
            candidate_zmq_config,
            transport_mode=candidate_zmq_config.transport_mode,
        )
        test_zmq_config = replace(
            candidate_zmq_config,
            default_port=endpoint_pair.data_port,
        )
        app = OpenHCSPyQtApp(
            sys.argv,
            runtime_context=PyQtGuiRuntimeContext(
                replace(base_ui_config, zmq=test_zmq_config),
                pipeline_runtime=_create_test_global_config(),
            ),
        )
        if get_current_global_config(GlobalPipelineConfig) is None:
            raise AssertionError("OpenHCS did not establish its global config context.")

        app.create_main_window()
        app.show_main_window()
        _wait_for_gui(TIMING.WINDOW_DELAY)

        try:
            yield app
        finally:
            app.cleanup()
            QApplication.processEvents()
            shutdown = EndpointShutdownService.for_config(
                test_zmq_config
            ).shutdown_ports(
                ports=[endpoint_pair.data_port],
                mode=EndpointShutdownMode.FORCE,
            )
            if not shutdown.succeeded or shutdown.terminated_ports != (
                endpoint_pair.data_port,
            ):
                raise AssertionError(shutdown.failure_message)

    @pytest.fixture
    def synthetic_plate_dir(self, tmp_path):
        """Create synthetic plate data for testing."""
        return _create_synthetic_plate(tmp_path)

    @pytest.fixture
    def global_config(self):
        """Create test global configuration."""
        return _create_test_global_config()

    @pytest.fixture(autouse=True)
    def cleanup_gui_state(self, application_lifecycle):
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
                raise AssertionError(
                    f"Error detected during test execution: {monitor.detected_error}"
                )

        finally:
            # Always stop monitoring
            monitor.stop_monitoring()

            # Teardown: Gentle cleanup to avoid main window closeEvent conflicts
            try:
                # First, close floating windows manually to avoid main window cleanup
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, OpenHCSMainWindow):
                        # Manually close floating windows without triggering main window closeEvent
                        for window_name, window in list(
                            widget.floating_windows.items()
                        ):
                            try:
                                window.hide()
                                window.deleteLater()
                            except Exception:
                                pass
                        widget.floating_windows.clear()

                        plate_manager = widget.embedded_widgets.require_plate_manager()
                        plates = list(plate_manager.plates)
                        if plates:
                            plate_manager.perform_delete(plates)
                            plate_manager.update_item_list()
                    elif widget.isVisible():
                        widget.close()
                        widget.deleteLater()

                # Process events gently
                QApplication.processEvents()

            except Exception as e:
                print(f"Warning: Error during GUI cleanup: {e}")
                # Continue anyway - don't fail the test due to cleanup issues

    @pytest.mark.parametrize(
        "test_scenario",
        list(TEST_SCENARIOS.values()),
        ids=lambda scenario: scenario.name,
    )
    def test_parameterized_end_to_end_workflow(
        self, synthetic_plate_dir, test_scenario: TestScenario
    ):
        """Exercise config inheritance, editing, persistence, and reset end to end."""
        print(f"\n=== Unified Workflow Test: {test_scenario.name} ===")
        print(f"Config: {test_scenario.pipeline_config}")

        # Create unified workflow using factory pattern
        workflow = self._create_unified_workflow(test_scenario)

        # Execute workflow with initial context
        initial_context = WorkflowContext(
            synthetic_plate_dir=synthetic_plate_dir, test_scenario=test_scenario
        )
        workflow.execute(initial_context)

        print(f"✅ Unified workflow '{test_scenario.name}' validation passed!")

    def _create_unified_workflow(self, test_scenario: TestScenario) -> WorkflowBuilder:
        """Create unified workflow using algebraic factoring approach."""

        # Base workflow steps (common to all scenarios)
        base_steps = [
            ("Launch OpenHCS Application", _launch_application, TIMING.WINDOW_DELAY),
            ("Access Plate Manager", _access_plate_manager, None),
            ("Add and Select Plate", _add_and_select_plate, TIMING.ACTION_DELAY),
            ("Initialize Plate", _initialize_plate, TIMING.SAVE_DELAY),
            (
                "Apply Parameterized Orchestrator Configuration",
                _apply_orchestrator_config,
                TIMING.ACTION_DELAY,
            ),
            (
                "Verify Step Editor Placeholder Resolution",
                _verify_step_editor_placeholder_resolution,
                TIMING.ACTION_DELAY,
            ),
            ("Open Configuration Window", _open_config_window, TIMING.WINDOW_DELAY),
            (
                "Validate Initial Placeholder Behavior",
                _validate_placeholder_behavior,
                None,
            ),
            (
                "Apply Scenario Setup Modifications",
                _apply_setup_modifications,
                TIMING.SAVE_DELAY,
            ),
            ("Modify Field", _modify_field, TIMING.SAVE_DELAY),
            ("Reopen Configuration Window", _reopen_config_window, TIMING.WINDOW_DELAY),
            ("Validate Field Persistence", _validate_field_persistence, None),
            (
                "Validate Declared Inheritance Relationships",
                _validate_inheritance_relationships,
                None,
            ),
            ("Reset Field", _reset_field, TIMING.ACTION_DELAY),
            ("Validate Full Lazy State", _validate_full_lazy_state, None),
        ]

        # Build workflow using unified step factory
        workflow = WorkflowBuilder()
        for name, operation, timing_delay in base_steps:
            workflow.add_step(
                WorkflowStep(name=name, operation=operation, timing_delay=timing_delay)
            )

        return workflow
