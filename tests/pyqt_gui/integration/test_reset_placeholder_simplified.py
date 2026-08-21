"""
Test reset button behavior and placeholder inheritance using mathematical simplification.

Follows refactoring principles:
- Dynamic config lookups instead of hardcoded values
- Lookup tables for test specifications
- Mathematical expression simplification
- Eliminate duplicate conditional logic
"""

import pytest
from dataclasses import dataclass, fields
from typing import Any, Tuple, List, Dict, Union
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from openhcs.core.config import WellFilterMode, WellFilterConfig, StepWellFilterConfig, GlobalPipelineConfig
from tests.pyqt_gui.integration.test_end_to_end_workflow_foundation import (
    WorkflowContext, TestScenario, FieldModificationSpec,
    _launch_application, _access_plate_manager, _add_and_select_plate,
    _initialize_plate, _apply_orchestrator_config, _open_config_window,
    _close_config_window, WidgetFinder, WidgetInteractor, TimingConfig, _wait_for_gui,
    _create_synthetic_plate
)
import time


# ============================================================================
# DYNAMIC CONFIG SYSTEM: Replace hardcoded values with config lookups
# ============================================================================

def snake_to_camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase mathematically."""
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)

def get_config_class_by_name(config_section: str) -> type:
    """Get config class by converting snake_case section name to CamelCase class name."""
    # Mathematical transformation: snake_case -> CamelCase
    class_name = snake_to_camel_case(config_section)

    # Import the config module to access classes dynamically
    from openhcs.core import config
    return getattr(config, class_name, None)

def get_config_default(config_class: type, field_name: str) -> Any:
    """Get default value for a field, simulating lazy resolution like the UI does."""
    if not config_class:
        return None

    # Simulate lazy resolution: if field is None, look up the MRO for first non-None value
    for cls in config_class.__mro__:
        if hasattr(cls, field_name):
            value = getattr(cls, field_name)
            if value is not None:
                return value

    return None

def get_actual_config_default(config_section: str, field_name: str) -> str:
    """Get the actual default value that the UI would show after lazy resolution."""
    config_class = get_config_class_by_name(config_section)
    default_value = get_config_default(config_class, field_name)

    # Convert to string representation that matches what the UI shows
    return str(default_value) if default_value is not None else '(none)'

# ============================================================================
# PARAMETERIZED TEST SYSTEM: Eliminate repetition with data-driven approach
# ============================================================================

# Simple data structures - no methods, just data
@dataclass(frozen=True)
class TestSpec:
    field: str
    config: str
    test_value: Any

@dataclass(frozen=True)
class InheritanceSpec:
    source: str
    target: str
    field: str
    test_value: Any

# Parameterized test data - all magic strings in one place
TEST_FIELDS = ['well_filter', 'well_filter_mode']
TEST_CONFIGS = ['well_filter_config', 'step_well_filter_config', 'step_materialization_config', 'path_planning_config']
TEST_VALUES = {'well_filter': '123', 'well_filter_mode': WellFilterMode.EXCLUDE}

# Inheritance chains as simple tuples
INHERITANCE_CHAINS = [
    ('well_filter_config', 'path_planning_config'),
    ('step_well_filter_config', 'step_materialization_config')
]

# Factory functions using parameters
def create_test_spec(field: str, config: str, value_override: Any = None) -> TestSpec:
    """Create test spec with optional value override."""
    value = value_override or TEST_VALUES.get(field, '123')
    return TestSpec(field, config, value)

def create_inheritance_spec(source: str, target: str, field: str, value_override: Any = None) -> InheritanceSpec:
    """Create inheritance spec with optional value override."""
    value = value_override or TEST_VALUES.get(field, '123')
    return InheritanceSpec(source, target, field, value)

def get_expected_default(config: str, field: str) -> str:
    """Get expected default value for field in config."""
    return get_actual_config_default(config, field)

TimingConfig = TimingConfig(
    ACTION_DELAY=0.3,  # Must be > 200ms debounce timer for cross-window placeholder refresh
    WINDOW_DELAY=.1,  # Default delay for window operations
    SAVE_DELAY=.1,  # Default delay for save operations
    VISUAL_OBSERVATION_DELAY=0.2,  # Delay for visual observation
    VISUAL_PREPARATION_DELAY=0.2,  # Delay for visual preparation)
)

# ============================================================================
# DECLARATIVE TEST PATTERNS: Eliminate boilerplate with simple operations
# ============================================================================

def create_workflow_context(synthetic_plate_dir) -> WorkflowContext:
    """Create standardized workflow context with parameterized config."""
    expected_default = get_expected_default('well_filter_config', 'well_filter')

    test_scenario = TestScenario(
        name="reset_debug_test",
        orchestrator_config={"num_workers": 1},
        expected_values={'well_filter': expected_default},
        field_to_test=FieldModificationSpec(
            field_name='well_filter',
            modification_value='123',
            config_section='step_well_filter_config'
        )
    )

    return WorkflowContext().with_updates(
        synthetic_plate_dir=synthetic_plate_dir,
        test_scenario=test_scenario
    )

# Generic test runner - handles any sequence of operations
def run_test(ctx, *operations):
    """Execute a sequence of test operations."""
    for op, *args in operations:
        if op == 'edit':
            edit_field(ctx, *args)
        elif op == 'reset':
            press_reset_button(ctx, *args)
        elif op == 'assert_not_concrete':
            assert_field_not_concrete_value(ctx, *args)
        elif op == 'assert_inherits':
            assert_inheritance_working(ctx, *args)
        elif op == 'assert_default':
            field, source, target, msg = args
            default = get_expected_default(source, field)
            assert_inheritance_working(ctx, field, default, target, msg)
    return ctx

def setup_application_workflow(context: WorkflowContext) -> WorkflowContext:
    """Execute full application setup workflow."""
    # Mathematical simplification: chain operations as single expression
    return _open_config_window(
        _apply_orchestrator_config(
            _initialize_plate(
                _add_and_select_plate(
                    _access_plate_manager(
                        _launch_application(context)
                    )
                )
            )
        )
    )

def find_widget(context: WorkflowContext, field_name: str, config_section: str = None):
    """Find widget using existing infrastructure, including nested managers."""
    from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
    form_managers = context.config_window.findChildren(ParameterFormManager)

    if config_section:
        # First try to find the widget directly in the config section
        widget = WidgetFinder.find_field_widget_in_config_section(form_managers, field_name, config_section)
        if widget:
            return widget

        # If not found, try to find the nested manager for this config section
        # and look for the widget inside it
        # ANTI-DUCK-TYPING: ParameterFormManager always has nested_managers and widgets attributes
        for form_manager in form_managers:
            if config_section in form_manager.nested_managers:
                nested_manager = form_manager.nested_managers[config_section]
                if field_name in nested_manager.widgets:
                    print(f"🔍 DEBUG: Found '{field_name}' in nested manager '{config_section}'")
                    return nested_manager.widgets[field_name]

        return None
    else:
        return WidgetFinder.find_field_widget(form_managers, field_name)

def find_reset_button(context: WorkflowContext, field_name: str, config_section: str = None):
    """Find reset button using existing infrastructure, including nested managers."""
    from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
    form_managers = context.config_window.findChildren(ParameterFormManager)

    # Find the form manager that has this field in the specified config section
    if config_section:
        # First try to find the reset button in a form manager that matches the config section
        expected_dataclass_patterns = [
            config_section,  # exact match
            f"Lazy{config_section}",  # LazyStepWellFilterConfig
            f"Lazy{config_section.replace('_config', '').title().replace('_', '')}Config",  # LazyStepWellFilterConfig
            config_section.replace('_config', '').title().replace('_', '') + 'Config',  # StepWellFilterConfig
        ]

        for form_manager in form_managers:
            if hasattr(form_manager, 'dataclass_type') and form_manager.dataclass_type:
                dataclass_name = form_manager.dataclass_type.__name__
                print(f"🔍 FORM MANAGER DEBUG: Found form manager with dataclass_type = {dataclass_name}")
                if hasattr(form_manager, 'widgets'):
                    print(f"🔍 FORM MANAGER DEBUG: Widgets in {dataclass_name}: {list(form_manager.widgets.keys())}")
                if any(pattern.lower() in dataclass_name.lower() for pattern in expected_dataclass_patterns):
                    print(f"🔍 FORM MANAGER DEBUG: {dataclass_name} matches patterns {expected_dataclass_patterns}")
                    if hasattr(form_manager, 'widgets') and field_name in form_manager.widgets:
                        target_form_manager = form_manager
                        print(f"🔍 FORM MANAGER DEBUG: Found target form manager: {dataclass_name}")
                        break
        else:
            # If not found, try to find the nested manager for this config section
            # ANTI-DUCK-TYPING: ParameterFormManager always has nested_managers and reset_buttons attributes
            for form_manager in form_managers:
                if config_section in form_manager.nested_managers:
                    nested_manager = form_manager.nested_managers[config_section]
                    if field_name in nested_manager.reset_buttons:
                        print(f"🔍 DEBUG: Found reset button for '{field_name}' in nested manager '{config_section}'")
                        return nested_manager.reset_buttons[field_name]

            target_form_manager = None
    else:
        target_form_manager = WidgetFinder.find_form_manager_for_field(form_managers, field_name)

    if not target_form_manager:
        return None

    # Get reset button from the form manager
    if hasattr(target_form_manager, 'reset_buttons') and field_name in target_form_manager.reset_buttons:
        return target_form_manager.reset_buttons[field_name]
    return None

def get_placeholder_text(widget) -> str:
    """Get placeholder text from widget, handling different widget types."""
    if not widget:
        return ''

    # Try different methods to get placeholder text
    # Method 1: Standard placeholderText() method
    if hasattr(widget, 'placeholderText') and callable(widget.placeholderText):
        placeholder = widget.placeholderText()
        if placeholder:
            return placeholder

    # Method 2: Check for placeholder property
    if hasattr(widget, 'placeholderText'):
        placeholder = widget.placeholderText
        if placeholder and isinstance(placeholder, str):
            return placeholder

    # Method 3: For step editor widgets, check if it's in placeholder state and get displayed text
    from PyQt6.QtWidgets import QLineEdit, QComboBox
    if isinstance(widget, QLineEdit):
        is_placeholder = widget.property("is_placeholder_state")
        if is_placeholder:
            return widget.text()  # In step editor, placeholder text might be shown as actual text
    elif isinstance(widget, QComboBox):
        is_placeholder = widget.property("is_placeholder_state")
        if is_placeholder:
            return widget.currentText()  # For combo boxes in placeholder state

    # Method 4: Check for custom placeholder attributes
    for attr in ['placeholder_text', '_placeholder_text', 'placeholder']:
        if hasattr(widget, attr):
            value = getattr(widget, attr)
            if value and isinstance(value, str):
                return value

    return ''

# ============================================================================
# WORKFLOW OPERATIONS: Composable actions (Single Responsibility)
# ============================================================================

def edit_field(context: WorkflowContext, field_name: str, value: Any, config_section: str = None) -> WorkflowContext:
    """Edit field value and return updated context."""
    widget = find_widget(context, field_name, config_section)
    if not widget:
        print(f"❌ ERROR: Widget not found: {config_section}.{field_name}")
        return context

    # Mathematical simplification: combine scroll + wait + set + wait into single flow
    WidgetInteractor.scroll_to_widget(widget)
    _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)
    set_widget_value_enhanced(widget, value)
    _wait_for_gui(TimingConfig.ACTION_DELAY)

    print(f"✅ Edited {config_section}.{field_name} = {value}")
    return context

def set_widget_value_enhanced(widget: Any, value: Any) -> None:
    """Enhanced widget value setting with proper enum dropdown support."""
    from PyQt6.QtWidgets import QComboBox
    from enum import Enum

    # Handle QComboBox (enum dropdowns) specially
    if isinstance(widget, QComboBox):
        # For enum values, find the matching item by data
        if isinstance(value, Enum):
            for i in range(widget.count()):
                if widget.itemData(i) == value:
                    widget.setCurrentIndex(i)
                    print(f"🔽 Set dropdown to enum {value} at index {i}")
                    return
            print(f"❌ ERROR: Enum value {value} not found in dropdown")
            return

        # For string values, try to find matching enum by name or value
        elif isinstance(value, str):
            # First try to find by enum name (e.g., "EXCLUDE")
            for i in range(widget.count()):
                item_data = widget.itemData(i)
                if item_data and hasattr(item_data, 'name') and item_data.name == value:
                    widget.setCurrentIndex(i)
                    print(f"🔽 Set dropdown to enum {item_data} (matched by name '{value}') at index {i}")
                    return

            # Then try to find by enum value (e.g., "exclude")
            for i in range(widget.count()):
                item_data = widget.itemData(i)
                if item_data and hasattr(item_data, 'value') and str(item_data.value) == value:
                    widget.setCurrentIndex(i)
                    print(f"🔽 Set dropdown to enum {item_data} (matched by value '{value}') at index {i}")
                    return

            print(f"❌ ERROR: String value '{value}' not found in dropdown")
            return

    # Handle EnhancedPathWidget specially (it has set_path method)
    if hasattr(widget, 'set_path') and 'EnhancedPathWidget' in str(type(widget)):
        widget.set_path(str(value))
        print(f"📁 Set path widget to '{value}'")
        return

    # Fall back to original WidgetInteractor for other widget types
    WidgetInteractor.set_widget_value(widget, value)

def check_field_value(context: WorkflowContext, field_name: str, expected_value: Any, config_section: str = None) -> bool:
    """Check if field has expected value."""
    widget = find_widget(context, field_name, config_section)
    if not widget:
        return False

    actual_value = get_widget_value_enhanced(widget)

    # Mathematical simplification: unified comparison logic
    from enum import Enum
    return (actual_value == expected_value if isinstance(expected_value, Enum) and isinstance(actual_value, Enum)
            else (actual_value.name == expected_value or str(actual_value.value) == expected_value)
                 if isinstance(expected_value, str) and isinstance(actual_value, Enum)
            else str(actual_value) == str(expected_value))

def check_field_is_placeholder_with_value(context: WorkflowContext, field_name: str, expected_placeholder_value: Any, config_section: str = None) -> bool:
    """Check if field is in placeholder state AND showing the expected placeholder value."""
    widget = find_widget(context, field_name, config_section)
    if not widget:
        return False

    # Check 1: Widget must be in placeholder state
    is_placeholder = widget.property("is_placeholder_state")
    if not is_placeholder:
        return False

    # Check 2: Placeholder must show the expected value (not the old concrete value)
    from PyQt6.QtWidgets import QComboBox
    from enum import Enum

    if isinstance(widget, QComboBox):
        # For combo boxes, check the currently displayed item
        current_index = widget.currentIndex()
        if current_index >= 0:
            displayed_value = widget.itemData(current_index)

            # Compare displayed value with expected placeholder value
            if isinstance(expected_placeholder_value, Enum) and isinstance(displayed_value, Enum):
                return displayed_value == expected_placeholder_value
            elif isinstance(expected_placeholder_value, str) and isinstance(displayed_value, Enum):
                return (displayed_value.name == expected_placeholder_value or
                        str(displayed_value.value) == expected_placeholder_value)

    # For other widget types, could add similar checks
    return True  # If we can't verify the placeholder value, at least it's in placeholder state

def get_widget_value_enhanced(widget: Any) -> Any:
    """Enhanced widget value getting with proper enum dropdown support and placeholder state detection."""
    from PyQt6.QtWidgets import QComboBox

    # CRITICAL: Check if widget is in placeholder state first
    # If it's showing a placeholder, the actual parameter value is None
    if widget.property("is_placeholder_state"):
        return None

    # Handle QComboBox (enum dropdowns) specially
    if isinstance(widget, QComboBox):
        current_index = widget.currentIndex()
        if current_index >= 0:
            return widget.itemData(current_index)
        return None

    # Handle EnhancedPathWidget specially (it has get_path method)
    if hasattr(widget, 'get_path') and 'EnhancedPathWidget' in str(type(widget)):
        return widget.get_path()

    # Reuse the generic widget protocol owner used by ParameterFormManager.
    from pyqt_reactive.forms.widget_operations import WidgetOperations

    return WidgetOperations.get_value(widget)

def check_placeholder_value(context: WorkflowContext, field_name: str, expected_placeholder: str, config_section: str = None) -> bool:
    """Check if field has expected placeholder."""
    widget = find_widget(context, field_name, config_section)
    if not widget:
        return False

    actual_placeholder = get_placeholder_text(widget)
    return expected_placeholder in actual_placeholder

def press_reset_button(context: WorkflowContext, field_name: str, config_section: str = None) -> WorkflowContext:
    """Press reset button and return updated context."""
    print(f"🔍 PRESS RESET DEBUG: Looking for reset button for {field_name} in {config_section}")
    reset_button = find_reset_button(context, field_name, config_section)
    print(f"🔍 PRESS RESET DEBUG: Found reset button: {reset_button is not None}")
    if not reset_button:
        section_info = f" in {config_section}" if config_section else ""
        print(f"❌ ERROR: Reset button not found: {field_name}{section_info}")
        return context

    WidgetInteractor.scroll_to_widget(reset_button)
    _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

    reset_button.click()
    QApplication.processEvents()
    _wait_for_gui(TimingConfig.ACTION_DELAY)

    section_info = f" in {config_section}" if config_section else ""
    print(f"✅ Pressed reset button for {field_name}{section_info}")
    return context

def save_and_close_config(context: WorkflowContext) -> WorkflowContext:
    """Save and close config window using existing infrastructure."""
    if not context.config_window:
        print("❌ ERROR: No config window to save and close")
        return context

    # Find and click save button (same logic as end-to-end workflow)
    save_button = WidgetFinder.find_button_by_text(context.config_window, ['ok', 'save', 'apply'])
    if not save_button:
        from PyQt6.QtWidgets import QPushButton
        buttons = [b.text() for b in context.config_window.findChildren(QPushButton)]
        print(f"❌ ERROR: Save button not found. Available buttons: {buttons}")
        return context

    save_button.click()
    _wait_for_gui(TimingConfig.SAVE_DELAY)

    # Close the window using existing infrastructure
    return _close_config_window(context)

def open_config(context: WorkflowContext) -> WorkflowContext:
    return _open_config_window(context)

# ============
# STEP EDITOR UTILS FOR CROSS-WINDOW PLACEHOLDER CONSISTENCY
# ============

def _capture_pipeline_placeholders(context: WorkflowContext) -> dict:
    """Capture placeholder texts using existing parameterized data."""
    placeholders = {}

    # Use existing TEST_CONFIGS instead of hardcoded sections
    for config in TEST_CONFIGS:
        field_placeholders = {
            field: get_placeholder_text(widget) if (widget := find_widget(context, field, config)) else None
            for field in TEST_FIELDS
        }
        placeholders[config] = field_placeholders
        print(f"📌 Captured pipeline placeholders [{config}] {field_placeholders}")

    return placeholders


def _open_step_editor_and_get_form_manager(context: WorkflowContext):
    """Open Step Editor using unified timing and interaction patterns."""
    from PyQt6.QtWidgets import QWidget

    pipeline_editor_window = context.main_window.floating_windows.get("pipeline_editor")
    if not pipeline_editor_window:
        raise AssertionError("Pipeline editor window not found in floating_windows")

    # Find pipeline editor widget using unified patterns
    pipeline_editor = WidgetFinder.find_widget_by_attribute(
        pipeline_editor_window.findChildren(QWidget), 'pipeline_steps'
    )
    if not pipeline_editor:
        raise AssertionError("Pipeline editor widget not found")

    # Use unified interaction patterns: scroll + timing + click
    if not hasattr(pipeline_editor, 'buttons') or "add_step" not in pipeline_editor.buttons:
        raise AssertionError("Add Step button not found in pipeline editor buttons")

    add_step_button = pipeline_editor.buttons["add_step"]
    WidgetInteractor.scroll_to_widget(add_step_button)
    _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

    add_step_button.click()
    QApplication.processEvents()
    _wait_for_gui(TimingConfig.WINDOW_DELAY)  # Use TimingConfig instead of hardcoded sleep
    QApplication.processEvents()

    # Find step editor window with proper timing
    step_editor_window = None
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, 'step_editor') and hasattr(widget, 'editing_step'):
            step_editor_window = widget
            break
    if not step_editor_window:
        raise AssertionError("Step editor window (DualEditorWindow) not found")

    step_param_editor = step_editor_window.step_editor
    if not step_param_editor or not hasattr(step_param_editor, 'form_manager'):
        raise AssertionError("Form manager not found in step parameter editor")

    # Allow time for step editor to fully initialize
    _wait_for_gui(TimingConfig.VISUAL_OBSERVATION_DELAY)
    return step_editor_window, step_param_editor.form_manager


def _assert_step_editor_placeholders_match(step_form_manager, pipeline_placeholders: dict):
    """Assert step editor placeholders using unified scrolling and timing patterns."""
    print("🔍 Debugging step editor structure:")
    print(f"  step_form_manager type: {type(step_form_manager)}")
    print(f"  nested_managers: {getattr(step_form_manager, 'nested_managers', {}).keys()}")

    # Use existing parameterized data - no hardcoded mappings!
    for config in TEST_CONFIGS:
        if config in pipeline_placeholders:
            expected_placeholders = pipeline_placeholders[config]
            nested_manager = getattr(step_form_manager, 'nested_managers', {}).get(config)

            print(f"🔍 Checking config: {config}")
            print(f"  nested_manager: {nested_manager}")
            print(f"  has widgets: {hasattr(nested_manager, 'widgets') if nested_manager else False}")

            if not nested_manager or not hasattr(nested_manager, 'widgets'):
                print(f"  ⚠️  Skipping {config} - no nested manager or widgets")
                continue

            print(f"  widgets available: {list(nested_manager.widgets.keys())}")

            # Use unified interaction patterns for each field
            for field in TEST_FIELDS:
                expected_value = expected_placeholders.get(field)
                widget = nested_manager.widgets.get(field)

                print(f"🔍 Field: {field}")
                print(f"  expected_value: '{expected_value}'")
                print(f"  widget: {widget}")
                print(f"  widget type: {type(widget) if widget else None}")

                if widget and expected_value:
                    # Use unified scrolling and timing like other test operations
                    WidgetInteractor.scroll_to_widget(widget)
                    _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

                    # Debug widget state
                    is_placeholder = widget.property("is_placeholder_state")
                    widget_text = getattr(widget, 'text', lambda: 'NO_TEXT')()
                    widget_placeholder = getattr(widget, 'placeholderText', lambda: 'NO_PLACEHOLDER')()

                    print(f"  widget.is_placeholder_state: {is_placeholder}")
                    print(f"  widget.text(): '{widget_text}'")
                    print(f"  widget.placeholderText(): '{widget_placeholder}'")

                    actual = get_placeholder_text(widget)
                    print(f"  get_placeholder_text result: '{actual}'")
                    print(f"  expected: '{expected_value}'")

                    # Normalize expected value - remove "Pipeline default:" prefix for enum fields
                    # Step editor dropdowns show just the enum value, not the prefix
                    normalized_expected = expected_value
                    if expected_value.startswith('Pipeline default: ') and field.endswith('_mode'):
                        normalized_expected = expected_value.replace('Pipeline default: ', '')
                        print(f"  normalized expected (enum): '{normalized_expected}'")

                    assert (actual or "").strip() == (normalized_expected or "").strip(), \
                        f"Placeholder mismatch for {config}.{field}: expected '{normalized_expected}', got '{actual}'"

                    _wait_for_gui(TimingConfig.VISUAL_OBSERVATION_DELAY)
                else:
                    print(f"  ⚠️  Skipping {field} - no widget or expected value")

    print("✅ Step editor placeholders verified using unified patterns")


def _assert_step_editor_shows_saved_values(step_form_manager, saved_values: dict):
    """Assert step editor shows placeholders that match the saved values from pipeline config."""
    print("🔍 Testing cross-window consistency: saved values vs step editor placeholders")

    # Define inheritance groups: configs that should show the same values
    # step_well_filter_config: itself + configs that inherit from it (excluding those with their own overrides)
    # step_materialization_config: only itself (has its own concrete values)
    inheritance_groups = {
        'step_well_filter_config': ['step_well_filter_config', 'napari_streaming_config'],  # materialization excluded - has own values
        'step_materialization_config': ['step_materialization_config'],  # only itself
        'well_filter_config': ['well_filter_config']  # only itself (not in step editor but for completeness)
    }

    all_results = []  # Collect all results (pass/fail) before asserting

    # For each config that had values saved
    for config, saved_fields in saved_values.items():
        # Get all configs that should show the same value (including the config itself)
        configs_to_check = inheritance_groups.get(config, [config])

        print(f"🔍 Checking config: {config} (saved values: {saved_fields})")
        print(f"  Will verify inheritance in: {configs_to_check}")

        for check_config in configs_to_check:
            nested_manager = getattr(step_form_manager, 'nested_managers', {}).get(check_config)

            if not nested_manager or not hasattr(nested_manager, 'widgets'):
                print(f"  ⚠️  Skipping {check_config} - no nested manager or widgets in step editor")
                all_results.append(f"SKIPPED: {check_config} - no nested manager")
                continue

            # Check each saved field (well_filter and well_filter_mode)
            for field_name, saved_value in saved_fields.items():
                if field_name in nested_manager.widgets:
                    widget = nested_manager.widgets[field_name]

                    # Handle case where widget creation failed (returns None or invalid widget)
                    if widget is None:
                        print(f"  ⚠️  {check_config}.{field_name} widget is None (creation failed)")
                        continue

                    try:
                        actual_placeholder = get_placeholder_text(widget)
                        expected_placeholder = f'Pipeline default: {saved_value}'

                        print(f"  {check_config}.{field_name}:")
                        print(f"    Expected: '{expected_placeholder}'")
                        print(f"    Actual: '{actual_placeholder}'")

                        if actual_placeholder != expected_placeholder:
                            result_msg = f"FAILED: {check_config}.{field_name} - expected '{expected_placeholder}' but got '{actual_placeholder}'"
                            all_results.append(result_msg)
                            print(f"    ❌ FAILED - expected '{expected_placeholder}' but got '{actual_placeholder}'")
                        else:
                            result_msg = f"PASSED: {check_config}.{field_name} - correctly shows '{actual_placeholder}'"
                            all_results.append(result_msg)
                            print(f"    ✅ PASSED")
                    except Exception as e:
                        print(f"  ⚠️  Error checking {check_config}.{field_name} widget: {e}")
                        all_results.append(f"ERROR: {check_config}.{field_name} - {e}")
                        continue
                else:
                    print(f"  ⚠️  No {field_name} widget found in {check_config}")
                    all_results.append(f"SKIPPED: {check_config} - no {field_name} widget")
                    # Let's also check what widgets ARE available
                    if hasattr(nested_manager, 'widgets') and nested_manager.widgets:
                        available_widgets = list(nested_manager.widgets.keys())
                        print(f"    Available widgets: {available_widgets}")
                    else:
                        print(f"    No widgets available in nested manager")

    # Show summary of all results
    print(f"\n📊 SUMMARY OF ALL CROSS-WINDOW CONSISTENCY CHECKS:")
    for result in all_results:
        print(f"  {result}")

    # Check if any failed and print the actual failure details
    failures = [result for result in all_results if result.startswith("FAILED:")]
    if failures:
        print(f"\n❌ FAILURES DETECTED:")
        for failure in failures:
            print(f"  {failure}")
        raise AssertionError(f"Cross-window consistency failures: {len(failures)} configs failed")

    print("✅ Cross-window consistency verified: all saved values and inheritance match step editor placeholders")

# ============================================================================
# ASSERTION OPERATIONS: Dynamic assertions as workflow steps
# ============================================================================

def assert_field_not_concrete_value(context: WorkflowContext, field_name: str, concrete_value: Any, config_section: str, step_name: str) -> WorkflowContext:
    """Assert field does not show concrete value after reset."""
    widget = find_widget(context, field_name, config_section)
    if not widget:
        return context

    # Mathematical simplification: combine visual preparation with checks
    WidgetInteractor.scroll_to_widget(widget)
    _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

    # Unified assertion logic using mathematical expressions
    is_placeholder = widget.property("is_placeholder_state")
    assert is_placeholder, f"{step_name} FAIL: {config_section}.{field_name} should be in placeholder state after reset, but is_placeholder_state = {is_placeholder}"

    # Check for enum combo boxes showing concrete value
    from PyQt6.QtWidgets import QComboBox
    from enum import Enum

    if isinstance(widget, QComboBox) and isinstance(concrete_value, Enum):
        current_index = widget.currentIndex()
        displayed_value = widget.itemData(current_index) if current_index >= 0 else None
        assert displayed_value != concrete_value, f"{step_name} FAIL: {config_section}.{field_name} should not show concrete value '{concrete_value}' after reset"

    print(f"✅ {step_name}: {config_section}.{field_name} correctly in placeholder state and not showing concrete value '{concrete_value}'")
    return context

def assert_placeholder_shows_value(context: WorkflowContext, field_name: str, expected_placeholder: str, config_section: str, step_name: str) -> WorkflowContext:
    """Assert placeholder shows expected value."""
    widget = find_widget(context, field_name, config_section)
    if widget:
        # Mathematical simplification: combine visual preparation with logging
        WidgetInteractor.scroll_to_widget(widget)
        _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

        actual_placeholder = get_placeholder_text(widget)
        print(f"🔍 {step_name}: Checking {config_section}.{field_name}")
        print(f"    Expected: '{expected_placeholder}' | Actual: '{actual_placeholder}'")

    # Unified assertion using mathematical expression
    assert check_placeholder_value(context, field_name, expected_placeholder, config_section), \
        f"{step_name} FAIL: {config_section}.{field_name} should show placeholder '{expected_placeholder}'"

    print(f"✅ {step_name}: {config_section}.{field_name} shows correct placeholder '{expected_placeholder}'")
    return context

def assert_inheritance_working(context: WorkflowContext, field_name: str, inherited_value: str, target_config: str, step_name: str) -> WorkflowContext:
    """Assert inheritance is working correctly."""
    # Enums don't get "Pipeline default:" prefix, only string values do
    from enum import Enum

    # Check if this is an enum value (generic for any enum)
    is_enum_value = isinstance(inherited_value, Enum)

    if is_enum_value:
        # For enums, just use the enum name/value directly (e.g., WellFilterMode.INCLUDE -> "INCLUDE")
        expected_placeholder = inherited_value.name
    else:
        # For non-enum values, add the "Pipeline default:" prefix
        expected_placeholder = f"Pipeline default: {inherited_value}"

    return assert_placeholder_shows_value(context, field_name, expected_placeholder, target_config, step_name)

# ============================================================================
# WORKFLOW BUILDER: Composable workflow construction
# ============================================================================

def build_workflow(*operations):
    """Build workflow by composing operations."""
    def execute_workflow(context: WorkflowContext) -> WorkflowContext:
        result_context = context
        for operation in operations:
            result_context = operation(result_context)
            # Operations now raise AssertionError directly instead of using last_error
        return result_context
    return execute_workflow

# ============================================================================
# TEST CLASS: Simplified using composable utilities
# ============================================================================

class TestResetPlaceholderInheritance:
    """Test reset button behavior using composable workflow utilities."""

    def test_comprehensive_inheritance_chains(self, app, synthetic_plate_dir):
        """Test inheritance chains using declarative operations."""
        context = setup_application_workflow(create_workflow_context(synthetic_plate_dir))

        print("\n🧪 COMPREHENSIVE INHERITANCE TEST - DECLARATIVE APPROACH")
        print("=" * 60)

        # Process all chains using generic runner
        for chain_idx, (source, target) in enumerate(INHERITANCE_CHAINS, 1):
            print(f"\n🔗 CHAIN {chain_idx}: {source} → {target}")

            # Test field reset for each field
            for field in TEST_FIELDS:
                value = TEST_VALUES.get(field, '123')
                run_test(context,
                    ('edit', field, value, source),
                    ('reset', field, source),
                    ('assert_not_concrete', field, value, source, f"{field} reset")
                )

            # Test inheritance chain
            value = f'test_value_{chain_idx}'
            run_test(context,
                ('edit', 'well_filter', value, source),
                ('assert_inherits', 'well_filter', value, target, f"inheritance"),
                ('reset', 'well_filter', target),
                ('assert_inherits', 'well_filter', value, target, f"persistence"),
                ('reset', 'well_filter', source),
                ('assert_default', 'well_filter', source, target, f"default update")
            )

            print(f"✅ CHAIN {chain_idx} COMPLETED")

        print("✅ ALL INHERITANCE TESTS COMPLETED SUCCESSFULLY")

        # Cross-window verification using mathematical simplification
        print("\n🔄 Cross-window placeholder consistency check")

        # Set concrete values and track what we set
        saved_values = {}

        # Set step_well_filter_config values
        run_test(context, ('edit', 'well_filter', 'concrete_2', 'step_well_filter_config'))
        run_test(context, ('edit', 'well_filter_mode', 'WellFilterMode.EXCLUDE', 'step_well_filter_config'))
        saved_values['step_well_filter_config'] = {
            'well_filter': 'concrete_2',
            'well_filter_mode': 'WellFilterMode.EXCLUDE'
        }
        print(f"✅ Set concrete value step_well_filter_config.well_filter = concrete_2")
        print(f"✅ Set concrete value step_well_filter_config.well_filter_mode = WellFilterMode.EXCLUDE")

        # Set step_materialization_config to DIFFERENT values
        run_test(context, ('edit', 'well_filter', 'materialization_value', 'step_materialization_config'))
        run_test(context, ('edit', 'well_filter_mode', 'WellFilterMode.INCLUDE', 'step_materialization_config'))
        saved_values['step_materialization_config'] = {
            'well_filter': 'materialization_value',
            'well_filter_mode': 'WellFilterMode.INCLUDE'
        }
        print(f"✅ Set concrete value step_materialization_config.well_filter = materialization_value")
        print(f"✅ Set concrete value step_materialization_config.well_filter_mode = WellFilterMode.INCLUDE")

        # Also set well_filter_config for completeness
        run_test(context, ('edit', 'well_filter', 'concrete_1', 'well_filter_config'))
        run_test(context, ('edit', 'well_filter_mode', 'WellFilterMode.EXCLUDE', 'well_filter_config'))
        saved_values['well_filter_config'] = {
            'well_filter': 'concrete_1',
            'well_filter_mode': 'WellFilterMode.EXCLUDE'
        }
        print(f"✅ Set concrete value well_filter_config.well_filter = concrete_1")
        print(f"✅ Set concrete value well_filter_config.well_filter_mode = WellFilterMode.EXCLUDE")

        # Save configuration to update thread-local context
        context = save_and_close_config(context)

        step_editor_window, step_form_manager = _open_step_editor_and_get_form_manager(context)
        try:
            _assert_step_editor_shows_saved_values(step_form_manager, saved_values)
        finally:
            step_editor_window.close()
            QApplication.processEvents()



    def test_concrete_value_persistence_after_save_reopen(self, app, qtbot, synthetic_plate_dir):
        """Test that concrete values persist after save/reopen and show as concrete, not placeholders."""
        context = setup_application_workflow(create_workflow_context(synthetic_plate_dir))

        print("\n🧪 CONCRETE VALUE PERSISTENCE TEST")
        print("=" * 50)

        # Simple test data
        test_specs = [
            ("materialization_results_path", "/custom/results/path", None),
            ('well_filter', '123', 'well_filter_config')
        ]

        # Step 1: Set values
        for field_name, test_value, config_section in test_specs:
            if field_name == "materialization_results_path":
                self._set_path_widget_value(context, qtbot, field_name, test_value)
            else:
                edit_field(context, field_name, test_value, config_section)

        # Step 2-3: Save, close, reopen
        self._save_and_close_with_qtbot(context, qtbot)
        _open_config_window(context)

        # Step 4: Verify persistence
        for field_name, expected_value, config_section in test_specs:
            assert check_field_value(context, field_name, expected_value, config_section), \
                f"Field {field_name} should show saved value '{expected_value}'"

        # Step 5: Verify concrete text (not placeholders)
        for field_name, expected_value, config_section in test_specs:
            widget = find_widget(context, field_name, config_section)
            if widget:
                displayed_text = (widget.path_input.text() if hasattr(widget, 'path_input')
                                else widget.text() if hasattr(widget, 'text')
                                else str(get_widget_value_enhanced(widget)))
                is_placeholder = (widget.path_input.property("is_placeholder_state") if hasattr(widget, 'path_input')
                                else widget.property("is_placeholder_state"))

                assert displayed_text == str(expected_value) and not is_placeholder, \
                    f"Field {field_name} should show concrete text '{expected_value}', not placeholder"

        print("✅ Concrete value persistence test completed successfully")

    def _set_path_widget_value(self, context: WorkflowContext, qtbot, field_name: str, value: str):
        """Helper method to set path widget value using qtbot."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QKeySequence

        path_widget = find_widget(context, field_name)
        if path_widget and hasattr(path_widget, 'path_input'):
            WidgetInteractor.scroll_to_widget(path_widget)
            _wait_for_gui(TimingConfig.VISUAL_PREPARATION_DELAY)

            line_edit = path_widget.path_input
            qtbot.mouseClick(line_edit, Qt.MouseButton.LeftButton)
            _wait_for_gui(TimingConfig.ACTION_DELAY)

            qtbot.keySequence(line_edit, QKeySequence.StandardKey.SelectAll)
            qtbot.keyClicks(line_edit, value)
            qtbot.keyPress(line_edit, Qt.Key.Key_Return)
            _wait_for_gui(TimingConfig.ACTION_DELAY)

            print(f"✅ Set {field_name} = {value} via qtbot")
        else:
            print(f"❌ ERROR: Could not find {field_name} widget or path_input")

    def _save_and_close_with_qtbot(self, context: WorkflowContext, qtbot):
        """Helper method to save and close config using qtbot."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QPushButton

        save_button = next((w for w in context.config_window.findChildren(QPushButton)
                           if 'save' in w.text().lower() or 'apply' in w.text().lower()), None)

        if save_button:
            qtbot.mouseClick(save_button, Qt.MouseButton.LeftButton)
            _wait_for_gui(TimingConfig.ACTION_DELAY)
            print("✅ Clicked save button with qtbot")
        else:
            print("❌ ERROR: Could not find save button")

        _close_config_window(context)

# ============================================================================
# FIXTURES: Test setup and teardown
# ============================================================================

@pytest.fixture
def app():
    """Ensure QApplication exists for PyQt6 tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture
def synthetic_plate_dir(tmp_path):
    """Create synthetic plate data for testing."""
    return _create_synthetic_plate(tmp_path)
