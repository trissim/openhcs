# PyQt6 GUI Tests

This directory contains tests for the PyQt6 GUI components of OpenHCS.

## Structure

- **`unit/`**: Unit tests for individual PyQt6 components
  - `test_parameter_form_manager.py`: Tests for the ParameterFormManager class
  - `test_widget_creation.py`: Tests for widget creation and configuration
  - `test_signal_handling.py`: Tests for PyQt6 signal/slot functionality
  - `test_placeholder_behavior.py`: Tests for placeholder text functionality

- **`integration/`**: Integration tests for PyQt6 GUI workflows
  - `test_form_workflows.py`: Tests for complete form creation and interaction workflows
  - `test_lazy_dataclass_integration.py`: Tests for lazy dataclass integration
  - `test_nested_form_behavior.py`: Tests for nested form functionality

## Test Categories

### Core Functionality Tests
- Parameter form creation and widget generation
- Parameter value updates and change signal emission
- Reset functionality (individual and bulk)
- Widget value retrieval and setting

### Advanced Functionality Tests
- Nested dataclass form handling
- Optional dataclass parameters with checkbox controls
- Threadlocal lazy dataclass loading behavior
- Placeholder text display for lazy-resolved values

### PyQt6-Specific Tests
- Signal/slot connections and emission
- Widget-specific behaviors (QLineEdit, QComboBox, etc.)
- Layout and UI structure validation
- Color scheme and styling integration

## Running Tests

### Run all PyQt6 GUI tests:
```bash
pytest tests/pyqt_gui/ -v
```

### Run specific test categories:
```bash
# Unit tests only
pytest tests/pyqt_gui/unit/ -v

# Integration tests only
pytest tests/pyqt_gui/integration/ -v

# Specific test file
pytest tests/pyqt_gui/unit/test_parameter_form_manager.py -v
```

### Run with GUI testing (if pytest-qt is available):
```bash
pytest tests/pyqt_gui/ --qt-gui -v
```

## Test Requirements

- PyQt6
- pytest
- OpenHCS test dependencies (see main tests/conftest.py)
- Optional: pytest-qt for enhanced GUI testing capabilities

## Notes

- All tests use the `qapp` fixture to ensure a QApplication instance exists
- Tests are designed to work with the refactored ParameterFormManager API
- Mock objects and fixtures are provided for testing without external dependencies
- Tests focus on the simplified API while ensuring backward compatibility
