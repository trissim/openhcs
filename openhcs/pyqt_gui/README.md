# OpenHCS PyQt6 GUI

Complete PyQt6 migration of the OpenHCS Textual TUI with full feature parity.

## Overview

This PyQt6 GUI provides a native desktop interface for OpenHCS (Open High-Content Screening) with enhanced usability, better desktop integration, and improved performance compared to the terminal-based Textual TUI.

## Features

### ✅ **Complete Feature Parity**
- All Textual TUI functionality preserved
- Plate management and pipeline editing
- Function library and parameter editing
- System monitoring and status display
- Configuration management

### 🚀 **Enhanced Capabilities**
- **Native Desktop Integration**: File dialogs, system tray, accessibility
- **Docking System**: Flexible window management with QDockWidget
- **Real-time Visualization**: Matplotlib integration for system monitoring
- **Improved Performance**: Qt's optimized rendering and event handling
- **Better Accessibility**: Screen reader support and keyboard navigation

### 🏗️ **Architecture**
- **Service Layer Preservation**: All business logic maintained
- **Hybrid Migration**: Extracted business logic + clean PyQt6 UI
- **Adapter Pattern**: Seamless integration with existing OpenHCS services
- **Signal/Slot System**: Robust event handling replacing Textual events

## Installation

### Prerequisites
```bash
# Required dependencies
pip install PyQt6 PyQt6-tools

# Optional dependencies for enhanced features
pip install matplotlib pyqtgraph psutil cupy dill
```

### Setup
```bash
# Install OpenHCS with PyQt6 GUI support
cd openhcs
pip install -e .
```

## Usage

### Launch Application
```bash
# Basic launch
python -m openhcs.pyqt_gui.launch

# With debug logging
python -m openhcs.pyqt_gui.launch --log-level DEBUG

# With custom configuration
python -m openhcs.pyqt_gui.launch --config my_config.json

# With log file
python -m openhcs.pyqt_gui.launch --log-file app.log
```

### Command Line Options
```
--log-level {DEBUG,INFO,WARNING,ERROR}  Set logging level
--log-file PATH                         Log file path
--config PATH                           Custom configuration file
--no-gpu                               Disable GPU acceleration
--version                              Show version information
```

## Architecture

### Component Structure
```
openhcs/pyqt_gui/
├── __init__.py              # Package initialization
├── app.py                   # Main application class
├── main.py                  # Main window implementation
├── launch.py                # Launch script and CLI
├── README.md                # This file
├── services/                # Service adapters
│   ├── __init__.py
│   ├── service_adapter.py   # PyQt service adapter
│   └── async_service_bridge.py  # Async operation bridge
├── widgets/                 # Core widgets
│   ├── __init__.py
│   ├── system_monitor.py    # System monitoring widget
│   ├── plate_manager.py     # Plate management widget
│   ├── pipeline_editor.py   # Pipeline editing widget
│   ├── function_pane.py     # Function parameter widget
│   └── status_bar.py        # Status display widget
├── windows/                 # Dialog windows
│   ├── __init__.py
│   ├── config_window.py     # Configuration dialog
│   ├── help_window.py       # Help dialog
│   ├── dual_editor_window.py    # Step/function editor
│   ├── file_browser_window.py   # File browser dialog
│   └── function_selector_window.py  # Function selector
└── shared/                  # Shared utilities
    ├── __init__.py
    ├── parameter_form_manager.py  # Parameter form management
    ├── signature_analyzer.py     # Function signature analysis
    └── typed_widget_factory.py   # Widget factory
```

### Service Integration

The PyQt6 GUI integrates seamlessly with existing OpenHCS services through an adapter pattern:

```python
# Service adapter bridges Textual dependencies to PyQt6
service_adapter = PyQtServiceAdapter(main_window)

# Existing services work unchanged
function_registry = FunctionRegistryService()
pattern_manager = PatternDataManager()
file_service = PatternFileService(service_adapter)  # Adapted
```

### Widget Migration Pattern

Widgets use a hybrid approach preserving business logic:

```python
# Extract business logic from Textual version
class PlateManagerWidget(QWidget):
    def __init__(self, file_manager, service_adapter):
        # Preserve business logic state
        self.plates = []
        self.orchestrators = {}
        
        # Clean PyQt6 UI implementation
        self.setup_ui()
        
        # Reuse business logic methods
        self.action_add_plate()  # From Textual version
```

## Key Components

### Main Application (`app.py`)
- `OpenHCSPyQtApp`: Main application class
- Global configuration management
- Exception handling and logging
- Application lifecycle management

### Main Window (`main.py`)
- `OpenHCSMainWindow`: Main window with dock system
- QDockWidget-based layout replacing textual-window
- Menu bar and status bar
- Window state persistence

### Service Adapters (`services/`)
- `PyQtServiceAdapter`: Bridges prompt_toolkit dependencies
- `AsyncServiceBridge`: Converts async/await to Qt threading
- Dialog management and system command execution

### Core Widgets (`widgets/`)
- `SystemMonitorWidget`: Real-time system monitoring with matplotlib
- `PlateManagerWidget`: Plate selection and management
- `PipelineEditorWidget`: Pipeline step editing
- `FunctionPaneWidget`: Function parameter editing
- `StatusBarWidget`: Application status display

### Dialog Windows (`windows/`)
- `ConfigWindow`: Configuration editing dialog
- `HelpWindow`: Help and documentation display
- `DualEditorWindow`: Advanced step/function editing
- `FileBrowserWindow`: File and directory selection
- `FunctionSelectorWindow`: Function library browser

## Migration Benefits

### Performance Improvements
- **Native Rendering**: Qt's optimized graphics system
- **GPU Integration**: Better GPU acceleration support
- **Memory Efficiency**: Optimized widget lifecycle management
- **Threading**: Proper async operation handling

### User Experience Enhancements
- **Desktop Integration**: Native file dialogs and system integration
- **Accessibility**: Screen reader and keyboard navigation support
- **Window Management**: Flexible docking and tabbing system
- **Visual Polish**: Professional appearance with consistent styling

### Developer Benefits
- **Maintainability**: Clean separation of UI and business logic
- **Extensibility**: Easy to add new widgets and features
- **Testing**: Better testing support with pytest-qt
- **Documentation**: Comprehensive inline documentation

## Development

### Adding New Widgets
1. Create widget class inheriting from appropriate Qt widget
2. Extract business logic from Textual equivalent (if exists)
3. Implement clean PyQt6 UI patterns
4. Connect to service adapter for external dependencies
5. Add to main window dock system

### Testing
```bash
# Run PyQt6 GUI tests
pytest tests/pyqt_gui/ -v

# Run with GUI testing
pytest tests/pyqt_gui/ --qt-gui
```

### Debugging
```bash
# Launch with debug logging
python -m openhcs.pyqt_gui.launch --log-level DEBUG --log-file debug.log

# Profile performance
py-spy record -o profile.svg -- python -m openhcs.pyqt_gui.launch
```

## Comparison with Textual TUI

| Feature | Textual TUI | PyQt6 GUI | Advantage |
|---------|-------------|-----------|-----------|
| **Performance** | Terminal-based | Native rendering | PyQt6 |
| **Desktop Integration** | Limited | Full native support | PyQt6 |
| **Accessibility** | Basic | Full screen reader support | PyQt6 |
| **Window Management** | Floating windows | Docking system | PyQt6 |
| **Visualization** | Text-based | Matplotlib integration | PyQt6 |
| **Resource Usage** | Lower | Higher | Textual |
| **Deployment** | SSH-friendly | Desktop-only | Textual |
| **Development Speed** | Rapid prototyping | More setup required | Textual |

## Future Enhancements

### Planned Features
- [ ] Plugin system for custom widgets
- [ ] Advanced visualization with PyQtGraph
- [ ] Integrated help system with searchable documentation
- [ ] Customizable themes and styling
- [ ] Workspace management and project files
- [ ] Advanced debugging and profiling tools

### Performance Optimizations
- [ ] Lazy loading for large datasets
- [ ] Virtual scrolling for large lists
- [ ] Background processing optimization
- [ ] Memory usage monitoring and optimization

## Contributing

1. Follow the hybrid migration pattern for new widgets
2. Preserve all business logic from Textual versions
3. Use clean PyQt6 UI patterns and best practices
4. Add comprehensive documentation and type hints
5. Include unit tests with pytest-qt

## License

Same as OpenHCS main project.

---

**OpenHCS PyQt6 GUI** - Professional desktop interface for high-content screening research.
