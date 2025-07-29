# OpenHCS Textual TUI to PyQt6 Migration Plan

## Executive Summary

This document outlines a comprehensive migration strategy for converting the OpenHCS Textual TUI to PyQt6, leveraging the excellent architectural separation already present in the codebase. The migration is estimated at **6-8 weeks** with **70-80% code reuse** from the existing service layer and business logic.

## Architecture Analysis

### Current Textual TUI Architecture

**Strengths:**
- ✅ Excellent separation of concerns between UI and business logic
- ✅ Service layer is completely UI-agnostic
- ✅ Shared utilities (`SignatureAnalyzer`, `ParameterFormManager`) are well-abstracted
- ✅ Consistent widget composition patterns
- ✅ Reactive state management with clear data flow

**Components:**
```
OpenHCSTUIApp (Main Application)
├── MainContent (System Monitor Background)
├── StatusBar (Status Messages)
├── CustomWindowBar (Window Management)
└── Windows (Floating)
    ├── PlateManagerWidget
    ├── PipelineEditorWidget
    ├── FunctionPaneWidget
    └── ConfigWindow

Service Layer (UI-Agnostic):
├── FunctionRegistryService
├── PatternDataManager
├── PatternFileService
├── ExternalEditorService
└── WindowService

Shared Utilities (Reusable):
├── SignatureAnalyzer
├── ParameterFormManager
├── TypedWidgetFactory
├── DocstringExtractor
└── UnifiedParameterAnalyzer
```

### Target PyQt6 Architecture

**Proposed Structure:**
```python
class OpenHCSMainWindow(QMainWindow):
    """Main application window with menu bar and central widget"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_services()  # Reuse existing services
        self.connect_signals()
    
    def setup_ui(self):
        # Central widget with splitter layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Add main components
        self.plate_manager = PlateManagerWidget()
        self.pipeline_editor = PipelineEditorWidget()
        
        splitter.addWidget(self.plate_manager)
        splitter.addWidget(self.pipeline_editor)
```

## Migration Strategy

### Phase 1: Foundation & Core Services (Week 1-2)
**Objective:** Establish PyQt6 foundation and migrate service layer

**Deliverables:**
1. **Project Structure Setup**
   ```
   openhcs/
   ├── pyqt_gui/
   │   ├── __init__.py
   │   ├── main_window.py
   │   ├── widgets/
   │   ├── services/  # Symlink to existing services
   │   ├── shared/    # Migrated shared utilities
   │   └── resources/
   ```

2. **Service Layer Integration** (90% reuse)
   - Copy existing service classes unchanged
   - Update imports only
   - Test service functionality in PyQt context

3. **Shared Utilities Migration** (80% reuse)
   - `SignatureAnalyzer` → Direct copy (pure business logic)
   - `ParameterFormManager` → Adapt widget creation methods
   - `TypedWidgetFactory` → Replace Textual widgets with Qt widgets

**Code Example - ParameterFormManager Migration:**
```python
# Current Textual version
def _create_widget_for_type(self, param_type, value):
    if param_type == str:
        return Input(value=str(value))
    elif param_type == bool:
        return Checkbox(value=value)

# PyQt6 version
def _create_widget_for_type(self, param_type, value):
    if param_type == str:
        widget = QLineEdit(str(value))
        return widget
    elif param_type == bool:
        widget = QCheckBox()
        widget.setChecked(value)
        return widget
```

### Phase 2: Core Widgets (Week 3-4)
**Objective:** Migrate primary UI components

**Priority Order:**
1. **ParameterFormManager** (Critical - used everywhere)
2. **FunctionPaneWidget** (Core functionality)
3. **PlateManagerWidget** (File browsing)
4. **PipelineEditorWidget** (Pipeline creation)

**Migration Pattern:**
```python
# Textual Pattern
class FunctionPaneWidget(Widget):
    def compose(self) -> ComposeResult:
        yield Button("Add", id="add_btn")
        yield Static("Function Name")
    
    @on(Button.Pressed, "#add_btn")
    def handle_add(self):
        # Business logic (preserved)

# PyQt6 Pattern  
class FunctionPaneWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.add_btn = QPushButton("Add")
        layout.addWidget(self.add_btn)
        layout.addWidget(QLabel("Function Name"))
        self.setLayout(layout)
    
    def connect_signals(self):
        self.add_btn.clicked.connect(self.handle_add)
    
    def handle_add(self):
        # Same business logic (preserved)
```

### Phase 3: Main Application & Window Management (Week 5-6)
**Objective:** Create main application structure and window system

**Components:**
1. **Main Window** with menu bar, toolbar, status bar
2. **Docking System** for flexible layout (replace textual-window)
3. **Settings Dialog** for configuration
4. **Help System** integration

**Key Features:**
- Native menu bar with keyboard shortcuts
- Dockable widgets for flexible layout
- Session persistence (window positions, sizes)
- Native file dialogs
- System tray integration (optional)

### Phase 4: Advanced Features & Polish (Week 7-8)
**Objective:** Complete feature parity and enhance UX

**Enhancements:**
1. **Native Integration**
   - File associations
   - Drag & drop support
   - Clipboard integration
   - Print support

2. **Performance Optimization**
   - Lazy loading for large datasets
   - Background processing with QThread
   - Progress indicators
   - Memory optimization

3. **Accessibility**
   - Screen reader support
   - Keyboard navigation
   - High contrast themes
   - Font scaling

## Code Reusability Assessment

### Highly Reusable (90-100% preservation)

**Service Layer:**
```python
# These classes transfer directly with minimal changes
- FunctionRegistryService      # Pure business logic
- PatternDataManager          # Data manipulation
- PatternFileService          # File I/O operations
- ExternalEditorService       # Process management
```

**Business Logic:**
```python
# Core analysis and processing - no changes needed
- SignatureAnalyzer._analyze_callable()
- DocstringExtractor.extract()
- Parameter validation logic
- File format handling
- Configuration management
```

### Moderately Reusable (50-70% preservation)

**Widget Logic:**
```python
# State management and event handling logic preserved
# Widget creation methods need adaptation
class ParameterFormManager:
    # Preserve: parameter analysis, validation, state management
    # Adapt: widget creation, layout management
    
    def build_form(self):  # Logic preserved
        for param_name, param_type in self.parameter_types.items():
            widget = self._create_widget_for_type(param_type, value)  # Adapt
            # Layout logic needs PyQt adaptation
```

### Requires Complete Rewrite (0-20% preservation)

**UI Composition:**
```python
# Textual compose() methods → PyQt __init__() + layouts
# CSS styling → Qt stylesheets or themes
# Message passing → Signal/slot connections
# Reactive variables → Property system or manual updates
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Setup PyQt6 project structure
- [ ] Migrate service layer (copy + test)
- [ ] Create base window class
- [ ] Setup development environment

### Week 2: Core Utilities
- [ ] Migrate SignatureAnalyzer (direct copy)
- [ ] Adapt ParameterFormManager for PyQt widgets
- [ ] Create TypedWidgetFactory for Qt widgets
- [ ] Unit tests for shared utilities

### Week 3: Primary Widgets
- [ ] FunctionPaneWidget migration
- [ ] Parameter form integration
- [ ] Event handling implementation
- [ ] Basic layout management

### Week 4: Data Widgets
- [ ] PlateManagerWidget migration
- [ ] File browser integration
- [ ] Tree view implementation
- [ ] Selection handling

### Week 5: Pipeline Editor
- [ ] PipelineEditorWidget migration
- [ ] Drag & drop functionality
- [ ] Visual pipeline representation
- [ ] Step editing integration

### Week 6: Main Application
- [ ] Main window implementation
- [ ] Menu bar and toolbar
- [ ] Docking system
- [ ] Window management

### Week 7: Integration & Testing
- [ ] End-to-end integration
- [ ] Performance testing
- [ ] Bug fixes and refinement
- [ ] Documentation updates

### Week 8: Polish & Deployment
- [ ] UI/UX improvements
- [ ] Accessibility features
- [ ] Packaging and distribution
- [ ] Migration documentation

## Risk Assessment & Mitigation

### High Risk Areas

**1. Complex Widget Interactions**
- *Risk:* Textual's reactive system vs PyQt's signal/slot
- *Mitigation:* Create adapter layer for state management

**2. Layout Management**
- *Risk:* Textual's automatic layout vs Qt's manual layout
- *Mitigation:* Use Qt Designer for complex layouts

**3. Performance Differences**
- *Risk:* Different rendering and event handling performance
- *Mitigation:* Profile early and optimize bottlenecks

### Medium Risk Areas

**1. Styling System**
- *Risk:* CSS-like styling vs Qt stylesheets
- *Mitigation:* Create style guide and reusable themes

**2. Window Management**
- *Risk:* textual-window vs Qt docking
- *Mitigation:* Implement similar floating window system

## Tools & Automation

### Development Tools
1. **Qt Designer** - Visual layout design
2. **PyQt6-tools** - UI compilation and resources
3. **pytest-qt** - GUI testing framework
4. **Qt Creator** - Advanced debugging

### Migration Helpers
```python
# Automated widget mapping
WIDGET_MAPPING = {
    'Input': 'QLineEdit',
    'Button': 'QPushButton', 
    'Static': 'QLabel',
    'Checkbox': 'QCheckBox',
    'Select': 'QComboBox',
}

# Event mapping
EVENT_MAPPING = {
    'Button.Pressed': 'clicked',
    'Input.Changed': 'textChanged',
    'Select.Changed': 'currentTextChanged',
}
```

### Code Generation
- Script to generate PyQt widget skeletons from Textual widgets
- Automated signal/slot connection generation
- Layout code generation from widget hierarchies

## Testing Strategy

### Unit Testing
- Preserve existing business logic tests
- Add PyQt-specific widget tests
- Mock Qt components for isolated testing

### Integration Testing  
- End-to-end workflow testing
- Cross-platform compatibility testing
- Performance benchmarking

### User Acceptance Testing
- Feature parity verification
- Usability testing with existing users
- Migration feedback collection

## Success Metrics

### Functional Parity
- [ ] All existing features working
- [ ] Performance equal or better than Textual version
- [ ] No regression in functionality

### User Experience
- [ ] Improved native OS integration
- [ ] Better accessibility support
- [ ] Enhanced visual design

### Technical Quality
- [ ] Maintainable code architecture
- [ ] Comprehensive test coverage
- [ ] Clear documentation

## Conclusion

The OpenHCS Textual TUI is exceptionally well-architected for this migration. The excellent separation of concerns means that **70-80% of the business logic can be preserved**, making this a **moderately complex** rather than a complete rewrite.

The migration will result in:
- **Better OS integration** and native look/feel
- **Improved performance** for complex UIs
- **Enhanced accessibility** support
- **Larger ecosystem** of Qt tools and widgets

**Recommendation:** Proceed with migration using the phased approach outlined above.

## Detailed Implementation Strategies

### Widget Migration Patterns

#### Pattern 1: Simple Widget Replacement
```python
# Textual → PyQt6 Direct Mapping
SIMPLE_WIDGETS = {
    'Static': 'QLabel',
    'Button': 'QPushButton',
    'Input': 'QLineEdit',
    'Checkbox': 'QCheckBox',
}

# Migration helper function
def migrate_simple_widget(textual_widget_code):
    """Convert simple Textual widget to PyQt6 equivalent"""
    for textual, pyqt in SIMPLE_WIDGETS.items():
        textual_widget_code = textual_widget_code.replace(textual, pyqt)
    return textual_widget_code
```

#### Pattern 2: Container Migration
```python
# Textual Container Pattern
class TextualWidget(Widget):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Title")
            with Horizontal():
                yield Button("OK")
                yield Button("Cancel")

# PyQt6 Equivalent
class PyQtWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        layout.addWidget(QLabel("Title"))

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addWidget(QPushButton("OK"))
        button_layout.addWidget(QPushButton("Cancel"))

        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        self.setLayout(layout)
```

#### Pattern 3: Event Handling Migration
```python
# Textual Event Pattern
@on(Button.Pressed, "#save_btn")
def handle_save(self, event):
    self.save_data()

# PyQt6 Signal/Slot Pattern
def setup_connections(self):
    self.save_btn.clicked.connect(self.handle_save)

def handle_save(self):
    self.save_data()  # Same business logic
```

### Service Layer Integration

#### Dependency Injection Pattern
```python
class PyQtServiceMixin:
    """Mixin to inject existing services into PyQt widgets"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_registry = FunctionRegistryService()
        self.pattern_manager = PatternDataManager()
        self.file_service = PatternFileService(self)

class PlateManagerWidget(QWidget, PyQtServiceMixin):
    def __init__(self):
        super().__init__()
        # Services automatically available via mixin
        self.setup_ui()
```

#### Async Operation Handling
```python
class AsyncWorker(QThread):
    """Handle async operations that were native in Textual"""

    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, async_func, *args, **kwargs):
        super().__init__()
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Convert async function to sync for QThread
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.async_func(*self.args, **self.kwargs)
            )
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Usage in widget
def load_data_async(self):
    worker = AsyncWorker(self.file_service.load_pattern_from_file, file_path)
    worker.result_ready.connect(self.on_data_loaded)
    worker.error_occurred.connect(self.on_error)
    worker.start()
```

### Advanced Migration Techniques

#### State Management Bridge
```python
class ReactiveProperty:
    """Bridge Textual reactive variables to PyQt properties"""

    def __init__(self, initial_value, signal=None):
        self._value = initial_value
        self._signal = signal

    def __get__(self, obj, objtype=None):
        return self._value

    def __set__(self, obj, value):
        old_value = self._value
        self._value = value
        if self._signal and old_value != value:
            self._signal.emit(value)

class ReactiveWidget(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._status = ReactiveProperty("Ready", self.status_changed)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
```

#### CSS to Qt Stylesheet Converter
```python
def convert_textual_css_to_qt(css_content):
    """Convert Textual CSS-like styles to Qt stylesheets"""

    # Mapping of Textual properties to Qt properties
    PROPERTY_MAPPING = {
        'color': 'color',
        'background': 'background-color',
        'border': 'border',
        'margin': 'margin',
        'padding': 'padding',
        'width': 'min-width',
        'height': 'min-height',
    }

    # Convert selectors
    SELECTOR_MAPPING = {
        '#': '#',  # ID selectors work the same
        '.': '.',  # Class selectors work the same
        'Button': 'QPushButton',
        'Input': 'QLineEdit',
        'Static': 'QLabel',
    }

    qt_stylesheet = css_content
    for textual_prop, qt_prop in PROPERTY_MAPPING.items():
        qt_stylesheet = qt_stylesheet.replace(textual_prop, qt_prop)

    for textual_sel, qt_sel in SELECTOR_MAPPING.items():
        qt_stylesheet = qt_stylesheet.replace(textual_sel, qt_sel)

    return qt_stylesheet
```

## Automation Tools & Scripts

### Migration Assistant Script
```python
#!/usr/bin/env python3
"""
OpenHCS TUI to PyQt6 Migration Assistant

Automates common migration tasks:
- Widget class conversion
- Event handler migration
- Import statement updates
- Basic layout generation
"""

import ast
import re
from pathlib import Path
from typing import Dict, List

class MigrationAssistant:
    def __init__(self):
        self.widget_mapping = {
            'Widget': 'QWidget',
            'Button': 'QPushButton',
            'Static': 'QLabel',
            'Input': 'QLineEdit',
            'Checkbox': 'QCheckBox',
            'Select': 'QComboBox',
            'Container': 'QWidget',
            'Vertical': 'QVBoxLayout',
            'Horizontal': 'QHBoxLayout',
        }

        self.event_mapping = {
            'Button.Pressed': 'clicked',
            'Input.Changed': 'textChanged',
            'Checkbox.Changed': 'toggled',
            'Select.Changed': 'currentTextChanged',
        }

    def migrate_file(self, textual_file: Path) -> str:
        """Migrate a single Textual widget file to PyQt6"""
        content = textual_file.read_text()

        # Parse AST to understand structure
        tree = ast.parse(content)

        # Extract widget classes
        widget_classes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and
            any(base.id == 'Widget' for base in node.bases if isinstance(base, ast.Name))
        ]

        migrated_content = content

        for widget_class in widget_classes:
            migrated_content = self._migrate_widget_class(migrated_content, widget_class)

        # Update imports
        migrated_content = self._update_imports(migrated_content)

        return migrated_content

    def _migrate_widget_class(self, content: str, widget_class: ast.ClassDef) -> str:
        """Migrate a single widget class"""
        class_name = widget_class.name

        # Find compose method
        compose_method = None
        for node in widget_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'compose':
                compose_method = node
                break

        if compose_method:
            # Generate PyQt6 equivalent
            pyqt_init = self._generate_pyqt_init(compose_method)

            # Replace compose method with __init__
            pattern = rf'def compose\(self\).*?(?=\n    def|\nclass|\Z)'
            replacement = pyqt_init
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        return content

    def _generate_pyqt_init(self, compose_method: ast.FunctionDef) -> str:
        """Generate PyQt6 __init__ method from Textual compose method"""
        init_code = [
            "def __init__(self):",
            "    super().__init__()",
            "    self.setup_ui()",
            "    self.connect_signals()",
            "",
            "def setup_ui(self):",
            "    layout = QVBoxLayout()",
        ]

        # Analyze compose method body to generate layout code
        for stmt in compose_method.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                widget_code = self._convert_yield_to_layout(stmt.value.value)
                init_code.append(f"    {widget_code}")

        init_code.extend([
            "    self.setLayout(layout)",
            "",
            "def connect_signals(self):",
            "    # TODO: Add signal connections",
            "    pass"
        ])

        return '\n'.join(init_code)

    def _convert_yield_to_layout(self, yield_value: ast.AST) -> str:
        """Convert Textual yield statement to PyQt layout code"""
        if isinstance(yield_value, ast.Call):
            func_name = yield_value.func.id if isinstance(yield_value.func, ast.Name) else str(yield_value.func)

            if func_name in self.widget_mapping:
                qt_widget = self.widget_mapping[func_name]

                # Extract arguments
                args = []
                for arg in yield_value.args:
                    if isinstance(arg, ast.Str):
                        args.append(f'"{arg.s}"')
                    elif isinstance(arg, ast.Constant):
                        args.append(f'"{arg.value}"')

                arg_str = ', '.join(args) if args else ''
                return f"layout.addWidget({qt_widget}({arg_str}))"

        return "# TODO: Convert complex widget"

    def _update_imports(self, content: str) -> str:
        """Update import statements for PyQt6"""
        # Remove Textual imports
        content = re.sub(r'from textual.*\n', '', content)

        # Add PyQt6 imports
        pyqt_imports = [
            "from PyQt6.QtWidgets import *",
            "from PyQt6.QtCore import *",
            "from PyQt6.QtGui import *",
            ""
        ]

        # Insert at top after existing imports
        import_section = '\n'.join(pyqt_imports)
        content = import_section + content

        return content

# Usage
if __name__ == "__main__":
    assistant = MigrationAssistant()

    # Migrate all widget files
    textual_widgets_dir = Path("openhcs/textual_tui/widgets")
    pyqt_widgets_dir = Path("openhcs/pyqt_gui/widgets")
    pyqt_widgets_dir.mkdir(parents=True, exist_ok=True)

    for widget_file in textual_widgets_dir.glob("*.py"):
        if widget_file.name.startswith("__"):
            continue

        print(f"Migrating {widget_file.name}...")
        migrated_content = assistant.migrate_file(widget_file)

        output_file = pyqt_widgets_dir / widget_file.name
        output_file.write_text(migrated_content)
        print(f"  → {output_file}")
```

### Testing Framework Integration
```python
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for testing"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()

class TestParameterFormManager:
    """Test migrated ParameterFormManager with PyQt6"""

    def test_widget_creation(self, qapp):
        """Test that widgets are created correctly"""
        from openhcs.pyqt_gui.shared.parameter_form_manager import ParameterFormManager

        parameters = {"test_param": "default_value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(parameters, parameter_types, "test")
        widget = manager._create_widget_for_type(str, "test_value")

        assert widget is not None
        assert hasattr(widget, 'text')  # QLineEdit method
        assert widget.text() == "test_value"

    def test_signal_connections(self, qapp):
        """Test that signals are connected properly"""
        # Test signal/slot connections
        pass

# Performance comparison tests
class TestPerformanceComparison:
    """Compare performance between Textual and PyQt6 versions"""

    def test_widget_creation_speed(self, qapp):
        """Compare widget creation performance"""
        import time

        # Time PyQt6 widget creation
        start = time.time()
        for _ in range(1000):
            widget = QPushButton("Test")
        pyqt_time = time.time() - start

        print(f"PyQt6 widget creation: {pyqt_time:.4f}s for 1000 widgets")

        # Could compare with Textual if both are available
        assert pyqt_time < 1.0  # Should be fast
```

## Migration Checklist

### Pre-Migration Setup
- [ ] Install PyQt6 and development tools
- [ ] Setup project structure
- [ ] Create migration branch in version control
- [ ] Backup existing Textual implementation

### Phase 1 Checklist
- [ ] Service layer copied and tested
- [ ] SignatureAnalyzer migrated
- [ ] ParameterFormManager adapted for PyQt6
- [ ] TypedWidgetFactory implemented
- [ ] Unit tests passing

### Phase 2 Checklist
- [ ] FunctionPaneWidget migrated
- [ ] PlateManagerWidget migrated
- [ ] PipelineEditorWidget migrated
- [ ] Event handling implemented
- [ ] Basic layouts working

### Phase 3 Checklist
- [ ] Main window implemented
- [ ] Menu system created
- [ ] Docking system working
- [ ] Window management functional
- [ ] Settings dialog implemented

### Phase 4 Checklist
- [ ] All features migrated
- [ ] Performance optimized
- [ ] Accessibility implemented
- [ ] Documentation updated
- [ ] User testing completed

### Post-Migration
- [ ] Feature parity verified
- [ ] Performance benchmarked
- [ ] User feedback collected
- [ ] Migration documentation complete
- [ ] Deployment ready

## Conclusion

This comprehensive migration plan provides a structured approach to converting the OpenHCS Textual TUI to PyQt6 while preserving the excellent architectural decisions already made. The phased approach minimizes risk while the automation tools accelerate the migration process.

The key to success is leveraging the existing service layer architecture and focusing migration efforts on the presentation layer where the most value can be gained from PyQt6's native capabilities.

## Scientific Application Specific Patterns

### Data Visualization Integration

OpenHCS processes large image datasets, so PyQt6's data visualization capabilities are crucial:

#### Matplotlib Integration
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class ImageDisplayWidget(QWidget):
    """Enhanced image display with matplotlib integration"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Controls
        controls_layout = QHBoxLayout()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)

        controls_layout.addWidget(QLabel("Colormap:"))
        controls_layout.addWidget(self.colormap_combo)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)

        self.setLayout(layout)

    def display_image(self, image_array, title="Image"):
        """Display image with OpenHCS integration"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(image_array, cmap=self.colormap_combo.currentText())
        ax.set_title(title)

        # Add colorbar
        self.figure.colorbar(im, ax=ax)

        self.canvas.draw()

    def update_colormap(self):
        """Update colormap when selection changes"""
        # Re-render with new colormap
        if hasattr(self, '_current_image'):
            self.display_image(self._current_image)
```

#### PyQtGraph for High-Performance Visualization
```python
import pyqtgraph as pg
from PyQt6.QtWidgets import *

class HighPerformanceImageWidget(QWidget):
    """High-performance image display using PyQtGraph"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # PyQtGraph ImageView for fast image display
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        # Histogram widget for intensity analysis
        self.histogram = pg.HistogramLUTItem()
        self.image_view.addItem(self.histogram)

        self.setLayout(layout)

    def display_image_stack(self, image_stack):
        """Display 3D image stack with time/Z navigation"""
        # PyQtGraph handles 3D data natively
        self.image_view.setImage(image_stack, axes={'t': 0, 'x': 1, 'y': 2})

        # Auto-levels and histogram
        self.image_view.autoLevels()
```

### Large Dataset Handling

#### Lazy Loading with QAbstractItemModel
```python
class PlateDataModel(QAbstractItemModel):
    """Efficient model for large plate datasets"""

    def __init__(self, plate_path):
        super().__init__()
        self.plate_path = plate_path
        self._cache = {}
        self._metadata = self._load_metadata()

    def _load_metadata(self):
        """Load only metadata, not actual images"""
        # Use OpenHCS FileManager to get metadata
        from openhcs.io.file_manager import FileManager
        fm = FileManager()
        return fm.get_plate_metadata(self.plate_path)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Lazy load data only when requested"""
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            # Return metadata for display
            item = self._metadata[index.row()]
            return item.get('filename', 'Unknown')

        elif role == Qt.ItemDataRole.UserRole:
            # Lazy load actual image data
            if index.row() not in self._cache:
                item = self._metadata[index.row()]
                # Load image using OpenHCS
                image = self._load_image(item['path'])
                self._cache[index.row()] = image

            return self._cache[index.row()]

        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self._metadata)

    def columnCount(self, parent=QModelIndex()):
        return 1
```

#### Background Processing with QThread
```python
class ImageProcessingWorker(QThread):
    """Background image processing for OpenHCS pipelines"""

    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, pipeline_steps, input_data):
        super().__init__()
        self.pipeline_steps = pipeline_steps
        self.input_data = input_data

    def run(self):
        """Execute OpenHCS pipeline in background"""
        try:
            from openhcs.processing.pipeline_executor import PipelineExecutor

            executor = PipelineExecutor()
            total_steps = len(self.pipeline_steps)

            for i, step in enumerate(self.pipeline_steps):
                # Execute step
                result = executor.execute_step(step, self.input_data)
                self.input_data = result

                # Update progress
                progress = int((i + 1) / total_steps * 100)
                self.progress_updated.emit(progress)

            self.result_ready.emit(self.input_data)

        except Exception as e:
            self.error_occurred.emit(str(e))

class PipelineExecutorWidget(QWidget):
    """Widget for executing OpenHCS pipelines with progress"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Execute button
        self.execute_btn = QPushButton("Execute Pipeline")
        self.execute_btn.clicked.connect(self.execute_pipeline)
        layout.addWidget(self.execute_btn)

        self.setLayout(layout)

    def execute_pipeline(self):
        """Execute pipeline in background thread"""
        self.execute_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # Create worker thread
        self.worker = ImageProcessingWorker(
            self.pipeline_steps,
            self.input_data
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.on_pipeline_complete)
        self.worker.error_occurred.connect(self.on_pipeline_error)

        # Start processing
        self.worker.start()

    def on_pipeline_complete(self, result):
        """Handle pipeline completion"""
        self.execute_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        # Display result
        self.display_result(result)

    def on_pipeline_error(self, error_msg):
        """Handle pipeline errors"""
        self.execute_btn.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", error_msg)
```

### Memory Management for Large Images

#### Efficient Image Caching
```python
from functools import lru_cache
import weakref

class ImageCache:
    """Memory-efficient image caching for OpenHCS"""

    def __init__(self, max_size_mb=1024):
        self.max_size_mb = max_size_mb
        self._cache = {}
        self._size_tracker = {}
        self._access_order = []

    def get_image(self, image_path):
        """Get image with automatic memory management"""
        if image_path in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(image_path)
            self._access_order.append(image_path)
            return self._cache[image_path]

        # Load image
        image = self._load_image(image_path)
        self._add_to_cache(image_path, image)

        return image

    def _add_to_cache(self, path, image):
        """Add image to cache with size management"""
        import sys
        image_size_mb = sys.getsizeof(image) / (1024 * 1024)

        # Remove old images if needed
        while (sum(self._size_tracker.values()) + image_size_mb > self.max_size_mb
               and self._access_order):
            oldest_path = self._access_order.pop(0)
            if oldest_path in self._cache:
                del self._cache[oldest_path]
                del self._size_tracker[oldest_path]

        # Add new image
        self._cache[path] = image
        self._size_tracker[path] = image_size_mb
        self._access_order.append(path)

class MemoryEfficientImageWidget(QWidget):
    """Image widget with memory management"""

    def __init__(self):
        super().__init__()
        self.image_cache = ImageCache(max_size_mb=512)  # 512MB cache
        self.setup_ui()

    def load_image(self, image_path):
        """Load image with caching"""
        try:
            image = self.image_cache.get_image(image_path)
            self.display_image(image)
        except MemoryError:
            QMessageBox.warning(
                self,
                "Memory Warning",
                "Image too large for available memory"
            )
```

### Advanced PyQt6 Features for Scientific Apps

#### Custom Widgets for Scientific Data
```python
class ParameterSliderWidget(QWidget):
    """Custom slider widget for scientific parameters"""

    value_changed = pyqtSignal(float)

    def __init__(self, param_name, min_val, max_val, default_val, step=0.1):
        super().__init__()
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.setup_ui(default_val)

    def setup_ui(self, default_val):
        layout = QHBoxLayout()

        # Parameter label
        self.label = QLabel(self.param_name)
        layout.addWidget(self.label)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int((self.max_val - self.min_val) / self.step))
        self.slider.setValue(int((default_val - self.min_val) / self.step))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        # Value display
        self.value_label = QLabel(f"{default_val:.3f}")
        self.value_label.setMinimumWidth(60)
        layout.addWidget(self.value_label)

        # Direct input
        self.value_input = QLineEdit(str(default_val))
        self.value_input.setMaximumWidth(80)
        self.value_input.editingFinished.connect(self._on_input_changed)
        layout.addWidget(self.value_input)

        self.setLayout(layout)

    def _on_slider_changed(self, slider_value):
        """Handle slider changes"""
        actual_value = self.min_val + (slider_value * self.step)
        self.value_label.setText(f"{actual_value:.3f}")
        self.value_input.setText(str(actual_value))
        self.value_changed.emit(actual_value)

    def _on_input_changed(self):
        """Handle direct input changes"""
        try:
            value = float(self.value_input.text())
            value = max(self.min_val, min(self.max_val, value))

            slider_pos = int((value - self.min_val) / self.step)
            self.slider.setValue(slider_pos)
            self.value_changed.emit(value)

        except ValueError:
            # Reset to current slider value
            current_value = self.min_val + (self.slider.value() * self.step)
            self.value_input.setText(str(current_value))

class ScientificParameterForm(QWidget):
    """Form widget optimized for scientific parameters"""

    def __init__(self, parameters_config):
        super().__init__()
        self.parameters_config = parameters_config
        self.parameter_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create parameter widgets based on config
        for param_name, config in self.parameters_config.items():
            if config['type'] == 'float':
                widget = ParameterSliderWidget(
                    param_name,
                    config['min'],
                    config['max'],
                    config['default'],
                    config.get('step', 0.1)
                )
            elif config['type'] == 'int':
                widget = QSpinBox()
                widget.setMinimum(config['min'])
                widget.setMaximum(config['max'])
                widget.setValue(config['default'])
            elif config['type'] == 'bool':
                widget = QCheckBox(param_name)
                widget.setChecked(config['default'])
            else:
                widget = QLineEdit(str(config['default']))

            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)

        self.setLayout(layout)

    def get_parameters(self):
        """Get current parameter values"""
        values = {}
        for param_name, widget in self.parameter_widgets.items():
            if isinstance(widget, ParameterSliderWidget):
                # Get value from custom widget
                values[param_name] = float(widget.value_input.text())
            elif isinstance(widget, QSpinBox):
                values[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[param_name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                values[param_name] = widget.text()

        return values
```

## Performance Optimization Strategies

### GPU Integration with PyQt6
```python
class GPUAcceleratedWidget(QWidget):
    """Widget with GPU acceleration for image processing"""

    def __init__(self):
        super().__init__()
        self.setup_gpu_context()
        self.setup_ui()

    def setup_gpu_context(self):
        """Initialize GPU context for processing"""
        try:
            import cupy as cp
            self.gpu_available = True
            self.gpu_context = cp.cuda.Device(0)
        except ImportError:
            self.gpu_available = False
            print("GPU acceleration not available")

    def process_image_gpu(self, image_array):
        """Process image on GPU if available"""
        if self.gpu_available:
            import cupy as cp
            with self.gpu_context:
                gpu_image = cp.asarray(image_array)
                # Perform GPU processing
                processed = self._gpu_processing_function(gpu_image)
                return cp.asnumpy(processed)
        else:
            # Fallback to CPU
            return self._cpu_processing_function(image_array)
```

### Memory Profiling Integration
```python
class MemoryProfiledWidget(QWidget):
    """Widget with built-in memory profiling"""

    def __init__(self):
        super().__init__()
        self.memory_tracker = {}
        self.setup_ui()

    def track_memory_usage(self, operation_name):
        """Decorator for tracking memory usage"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import psutil
                import os

                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

                result = func(*args, **kwargs)

                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = memory_after - memory_before

                self.memory_tracker[operation_name] = {
                    'before': memory_before,
                    'after': memory_after,
                    'delta': memory_delta
                }

                print(f"{operation_name}: {memory_delta:.2f} MB")
                return result
            return wrapper
        return decorator
```

This comprehensive migration plan provides the technical depth and practical guidance needed to successfully convert OpenHCS from Textual to PyQt6 while leveraging the excellent existing architecture.

## Immediate Next Steps

### 1. Proof of Concept (Week 1)
Create a minimal PyQt6 application that demonstrates the core concepts:

```bash
# Setup development environment
pip install PyQt6 PyQt6-tools pyqtgraph matplotlib
pip install pytest-qt  # For testing

# Create proof of concept structure
mkdir openhcs_pyqt_poc
cd openhcs_pyqt_poc
```

**POC Goals:**
- [ ] Basic PyQt6 window with menu bar
- [ ] Migrate one simple widget (e.g., ParameterFormManager)
- [ ] Integrate one existing service (e.g., FunctionRegistryService)
- [ ] Demonstrate signal/slot event handling
- [ ] Show matplotlib integration for image display

### 2. Architecture Validation (Week 2)
Validate the migration approach with key stakeholders:

- [ ] Demo POC to development team
- [ ] Validate UI/UX design decisions
- [ ] Confirm performance requirements
- [ ] Review migration timeline and resource allocation
- [ ] Get approval for full migration

### 3. Development Environment Setup
```bash
# Install development tools
pip install PyQt6-tools  # Includes Qt Designer
pip install black isort  # Code formatting
pip install mypy         # Type checking
pip install sphinx       # Documentation

# Setup pre-commit hooks for code quality
pip install pre-commit
```

## Resource Links & References

### Essential PyQt6 Documentation
- **Official PyQt6 Documentation**: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- **Qt6 Documentation**: https://doc.qt.io/qt-6/
- **PyQt6 Tutorial**: https://www.pythonguis.com/pyqt6-tutorial/

### Scientific Application Examples
- **PyQtGraph Documentation**: https://pyqtgraph.readthedocs.io/
- **Matplotlib Qt Integration**: https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
- **Scientific PyQt Examples**: https://github.com/pyqtgraph/pyqtgraph/tree/master/examples

### Migration Tools & Libraries
- **Qt Designer**: Visual UI design tool
- **PyQt6-tools**: Command line tools for PyQt6
- **pytest-qt**: Testing framework for Qt applications
- **QDarkStyle**: Dark theme for PyQt applications

### Performance & Optimization
- **Qt Performance Tips**: https://doc.qt.io/qt-6/qtquick-performance.html
- **Memory Management**: https://doc.qt.io/qt-6/memory.html
- **Threading in Qt**: https://doc.qt.io/qt-6/thread-basics.html

## Decision Matrix: PyQt6 vs Alternatives

| Criteria | PyQt6 | PySide6 | Tkinter | Kivy | Score |
|----------|-------|---------|---------|------|-------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | PyQt6 ✓ |
| **Native Look** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | PyQt6 ✓ |
| **Scientific Ecosystem** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | PyQt6 ✓ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | PyQt6 ✓ |
| **Community** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | PyQt6 ✓ |
| **License** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | PySide6 |
| **Migration Effort** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | PyQt6 ✓ |

**Recommendation**: PyQt6 wins on technical merits. PySide6 is equivalent technically but has better licensing (LGPL vs GPL).

## Risk Mitigation Strategies

### Technical Risks

**Risk**: Complex widget interactions don't translate well
- **Mitigation**: Create adapter layer for complex state management
- **Fallback**: Implement simplified interaction patterns initially

**Risk**: Performance degradation compared to Textual
- **Mitigation**: Profile early and optimize bottlenecks
- **Fallback**: Hybrid approach keeping performance-critical parts in Textual

**Risk**: Memory usage increases significantly
- **Mitigation**: Implement lazy loading and caching strategies
- **Fallback**: Add memory usage monitoring and warnings

### Project Risks

**Risk**: Timeline overruns due to complexity
- **Mitigation**: Phased approach with clear milestones
- **Fallback**: Reduce scope for initial release

**Risk**: User resistance to UI changes
- **Mitigation**: Involve users in design process, provide training
- **Fallback**: Maintain both UIs temporarily

**Risk**: Developer productivity loss during transition
- **Mitigation**: Comprehensive documentation and training
- **Fallback**: Dedicated migration team

## Success Metrics & KPIs

### Technical Metrics
- **Performance**: Response time ≤ current Textual implementation
- **Memory**: Memory usage ≤ 150% of current implementation
- **Stability**: Crash rate < 0.1% of operations
- **Test Coverage**: ≥ 80% code coverage

### User Experience Metrics
- **Feature Parity**: 100% of current features working
- **User Satisfaction**: ≥ 4/5 rating from existing users
- **Learning Curve**: New users productive within 2 hours
- **Accessibility**: WCAG 2.1 AA compliance

### Development Metrics
- **Code Quality**: Maintainability index ≥ 70
- **Documentation**: All public APIs documented
- **Build Time**: CI/CD pipeline ≤ 10 minutes
- **Bug Rate**: ≤ 1 bug per 1000 lines of code

## Final Recommendations

### Go/No-Go Decision Criteria

**GO if:**
- ✅ POC demonstrates successful service layer integration
- ✅ Performance meets or exceeds current implementation
- ✅ Development team has PyQt6 expertise or training plan
- ✅ Timeline and resources are adequate (6-8 weeks)
- ✅ User feedback on POC is positive

**NO-GO if:**
- ❌ POC reveals fundamental architectural incompatibilities
- ❌ Performance is significantly worse than Textual
- ❌ Resource constraints prevent proper implementation
- ❌ Critical dependencies are missing or incompatible

### Alternative Approaches

If full migration is not feasible:

1. **Hybrid Approach**: Keep Textual for core functionality, add PyQt6 for specific features
2. **Gradual Migration**: Migrate one widget at a time over longer period
3. **Web-based Alternative**: Consider Electron or web-based UI instead
4. **Enhanced Textual**: Improve current Textual implementation instead

### Long-term Vision

The PyQt6 migration positions OpenHCS for:
- **Better Integration** with scientific Python ecosystem
- **Enhanced Visualization** capabilities for large datasets
- **Improved Accessibility** for diverse user base
- **Future Extensibility** with Qt's rich widget ecosystem
- **Professional Appearance** for commercial deployments

This migration represents a strategic investment in OpenHCS's future as a professional scientific application platform.
