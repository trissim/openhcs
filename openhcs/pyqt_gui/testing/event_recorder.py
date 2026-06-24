"""
GUI Event Recorder for Test Generation

Records user interactions with the PyQt6 GUI and generates pytest-qt test code
that can replay the exact sequence of actions.

Usage:
    # Start recording
    python -m openhcs.pyqt_gui.launch --record-test my_workflow_test

    # Interact with GUI normally
    # When done, close the application

    # Generated test will be saved to:
    # tests/pyqt_gui/recorded/test_my_workflow_test.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from PyQt6.QtCore import QObject, QEvent, Qt, QTimer
from PyQt6.QtWidgets import QWidget, QApplication, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox


@dataclass
class RecordedEvent:
    """Represents a single recorded GUI event."""
    timestamp: float
    event_type: str
    widget_path: str
    widget_type: str
    action: str
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_pytest_code(self, indent: int = 1) -> str:
        """Generate pytest-qt code for this event."""
        ind = "    " * indent

        if self.action == "click":
            # Use _wait_for_gui() for robust timing that works on slow machines
            return (
                f"{ind}qtbot.mouseClick({self.widget_path}, Qt.MouseButton.LeftButton)\n"
                f"{ind}_wait_for_gui(TIMING.ACTION_DELAY)"
            )

        elif self.action == "setText":
            # Use _wait_for_gui() for robust timing
            return (
                f"{ind}qtbot.keyClicks({self.widget_path}, {repr(self.value)})\n"
                f"{ind}_wait_for_gui(TIMING.ACTION_DELAY)"
            )

        elif self.action == "setCurrentText":
            # Use _wait_for_gui() for robust timing
            return (
                f"{ind}{self.widget_path}.setCurrentText({repr(self.value)})\n"
                f"{ind}_wait_for_gui(TIMING.ACTION_DELAY)"
            )

        elif self.action == "setChecked":
            return (
                f"{ind}{self.widget_path}.setChecked({self.value})\n"
                f"{ind}_wait_for_gui(TIMING.ACTION_DELAY)"
            )

        elif self.action == "setValue":
            return (
                f"{ind}{self.widget_path}.setValue({self.value})\n"
                f"{ind}_wait_for_gui(TIMING.ACTION_DELAY)"
            )

        else:
            return f"{ind}# Unknown action: {self.action}"


class EventRecorder(QObject):
    """Records GUI events and generates pytest-qt test code."""
    
    def __init__(self, app: QApplication, test_name: str):
        super().__init__()
        self.app = app
        self.test_name = test_name
        self.events: List[RecordedEvent] = []
        self.widget_registry: Dict[int, str] = {}  # id(widget) -> variable name
        self.last_event_time = datetime.now().timestamp()
        self.recording = False
        
    def start_recording(self):
        """Start recording GUI events."""
        output_path = Path(f"tests/pyqt_gui/recorded/test_{self.test_name}.py")
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        print(f"🔴 Recording test: {self.test_name}")
        print(f"   Output file: {output_path.absolute()}")
        print("   Interact with the GUI normally. Close the app when done.")
        sys.stdout.flush()

        self.recording = True
        self.app.installEventFilter(self)
        
    def stop_recording(self):
        """Stop recording and generate test code."""
        print("🛑 stop_recording() called")
        sys.stdout.flush()

        self.recording = False
        self.app.removeEventFilter(self)
        print(f"⏹️  Recording stopped. {len(self.events)} events captured.")
        sys.stdout.flush()

        # Generate test code immediately
        try:
            print("📝 Generating test code...")
            sys.stdout.flush()
            self.generate_test_code()
            print("✅ Test code generation complete")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Error generating test code: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter and record relevant GUI events."""
        if not self.recording:
            return False
            
        # Only record events from widgets
        if not isinstance(watched, QWidget):
            return False
            
        # Record different event types
        if event.type() == QEvent.Type.MouseButtonPress:
            self._record_click(watched)
            
        elif event.type() == QEvent.Type.KeyPress:
            # We'll capture text changes instead of individual key presses
            pass
            
        # Don't block the event
        return False
    
    def _record_click(self, widget: QWidget):
        """Record a mouse click event."""
        if isinstance(widget, QPushButton):
            widget_path = self._get_widget_path(widget)
            self._add_event(
                event_type="MouseClick",
                widget_path=widget_path,
                widget_type="QPushButton",
                action="click",
                metadata={"text": widget.text()}
            )
    
    def _record_text_change(self, widget: QLineEdit, text: str):
        """Record text input."""
        widget_path = self._get_widget_path(widget)
        self._add_event(
            event_type="TextInput",
            widget_path=widget_path,
            widget_type="QLineEdit",
            action="setText",
            value=text
        )
    
    def _record_combo_change(self, widget: QComboBox, text: str):
        """Record combo box selection."""
        widget_path = self._get_widget_path(widget)
        self._add_event(
            event_type="ComboSelection",
            widget_path=widget_path,
            widget_type="QComboBox",
            action="setCurrentText",
            value=text
        )
    
    def _record_checkbox_change(self, widget: QCheckBox, checked: bool):
        """Record checkbox state change."""
        widget_path = self._get_widget_path(widget)
        self._add_event(
            event_type="CheckboxToggle",
            widget_path=widget_path,
            widget_type="QCheckBox",
            action="setChecked",
            value=checked
        )
    
    def _get_widget_path(self, widget: QWidget) -> str:
        """Get a unique path to the widget for code generation."""
        # Try to find widget by object name
        if widget.objectName():
            return f'main_window.findChild(QWidget, "{widget.objectName()}")'
        
        # Try to find by text (for buttons)
        if isinstance(widget, QPushButton) and widget.text():
            return f'WidgetFinder.find_button_by_text(main_window, ["{widget.text().lower()}"])'
        
        # Build path through parent hierarchy
        path_parts = []
        current = widget
        while current:
            if hasattr(current, 'objectName') and current.objectName():
                path_parts.insert(0, f'findChild(QWidget, "{current.objectName()}")')
                break
            current = current.parentWidget()
        
        if path_parts:
            return f"main_window.{'.'.join(path_parts)}"
        
        # Fallback: use widget registry
        widget_id = id(widget)
        if widget_id not in self.widget_registry:
            var_name = f"widget_{len(self.widget_registry)}"
            self.widget_registry[widget_id] = var_name
        
        return self.widget_registry[widget_id]
    
    def _add_event(self, event_type: str, widget_path: str, widget_type: str,
                   action: str, value: Any = None, metadata: Dict = None):
        """Add a recorded event."""
        now = datetime.now().timestamp()

        # Don't record timing delays - use smart waits instead
        # This prevents timing issues on slower machines
        # The generated test will use qtbot.waitUntil() for proper synchronization

        # Add the actual event
        self.events.append(RecordedEvent(
            timestamp=now,
            event_type=event_type,
            widget_path=widget_path,
            widget_type=widget_type,
            action=action,
            value=value,
            metadata=metadata or {}
        ))

        self.last_event_time = now
        print(f"  📝 Recorded: {event_type} on {widget_type} ({action})")
    
    def generate_test_code(self, output_path: Optional[Path] = None) -> str:
        """Generate pytest-qt test code from recorded events."""
        if output_path is None:
            output_path = Path(f"tests/pyqt_gui/recorded/test_{self.test_name}.py")

        # Make path absolute
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        
        # Generate imports
        imports = [
            "import pytest",
            "import os",
            "from PyQt6.QtCore import Qt",
            "from PyQt6.QtWidgets import QWidget, QPushButton",
            "",
            "# Skip in CPU-only mode",
            "if os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true':",
            "    pytest.skip('PyQt6 GUI tests skipped in CPU-only mode', allow_module_level=True)",
            "",
            "from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, get_default_pyqt_gui_config",
            "from openhcs.pyqt_gui.main import OpenHCSMainWindow",
            "from tests.pyqt_gui.integration.test_end_to_end_workflow_foundation import (",
            "    WidgetFinder,",
            "    TimingConfig,",
            "    _wait_for_gui,",
            ")",
            "",
            "# Use environment-configurable timing",
            "TIMING = TimingConfig.from_environment()",
            "",
        ]
        
        # Generate test function
        test_func = [
            f"def test_{self.test_name}(qtbot):",
            '    """',
            f'    Auto-generated test from GUI recording: {self.test_name}',
            f'    Recorded on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'    Total events: {len(self.events)}',
            '    """',
            "    # Create main window",
            "    main_window = OpenHCSMainWindow(runtime_context=PyQtGuiRuntimeContext(get_default_pyqt_gui_config()))",
            "    qtbot.addWidget(main_window)",
            "    main_window.show()",
            "    qtbot.wait(1500)",
            "",
            "    # Replay recorded interactions",
        ]
        
        # Add recorded events
        for event in self.events:
            test_func.append(event.to_pytest_code(indent=1))
        
        # Combine all parts
        code = "\n".join(imports + test_func)
        
        # Write to file
        print(f"📁 Creating directory: {output_path.parent}")
        sys.stdout.flush()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Writing test file: {output_path}")
        sys.stdout.flush()
        output_path.write_text(code)

        print(f"✅ Test code generated: {output_path.absolute()}")
        print(f"   Total events recorded: {len(self.events)}")
        print(f"   Run with: pytest {output_path} -v")
        sys.stdout.flush()

        return code


def install_recorder(app: QApplication, test_name: str) -> EventRecorder:
    """Install event recorder on the application."""
    recorder = EventRecorder(app, test_name)
    
    # Connect to widget signals for better event capture
    def connect_widget_signals(widget: QWidget):
        """Recursively connect to widget signals."""
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda text: recorder._record_text_change(widget, text))
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(lambda text: recorder._record_combo_change(widget, text))
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda checked: recorder._record_checkbox_change(widget, checked))
        
        # Recursively connect children
        for child in widget.findChildren(QWidget):
            connect_widget_signals(child)
    
    # Connect to all existing widgets
    for widget in app.topLevelWidgets():
        connect_widget_signals(widget)
    
    # Start recording
    recorder.start_recording()

    # Stop recording when app closes (generate_test_code is called inside stop_recording)
    app.aboutToQuit.connect(recorder.stop_recording)

    return recorder
