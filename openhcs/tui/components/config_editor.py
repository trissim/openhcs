"""
Clean Configuration Editor for OpenHCS TUI.

Properly separated concerns, no inheritance abuse, declarative architecture.
"""

import dataclasses
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, get_origin, get_args, Union as TypingUnion, Coroutine
from functools import singledispatch
from dataclasses import dataclass

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Dimension
from prompt_toolkit.layout.containers import Container
from prompt_toolkit.widgets import Box, Label, TextArea, RadioList, Checkbox

from prompt_toolkit.widgets import Button

logger = logging.getLogger(__name__)

@dataclass
class FieldSpec:
    """Specification for a dataclass field."""
    name: str
    label: str
    field_type: type
    actual_type: type
    current_value: Any
    default_value: Any
    is_optional: bool

class FieldIntrospector:
    """Pure field analysis - no UI concerns."""
    
    @staticmethod
    def analyze_dataclass(config_class: type, current_config: Any) -> List[FieldSpec]:
        """Extract field specifications from dataclass."""
        if not dataclasses.is_dataclass(config_class):
            raise ValueError(f"{config_class} is not a dataclass")
        
        specs = []
        fields = dataclasses.fields(config_class)
        
        for field in fields:
            if field.name.startswith('_'):
                continue
                
            spec = FieldIntrospector._analyze_field(field, current_config)
            specs.append(spec)
        
        return specs
    
    @staticmethod
    def _analyze_field(field: dataclasses.Field, config_instance: Any) -> FieldSpec:
        """Analyze single field."""
        # Extract type information
        field_type = field.type
        actual_type = field_type
        is_optional = get_origin(field_type) is TypingUnion and type(None) in get_args(field_type)
        
        if is_optional:
            actual_type = next((t for t in get_args(field_type) if t is not type(None)), actual_type)
        
        # Get values
        current_value = getattr(config_instance, field.name, None)
        default_value = None
        
        if field.default is not dataclasses.MISSING:
            default_value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_value = field.default_factory()
        
        # Create label
        label = field.name.replace('_', ' ').title()
        
        return FieldSpec(
            name=field.name,
            label=label,
            field_type=field_type,
            actual_type=actual_type,
            current_value=current_value,
            default_value=default_value,
            is_optional=is_optional
        )

class WidgetFactory:
    """Creates widgets based on field types - no business logic."""
    
    def __init__(self, on_change: Callable[[str, Any], None]):
        self.on_change = on_change
    
    def create_widget(self, spec: FieldSpec) -> Any:
        """Create appropriate widget for field type."""
        return self._dispatch_widget_creation(spec.actual_type, spec)
    
    def _dispatch_widget_creation(self, field_type: type, spec: FieldSpec) -> Any:
        """Type-based widget dispatch."""
        if field_type is bool:
            return self._create_bool_widget(spec)
        elif field_type is int:
            return self._create_int_widget(spec)
        elif field_type is float:
            return self._create_float_widget(spec)
        elif field_type is str:
            return self._create_string_widget(spec)
        elif field_type is Path:
            return self._create_path_widget(spec)
        elif self._is_enum(field_type):
            return self._create_enum_widget(spec)
        elif self._is_literal(field_type):
            return self._create_literal_widget(spec)
        elif dataclasses.is_dataclass(field_type):
            return self._create_nested_widget(spec)
        else:
            return self._create_fallback_widget(spec)
    
    def _create_bool_widget(self, spec: FieldSpec) -> Checkbox:
        widget = Checkbox(checked=bool(spec.current_value))
        
        # Hook into checkbox events
        original_handler = widget.control.mouse_handler
        def new_handler(mouse_event):
            result = original_handler(mouse_event)
            self.on_change(spec.name, widget.checked)
            return result
        
        widget.control.mouse_handler = new_handler
        return widget
    
    def _create_int_widget(self, spec: FieldSpec) -> TextArea:
        text = str(spec.current_value) if spec.current_value is not None else ""
        widget = TextArea(text=text, multiline=False, height=1)
        widget.buffer.on_text_changed += lambda buff: self._handle_text_change(spec.name, buff.text, int, spec.is_optional)
        return widget
    
    def _create_float_widget(self, spec: FieldSpec) -> TextArea:
        text = str(spec.current_value) if spec.current_value is not None else ""
        widget = TextArea(text=text, multiline=False, height=1)
        widget.buffer.on_text_changed += lambda buff: self._handle_text_change(spec.name, buff.text, float, spec.is_optional)
        return widget
    
    def _create_string_widget(self, spec: FieldSpec) -> TextArea:
        text = str(spec.current_value) if spec.current_value is not None else ""
        widget = TextArea(text=text, multiline=False, height=1)
        widget.buffer.on_text_changed += lambda buff: self.on_change(spec.name, buff.text or None)
        return widget
    
    def _create_path_widget(self, spec: FieldSpec) -> TextArea:
        text = str(spec.current_value) if spec.current_value is not None else ""
        widget = TextArea(text=text, multiline=False, height=1)
        widget.buffer.on_text_changed += lambda buff: self._handle_path_change(spec.name, buff.text, spec.is_optional)
        return widget
    
    def _create_enum_widget(self, spec: FieldSpec) -> RadioList:
        enum_type = spec.actual_type
        options = [(member, member.name) for member in enum_type]
        default = spec.current_value if spec.current_value in enum_type else None
        
        widget = RadioList(values=options, default=default)
        widget.handler = lambda val: self.on_change(spec.name, val)
        return widget
    
    def _create_literal_widget(self, spec: FieldSpec) -> RadioList:
        literal_values = get_args(spec.actual_type)
        options = [(val, str(val)) for val in literal_values]
        default = spec.current_value if spec.current_value in literal_values else None
        
        widget = RadioList(values=options, default=default)
        widget.handler = lambda val: self.on_change(spec.name, val)
        return widget
    
    def _create_nested_widget(self, spec: FieldSpec) -> Label:
        # Placeholder for nested dataclass editing
        return Label(f"[Nested] {spec.actual_type.__name__}")
    
    def _create_fallback_widget(self, spec: FieldSpec) -> TextArea:
        logger.warning(f"Unknown field type {spec.actual_type} for {spec.name}, using text widget")
        text = str(spec.current_value) if spec.current_value is not None else ""
        widget = TextArea(text=text, multiline=False, height=1)
        widget.buffer.on_text_changed += lambda buff: self.on_change(spec.name, buff.text or None)
        return widget
    
    def _handle_text_change(self, field_name: str, text: str, convert_type: type, is_optional: bool) -> None:
        """Handle text changes with type conversion."""
        if not text.strip() and is_optional:
            self.on_change(field_name, None)
            return
        
        try:
            value = convert_type(text) if text.strip() else (None if is_optional else convert_type())
            self.on_change(field_name, value)
        except ValueError:
            # Keep previous value on invalid input
            pass
    
    def _handle_path_change(self, field_name: str, text: str, is_optional: bool) -> None:
        """Handle Path field changes."""
        if not text.strip() and is_optional:
            self.on_change(field_name, None)
        else:
            self.on_change(field_name, Path(text) if text.strip() else None)
    
    @staticmethod
    def _is_enum(field_type: type) -> bool:
        try:
            return issubclass(field_type, Enum)
        except TypeError:
            return False
    
    @staticmethod
    def _is_literal(field_type: type) -> bool:
        origin = get_origin(field_type)
        return origin is not None and getattr(origin, '__name__', '') == 'Literal'

class ConfigEditor:
    """Clean config editor - coordinates components, no inheritance abuse."""
    
    def __init__(
        self,
        config_class: type,
        current_config: Any,
        backend: str,
        scope: str = "global",
        base_config: Optional[Any] = None,
        on_config_change: Optional[Callable[[str, Any, str], Coroutine]] = None,
        on_reset_field: Optional[Callable[[str, str], Coroutine]] = None,
        on_reset_all: Optional[Callable[[str], Coroutine]] = None
    ):
        # Dependencies - all required
        self.config_class = config_class
        self.current_config = current_config
        self.backend = backend
        self.scope = scope
        self.base_config = base_config
        
        # Callbacks
        self.on_config_change = on_config_change
        self.on_reset_field = on_reset_field
        self.on_reset_all = on_reset_all
        
        # Components
        self.introspector = FieldIntrospector()
        self.widget_factory = WidgetFactory(on_change=self._handle_field_change)
        
        # State
        self.field_specs: List[FieldSpec] = []
        self.widgets: Dict[str, Any] = {}
        
        # No work in constructor - just setup

    def build_ui(self) -> Container:
        """Build UI container - pure function."""
        # Analyze fields
        self.field_specs = self.introspector.analyze_dataclass(self.config_class, self.current_config)
        
        # Create widgets
        self.widgets = {}
        field_rows = []
        
        for spec in self.field_specs:
            widget = self.widget_factory.create_widget(spec)
            self.widgets[spec.name] = widget
            
            row = self._create_field_row(spec, widget)
            field_rows.append(row)
        
        # Build header
        title = f"{self.scope.title()} Configuration"
        reset_all_btn = Button("Reset All", handler=self._handle_reset_all, width=len("Reset All") + 2)
        
        header = VSplit([
            Label(title, style="class:frame.title"),
            Box(reset_all_btn, padding_left=2)
        ], height=1)
        
        # Build container
        if field_rows:
            return HSplit([header, HSplit(field_rows)])
        else:
            return HSplit([Label(f"No editable fields in {self.config_class.__name__}")])
    
    def _create_field_row(self, spec: FieldSpec, widget: Any) -> VSplit:
        """Create UI row for field."""
        # Build label with override indicator
        label_text = f"{spec.label}:"
        if self.scope == "plate" and self.base_config:
            base_value = getattr(self.base_config, spec.name, None)
            if spec.current_value != base_value:
                label_text += " [Override]"
        
        # Reset button
        reset_btn = Button(
            "Reset",
            handler=lambda name=spec.name: self._handle_reset_field(name),
            width=8
        )
        
        # RadioList can be used directly in VSplit - no wrapping needed

        return VSplit([
            Label(label_text, width=25),
            widget,
            Box(reset_btn, width=10, padding_left=1)
        ], padding=0)
    
    def update_config(self, new_config: Any) -> None:
        """Update config values in existing widgets."""
        self.current_config = new_config
        
        # Update widget values without rebuilding
        for spec in self.field_specs:
            widget = self.widgets.get(spec.name)
            if not widget:
                continue
                
            new_value = getattr(new_config, spec.name, None)
            self._update_widget_value(widget, new_value, spec.actual_type)
        
        self._invalidate_ui()
    
    def _update_widget_value(self, widget: Any, value: Any, field_type: type) -> None:
        """Update widget value based on type."""
        if isinstance(widget, Checkbox):
            widget.checked = bool(value)
        elif isinstance(widget, RadioList):
            widget.current_value = value
        elif isinstance(widget, TextArea):
            widget.text = str(value) if value is not None else ""
    
    def get_current_config(self) -> Any:
        """Extract current config from widgets."""
        field_values = {}
        
        for spec in self.field_specs:
            widget = self.widgets.get(spec.name)
            if not widget:
                field_values[spec.name] = spec.current_value
                continue
            
            try:
                value = self._extract_widget_value(widget, spec)
                field_values[spec.name] = value
            except Exception as e:
                logger.warning(f"Error extracting {spec.name}: {e}")
                field_values[spec.name] = spec.current_value
        
        try:
            return self.config_class(**field_values)
        except Exception as e:
            logger.error(f"Error creating config: {e}")
            return self.current_config
    
    def _extract_widget_value(self, widget: Any, spec: FieldSpec) -> Any:
        """Extract value from widget based on type."""
        if isinstance(widget, Checkbox):
            return widget.checked
        elif isinstance(widget, RadioList):
            return widget.current_value
        elif isinstance(widget, TextArea):
            return self._convert_text_value(widget.text, spec)
        else:
            return spec.current_value
    
    def _convert_text_value(self, text: str, spec: FieldSpec) -> Any:
        """Convert text to appropriate type."""
        text = text.strip()
        
        if not text and spec.is_optional:
            return None
        
        if spec.actual_type is int:
            return int(text) if text else 0
        elif spec.actual_type is float:
            return float(text) if text else 0.0
        elif spec.actual_type is Path:
            return Path(text) if text else None
        elif spec.actual_type is str:
            return text if text else None
        else:
            return text if text else None
    
    # Event handlers
    def _handle_field_change(self, field_name: str, value: Any) -> None:
        """Handle field value change."""
        if self.on_config_change:
            self._run_async(self.on_config_change(field_name, value, self.scope))
    
    def _handle_reset_field(self, field_name: str) -> None:
        """Handle field reset."""
        if self.on_reset_field:
            self._run_async(self.on_reset_field(field_name, self.scope))
    
    def _handle_reset_all(self) -> None:
        """Handle reset all."""
        if self.on_reset_all:
            self._run_async(self.on_reset_all(self.scope))
    
    def _run_async(self, coro: Coroutine) -> None:
        """Centralized async task dispatch."""
        from openhcs.tui.utils.unified_task_manager import get_task_manager
        get_task_manager().fire_and_forget(coro, "config_async")
    
    def _invalidate_ui(self) -> None:
        """Invalidate UI."""
        try:
            get_app().invalidate()
        except Exception:
            pass
