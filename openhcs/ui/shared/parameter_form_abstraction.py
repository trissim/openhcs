"""Simplified Parameter Form Abstraction"""

import dataclasses
from typing import Any, Dict, Type, Optional, get_origin, get_args, Union
from .widget_creation_registry import WidgetRegistry


class ParameterFormAbstraction:
    """Simplified parameter form logic."""

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, Type],
                 field_id: str, widget_registry: WidgetRegistry, parameter_info: Optional[Dict] = None):
        self.parameters = parameters
        self.parameter_types = parameter_types
        self.field_id = field_id
        self.widget_registry = widget_registry
        self.parameter_info = parameter_info or {}

    def create_widget_for_parameter(self, param_name: str, param_type: Type, current_value: Any) -> Any:
        """Create widget using registry."""
        return self.widget_registry.create_widget(
            param_name, param_type, current_value,
            f"{self.field_id}_{param_name}",
            self.parameter_info.get(param_name)
        )

    def is_optional_dataclass(self, param_type: Type) -> bool:
        """Check if type is Optional[dataclass]."""
        origin = get_origin(param_type)
        if origin is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                inner_type = next(arg for arg in args if arg is not type(None))
                return dataclasses.is_dataclass(inner_type)
        return False

    def get_optional_inner_type(self, param_type: Type) -> Type:
        """Extract T from Optional[T]."""
        origin = get_origin(param_type)
        if origin is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                return next(arg for arg in args if arg is not type(None))
        return param_type


# Simplified placeholder application - no unnecessary class hierarchy
def apply_lazy_default_placeholder(widget: Any, param_name: str, current_value: Any,
                                 parameter_types: Dict[str, Type], framework: str = 'textual',
                                 is_global_config_editing: bool = False,
                                 global_config_type: Optional[Type] = None,
                                 placeholder_prefix: str = "Pipeline default") -> None:
    """Apply lazy default placeholder if value is None."""
    if current_value is not None:
        return

    dataclass_type = _get_dataclass_type(parameter_types)
    if not dataclass_type:
        return

    try:
        # Try lazy placeholder service first (for special lazy dataclasses)
        placeholder_text = None
        try:
            from openhcs.core.config import LazyDefaultPlaceholderService
            if LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type):
                placeholder_text = LazyDefaultPlaceholderService.get_lazy_resolved_placeholder(
                    dataclass_type, param_name, force_static_defaults=is_global_config_editing, placeholder_prefix=placeholder_prefix
                )
        except Exception:
            pass

        # Fallback to thread-local resolution for regular dataclasses
        if not placeholder_text:
            try:
                # For regular dataclasses, create a dynamic lazy version that resolves from thread-local context
                # Determine the field path for nested forms
                field_path = _get_field_path_for_nested_form(dataclass_type, parameter_types, global_config_type)
                placeholder_text = _get_thread_local_placeholder(dataclass_type, param_name, is_global_config_editing, field_path, global_config_type, placeholder_prefix)
            except Exception:
                # Final fallback to static defaults
                try:
                    instance = dataclass_type()
                    default_value = getattr(instance, param_name, None)
                    if default_value is not None:
                        placeholder_text = f"{placeholder_prefix}: {default_value}"
                    else:
                        placeholder_text = f"{placeholder_prefix}: (none)"
                except Exception:
                    placeholder_text = f"{placeholder_prefix}: (default)"

        if placeholder_text:
            if framework == 'textual':
                if hasattr(widget, 'placeholder'):
                    widget.placeholder = placeholder_text
            elif framework == 'pyqt6':
                try:
                    from .pyqt6_widget_strategies import PyQt6WidgetEnhancer
                    PyQt6WidgetEnhancer.apply_placeholder_text(widget, placeholder_text)
                except ImportError:
                    # PyQt6 not available - fallback to basic placeholder setting
                    if hasattr(widget, 'placeholder'):
                        widget.placeholder = placeholder_text
    except Exception:
        pass


def _is_global_config_editing_mode(parameter_types: Dict[str, Type]) -> bool:
    """
    Detect if we're in global config editing mode vs orchestrator config editing mode.

    Global config editing: Fields have concrete values (preserve_values=True)
    Orchestrator config editing: Fields are None for placeholders (preserve_values=False)

    We can detect this by checking if the parameter types match PipelineConfig fields
    and if we're dealing with a lazy dataclass that should use static defaults.
    """
    try:
        # Check if this looks like PipelineConfig editing
        from openhcs.core.lazy_config import PipelineConfig
        import dataclasses

        if dataclasses.is_dataclass(PipelineConfig):
            pipeline_fields = {field.name for field in dataclasses.fields(PipelineConfig)}
            param_names = set(parameter_types.keys())

            # If the parameter names match PipelineConfig fields, we're in config editing mode
            if param_names == pipeline_fields:
                # For now, we'll use a heuristic: if we're editing PipelineConfig,
                # assume it's global config editing and use static defaults
                # This can be refined later if needed
                return True
    except Exception:
        pass
    return False


def _get_thread_local_placeholder(dataclass_type: Type, param_name: str, is_global_config_editing: bool,
                                 field_path: Optional[str] = None, global_config_type: Optional[Type] = None,
                                 placeholder_prefix: str = "Pipeline default") -> Optional[str]:
    """Get placeholder text using thread-local resolution for regular dataclasses."""
    try:
        from openhcs.core.lazy_config import LazyDataclassFactory
        from openhcs.core.config import LazyDefaultPlaceholderService

        if is_global_config_editing:
            # Global config editing: use static defaults
            instance = dataclass_type()
            default_value = getattr(instance, param_name, None)
            if default_value is not None:
                return f"{placeholder_prefix}: {default_value}"
            else:
                return f"{placeholder_prefix}: (none)"
        else:
            # Orchestrator config editing: resolve from thread-local global config
            # Create a dynamic lazy version of the dataclass that resolves from thread-local context
            if global_config_type is None:
                # Default to GlobalPipelineConfig for backward compatibility
                from openhcs.core.config import GlobalPipelineConfig
                global_config_type = GlobalPipelineConfig

            dynamic_lazy_class = LazyDataclassFactory.make_lazy_thread_local(
                base_class=dataclass_type,
                global_config_type=global_config_type,
                field_path=field_path,  # Use the provided field path for nested forms
                lazy_class_name=f"Dynamic{dataclass_type.__name__}"
            )

            # Use the lazy placeholder service to resolve from thread-local context
            placeholder_text = LazyDefaultPlaceholderService.get_lazy_resolved_placeholder(
                dynamic_lazy_class, param_name, force_static_defaults=False, placeholder_prefix=placeholder_prefix
            )

            return placeholder_text

    except Exception as e:
        # Fallback to static defaults if thread-local resolution fails
        try:
            instance = dataclass_type()
            default_value = getattr(instance, param_name, None)
            if default_value is not None:
                return f"{placeholder_prefix}: {default_value}"
            else:
                return f"{placeholder_prefix}: (none)"
        except Exception:
            return f"{placeholder_prefix}: (default)"


def _get_field_path_for_nested_form(dataclass_type: Type, parameter_types: Dict[str, Type],
                                   global_config_type: Optional[Type] = None) -> Optional[str]:
    """Determine the field path for nested form placeholder generation."""
    # Use consolidated field path detection utility
    from openhcs.core.field_path_detection import FieldPathDetector

    # If no global config type specified, use default
    if global_config_type is None:
        from openhcs.core.config import GlobalPipelineConfig
        global_config_type = GlobalPipelineConfig

    return FieldPathDetector.find_field_path_for_type(global_config_type, dataclass_type)


def _get_dataclass_type(parameter_types: Dict[str, Type]) -> Optional[Type]:
    """Get dataclass type using introspection - works for ANY dataclass, not just lazy ones."""
    try:
        param_names = set(parameter_types.keys())

        # First, check if any of the parameter types directly is a dataclass
        for param_type in parameter_types.values():
            if dataclasses.is_dataclass(param_type):
                dataclass_fields = {field.name for field in dataclasses.fields(param_type)}
                if param_names == dataclass_fields:
                    return param_type

        # Then check both config module and lazy_config module for dataclasses
        import inspect
        from openhcs.core import config, lazy_config

        modules_to_check = [config, lazy_config]

        for module in modules_to_check:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if dataclasses.is_dataclass(obj):
                    dataclass_fields = {field.name for field in dataclasses.fields(obj)}
                    if param_names == dataclass_fields:
                        return obj

        # Finally, check the calling frame for locally defined dataclasses (like in tests)
        import sys
        frame = sys._getframe(1)
        while frame:
            for name, obj in frame.f_locals.items():
                if (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
                    dataclass_fields = {field.name for field in dataclasses.fields(obj)}
                    if param_names == dataclass_fields:
                        return obj
            frame = frame.f_back

    except Exception:
        pass
    return None
