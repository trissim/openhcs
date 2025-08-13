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
                                 parameter_types: Dict[str, Type], framework: str = 'textual') -> None:
    """Apply lazy default placeholder if value is None."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"=== PLACEHOLDER DEBUG === {param_name}: value={current_value}, framework={framework}")

    if current_value is not None:
        logger.info(f"Skipping placeholder for {param_name} - value is not None: {current_value}")
        return

    dataclass_type = _get_dataclass_type(parameter_types)
    logger.info(f"Dataclass type for {param_name}: {dataclass_type}")
    if not dataclass_type:
        logger.info(f"No dataclass type found for {param_name}")
        return

    try:
        from openhcs.core.config import LazyDefaultPlaceholderService
        placeholder_text = LazyDefaultPlaceholderService.get_lazy_resolved_placeholder(
            dataclass_type, param_name
        )
        logger.info(f"Generated placeholder for {param_name}: {placeholder_text}")
        if placeholder_text:
            if framework == 'textual':
                if hasattr(widget, 'placeholder'):
                    widget.placeholder = placeholder_text
                    logger.info(f"Applied textual placeholder to {param_name}")
            elif framework == 'pyqt6':
                try:
                    from .pyqt6_widget_strategies import PyQt6WidgetEnhancer
                    PyQt6WidgetEnhancer.apply_placeholder_text(widget, placeholder_text)
                    logger.info(f"Applied PyQt6 placeholder to {param_name}")
                except ImportError:
                    # PyQt6 not available - fallback to basic placeholder setting
                    if hasattr(widget, 'placeholder'):
                        widget.placeholder = placeholder_text
                        logger.info(f"Applied fallback placeholder to {param_name}")
        else:
            logger.info(f"No placeholder text generated for {param_name}")
    except Exception as e:
        logger.error(f"Exception applying placeholder for {param_name}: {e}")
        pass


def _get_dataclass_type(parameter_types: Dict[str, Type]) -> Optional[Type]:
    """Get dataclass type using introspection."""
    try:
        from openhcs.core.config import LazyDefaultPlaceholderService
        param_names = set(parameter_types.keys())

        # Check both config module and lazy_config module for lazy dataclasses
        import inspect
        from openhcs.core import config, lazy_config

        modules_to_check = [config, lazy_config]

        for module in modules_to_check:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (dataclasses.is_dataclass(obj) and
                    LazyDefaultPlaceholderService.has_lazy_resolution(obj)):
                    dataclass_fields = {field.name for field in dataclasses.fields(obj)}
                    if param_names == dataclass_fields:
                        return obj
    except Exception:
        pass
    return None
