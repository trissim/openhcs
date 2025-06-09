"""Config reflection service - port from old TUI."""

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, List, Optional, get_origin, get_args, Union as TypingUnion
from pathlib import Path
from enum import Enum


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
    is_nested_dataclass: bool = False
    nested_fields: Optional[List['FieldSpec']] = None


class FieldIntrospector:
    """Analyzes dataclass fields for form generation."""
    
    @staticmethod
    def analyze_dataclass(dataclass_type: type, instance: Any) -> List[FieldSpec]:
        """Analyze dataclass and return field specifications."""
        if not dataclasses.is_dataclass(dataclass_type):
            raise ValueError(f"{dataclass_type} is not a dataclass")
        
        specs = []
        for field in fields(dataclass_type):
            if field.name.startswith('_'):
                continue
            
            spec = FieldIntrospector._analyze_field(field, instance)
            specs.append(spec)
        
        return specs
    
    @staticmethod
    def _analyze_field(field: dataclasses.Field, instance: Any) -> FieldSpec:
        """Analyze individual field."""
        # Extract type information
        field_type = field.type
        actual_type = field_type
        is_optional = get_origin(field_type) is TypingUnion and type(None) in get_args(field_type)

        if is_optional:
            actual_type = next((t for t in get_args(field_type) if t is not type(None)), actual_type)

        # Get values
        current_value = getattr(instance, field.name, None)
        default_value = None

        if field.default is not dataclasses.MISSING:
            default_value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_value = field.default_factory()

        # Create label
        label = field.name.replace('_', ' ').title()

        # Check if this is a nested dataclass
        is_nested_dataclass = dataclasses.is_dataclass(actual_type)
        nested_fields = None

        if is_nested_dataclass and current_value is not None:
            # Recursively analyze nested dataclass
            nested_fields = FieldIntrospector.analyze_dataclass(actual_type, current_value)

        return FieldSpec(
            name=field.name,
            label=label,
            field_type=field_type,
            actual_type=actual_type,
            current_value=current_value,
            default_value=default_value,
            is_optional=is_optional,
            is_nested_dataclass=is_nested_dataclass,
            nested_fields=nested_fields
        )
