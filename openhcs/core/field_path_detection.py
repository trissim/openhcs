"""
Automatic field path detection utility for OpenHCS configuration types.

This module provides utilities for automatically detecting field paths in dataclass
types, eliminating hardcoded field names and providing a single source of truth
for type introspection logic across the UI system.
"""

import dataclasses
import typing
from typing import Type, Optional, Union, get_origin, get_args
from dataclasses import fields


class FieldPathDetector:
    """Automatic field path detection utility for dataclass type introspection."""

    @staticmethod
    def find_field_path_for_type(parent_type: Type, child_type: Type) -> Optional[str]:
        """
        Find field path by inspecting parent type annotations.

        Consolidates the logic from scattered implementations in:
        - PyQt parameter form manager
        - Textual parameter form manager
        - Parameter form abstraction

        Args:
            parent_type: The parent dataclass type to search within
            child_type: The child dataclass type to find the field path for

        Returns:
            The field path string (e.g., 'path_planning', 'vfs') or None if not found
        """
        try:
            if not dataclasses.is_dataclass(parent_type):
                return None

            # Get all fields from parent type
            parent_fields = fields(parent_type)

            for field in parent_fields:
                field_type = FieldPathDetector._unwrap_optional_type(field.type)

                # Check for direct type match (handles the common case)
                if field_type == child_type:
                    return field.name

            return None

        except Exception:
            # Fail gracefully for any type introspection issues
            return None

    @staticmethod
    def _unwrap_optional_type(field_type: Type) -> Type:
        """
        Convert Optional[T] -> T, Union[T, None] -> T, etc.

        Extracted from the common logic in scattered implementations.
        """
        # Handle Optional types (Union[Type, None])
        if hasattr(typing, 'get_origin') and get_origin(field_type) is Union:
            # Get the non-None type from Optional[Type] or Union[Type, None]
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # Return the non-None type
                return args[0] if args[1] is type(None) else args[1]

        # Return the type as-is if not a generic/optional type
        return field_type