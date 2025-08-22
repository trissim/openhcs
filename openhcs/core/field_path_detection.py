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

    @staticmethod
    def find_all_field_paths_for_type(parent_type: Type, target_type: Type) -> list[str]:
        """
        Find ALL field paths that contain the target type in the parent config structure.

        This enables automatic hierarchy discovery for lazy resolution by recursively
        searching through nested dataclass structures to find all instances of a
        target type.

        Args:
            parent_type: The parent dataclass type to search within
            target_type: The target dataclass type to find all field paths for

        Returns:
            List of field paths (e.g., ['materialization_defaults', 'nested.path'])

        Examples:
            >>> FieldPathDetector.find_all_field_paths_for_type(
            ...     GlobalPipelineConfig, StepMaterializationConfig
            ... )
            ['materialization_defaults']
        """
        paths = []

        def _recursive_search(current_type: Type, current_path: str = ""):
            if not dataclasses.is_dataclass(current_type):
                return

            for field in dataclasses.fields(current_type):
                field_type = FieldPathDetector._unwrap_optional_type(field.type)
                field_path = f"{current_path}.{field.name}" if current_path else field.name

                # Direct type match
                if field_type == target_type:
                    paths.append(field_path)
                # Recursive search in nested dataclasses
                elif dataclasses.is_dataclass(field_type):
                    _recursive_search(field_type, field_path)

        _recursive_search(parent_type)
        return paths

    @staticmethod
    def find_inheritance_relationships(target_type: Type) -> list[Type]:
        """
        Find all parent dataclasses that target_type inherits from.

        This method recursively traverses the inheritance chain to discover
        all dataclass parents, enabling automatic sibling inheritance detection
        for lazy configuration resolution.

        Args:
            target_type: The dataclass type to analyze for inheritance relationships

        Returns:
            List of parent dataclass types in inheritance order

        Examples:
            >>> FieldPathDetector.find_inheritance_relationships(StepMaterializationConfig)
            [PathPlanningConfig]
        """
        inheritance_chain = []

        for base in target_type.__bases__:
            if base != object and dataclasses.is_dataclass(base):
                inheritance_chain.append(base)
                # Recursively find parent relationships
                inheritance_chain.extend(FieldPathDetector.find_inheritance_relationships(base))

        return inheritance_chain