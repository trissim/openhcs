"""
Core framework for generic component configuration.

This module provides the foundational classes for configuring any enum as components
with configurable multiprocessing axis and validation constraints.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Set, List, Optional, Type
from enum import Enum

T = TypeVar('T', bound=Enum)


@dataclass(frozen=True)
class ComponentConfiguration(Generic[T]):
    """
    Generic configuration for any enum-based component system.
    
    This class encapsulates the configuration for a component system where:
    - Components are defined by an enum
    - One component serves as the multiprocessing axis
    - Default variable components and group_by are specified
    - Generic constraint validation is enforced: group_by ∉ variable_components
    """
    
    all_components: Set[T]
    multiprocessing_axis: T
    default_variable: List[T]
    default_group_by: Optional[T]
    
    def __post_init__(self):
        """Validate configuration constraints."""
        # Ensure multiprocessing_axis is in all_components
        if self.multiprocessing_axis not in self.all_components:
            raise ValueError(
                f"multiprocessing_axis {self.multiprocessing_axis.value} "
                f"must be in all_components"
            )
        
        # Ensure default_variable components are in all_components
        for component in self.default_variable:
            if component not in self.all_components:
                raise ValueError(
                    f"default_variable component {component.value} "
                    f"must be in all_components"
                )
        
        # Ensure default_group_by is in all_components (if specified)
        if self.default_group_by and self.default_group_by not in self.all_components:
            raise ValueError(
                f"default_group_by {self.default_group_by.value} "
                f"must be in all_components"
            )
        
        # Validate default combination
        self.validate_combination(self.default_variable, self.default_group_by)
    
    def validate_combination(self, variable: List[T], group_by: Optional[T]) -> None:
        """
        Validate that group_by is not in variable_components.

        This enforces the core constraint: group_by ∉ variable_components

        Args:
            variable: List of variable components
            group_by: Optional group_by component

        Raises:
            ValueError: If group_by is in variable_components
        """
        if group_by and group_by in variable:
            raise ValueError(
                f"group_by {group_by.value} cannot be in variable_components "
                f"{[v.value for v in variable]}"
            )

    def get_remaining_components(self) -> Set[T]:
        """
        Get components available for variable_components and group_by selection.

        Returns all components except the multiprocessing_axis.

        Returns:
            Set of components available for variable/group_by selection
        """
        return self.all_components - {self.multiprocessing_axis}

    def get_available_variable_components(self) -> List[T]:
        """
        Get all components that can be used as variable_components.

        Returns:
            List of components available as variable components
        """
        return list(self.get_remaining_components())

    def get_available_group_by_components(self, exclude_variable: Optional[List[T]] = None) -> List[T]:
        """
        Get components that can be used as group_by, excluding variable components.

        Args:
            exclude_variable: Variable components to exclude from group_by options

        Returns:
            List of components available as group_by
        """
        remaining = self.get_remaining_components()
        if exclude_variable:
            remaining = remaining - set(exclude_variable)
        return list(remaining)


class ComponentConfigurationFactory:
    """Factory for creating ComponentConfiguration instances."""
    
    @staticmethod
    def create_configuration(
        component_enum: Type[T],
        multiprocessing_axis: T,
        default_variable: Optional[List[T]] = None,
        default_group_by: Optional[T] = None
    ) -> ComponentConfiguration[T]:
        """
        Create a ComponentConfiguration for the given enum with dynamic component resolution.

        When multiprocessing_axis is specified, the remaining components are automatically
        available for variable_components and group_by selection.

        Args:
            component_enum: The enum class defining all components
            multiprocessing_axis: Component to use for multiprocessing
            default_variable: Default variable components (auto-resolved if None)
            default_group_by: Default group_by component (auto-resolved if None)

        Returns:
            ComponentConfiguration instance
        """
        all_components = set(component_enum)

        # Dynamic resolution: remaining components = all_components - multiprocessing_axis
        remaining_components = all_components - {multiprocessing_axis}

        # Auto-resolve default_variable if not specified
        if default_variable is None:
            # Use the first remaining component as default variable
            default_variable = [list(remaining_components)[0]] if remaining_components else []

        # Auto-resolve default_group_by if not specified
        if default_group_by is None and len(remaining_components) > 1:
            # Use the second remaining component as default group_by (if available)
            remaining_list = list(remaining_components)
            # Ensure group_by is not in default_variable
            for component in remaining_list:
                if component not in default_variable:
                    default_group_by = component
                    break

        return ComponentConfiguration(
            all_components=all_components,
            multiprocessing_axis=multiprocessing_axis,
            default_variable=default_variable,
            default_group_by=default_group_by
        )
    
    @staticmethod
    def create_openhcs_default_configuration():
        """
        Create the default OpenHCS configuration.
        
        This maintains backward compatibility with the current OpenHCS setup:
        - Well as multiprocessing axis
        - Site as default variable component
        - Channel as default group_by
        """
        from openhcs.constants.constants import VariableComponents
        
        return ComponentConfigurationFactory.create_configuration(
            VariableComponents,
            multiprocessing_axis=VariableComponents.WELL,
            default_variable=[VariableComponents.SITE],
            default_group_by=VariableComponents.CHANNEL
        )
