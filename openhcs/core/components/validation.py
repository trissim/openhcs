"""
Generic validation system for component-agnostic validation.

This module provides a generic replacement for the component-specific validation
logic, supporting any component configuration and validation patterns.
"""

import logging
from typing import Generic, TypeVar, List, Optional, Dict, Any, Union, Type
from enum import Enum
from dataclasses import dataclass

from .framework import ComponentConfiguration

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Enum)
U = TypeVar('U', bound=Enum)


def convert_enum_by_value(source_enum: T, target_enum_class: Type[U]) -> Optional[U]:
    """
    Generic utility to convert between enum types with matching .value attributes.

    This function enables conversion between any two enum classes that have
    overlapping values, without requiring hardcoded mappings.

    Args:
        source_enum: Source enum instance to convert from
        target_enum_class: Target enum class to convert to

    Returns:
        Target enum instance with matching value, or None if no match found

    Example:
        >>> convert_enum_by_value(VariableComponents.CHANNEL, GroupBy)
        <GroupBy.CHANNEL: 'channel'>
    """
    source_value = source_enum.value

    for target_enum in target_enum_class:
        if target_enum.value == source_value:
            return target_enum

    return None


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None


class GenericValidator(Generic[T]):
    """
    Generic validator for component-agnostic validation.
    
    This class replaces the hardcoded component-specific validation logic
    with a configurable system that works with any component configuration.
    """
    
    def __init__(self, config: ComponentConfiguration[T]):
        """
        Initialize the validator with a component configuration.
        
        Args:
            config: ComponentConfiguration for validation rules
        """
        self.config = config
        logger.debug(f"GenericValidator initialized for components: {[c.value for c in config.all_components]}")
    
    def validate_step(
        self,
        variable_components: List[T],
        group_by: Optional[Union[T, 'GroupBy']],
        func_pattern: Any,
        step_name: str
    ) -> ValidationResult:
        """
        Validate a step configuration using generic rules.
        
        Args:
            variable_components: List of variable components
            group_by: Optional group_by component
            func_pattern: Function pattern (callable, dict, or list)
            step_name: Name of the step for error reporting
            
        Returns:
            ValidationResult indicating success or failure
        """
        try:
            # 1. Validate component combination
            self.config.validate_combination(variable_components, group_by)
            
            # 2. Validate dict pattern requirements
            if isinstance(func_pattern, dict) and not group_by:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Dict pattern requires group_by in step '{step_name}'"
                )
            
            # 3. Validate components are in remaining components (not multiprocessing axis)
            remaining_components = self.config.get_remaining_components()
            remaining_values = {comp.value for comp in remaining_components}

            for component in variable_components:
                if component.value not in remaining_values:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Variable component {component.value} not available (multiprocessing axis: {self.config.multiprocessing_axis.value})"
                    )

            if group_by and group_by.value not in remaining_values:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Group_by component {group_by.value} not available (multiprocessing axis: {self.config.multiprocessing_axis.value})"
                )
            
            return ValidationResult(is_valid=True)
            
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                error_message=str(e)
            )
    
    def validate_dict_pattern_keys(
        self,
        func_pattern: Dict[str, Any],
        group_by: T,
        step_name: str,
        orchestrator
    ) -> ValidationResult:
        """
        Validate that dict function pattern keys match available component keys.
        
        This validation ensures compile-time guarantee that dict patterns will work
        at runtime by checking that all dict keys exist in the actual component data.
        
        Args:
            func_pattern: Dict function pattern to validate
            group_by: GroupBy component specifying component type
            step_name: Name of the step containing the function
            orchestrator: Orchestrator for component key access
            
        Returns:
            ValidationResult indicating success or failure
        """
        try:
            # Use enum objects directly - orchestrator now accepts VariableComponents
            available_keys = orchestrator.get_component_keys(group_by)
            available_keys_set = set(str(key) for key in available_keys)
            
            # Check each dict key against available keys
            pattern_keys = list(func_pattern.keys())
            pattern_keys_set = set(str(key) for key in pattern_keys)
            
            # Try direct string match first
            missing_keys = pattern_keys_set - available_keys_set
            
            if missing_keys:
                # Try numeric conversion for better error reporting
                try:
                    available_numeric = {str(int(float(k))) for k in available_keys if str(k).replace('.', '').isdigit()}
                    pattern_numeric = {str(int(float(k))) for k in pattern_keys if str(k).replace('.', '').isdigit()}
                    missing_numeric = pattern_numeric - available_numeric
                    
                    if missing_numeric:
                        return ValidationResult(
                            is_valid=False,
                            error_message=(
                                f"Function pattern keys {sorted(missing_numeric)} not found in available "
                                f"{group_by.value} components {sorted(available_numeric)} for step '{step_name}'"
                            )
                        )
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    return ValidationResult(
                        is_valid=False,
                        error_message=(
                            f"Function pattern keys {sorted(missing_keys)} not found in available "
                            f"{group_by.value} components {sorted(available_keys_set)} for step '{step_name}'"
                        )
                    )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Failed to validate dict pattern keys for {group_by.value}: {e}"
            )
    
    def validate_component_combination_constraint(
        self,
        variable_components: List[T],
        group_by: Optional[T]
    ) -> ValidationResult:
        """
        Validate the core constraint: group_by ∉ variable_components.
        
        Args:
            variable_components: List of variable components
            group_by: Optional group_by component
            
        Returns:
            ValidationResult indicating success or failure
        """
        try:
            self.config.validate_combination(variable_components, group_by)
            return ValidationResult(is_valid=True)
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                error_message=str(e)
            )
