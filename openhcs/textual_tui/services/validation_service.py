"""Form validation service for Textual TUI."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum


class ValidationResult:
    """Result of field validation."""
    
    def __init__(self, is_valid: bool, error_message: Optional[str] = None):
        self.is_valid = is_valid
        self.error_message = error_message


class ValidationService:
    """Service for validating form field values."""
    
    @staticmethod
    def validate_field(field_type: type, value: Any, is_optional: bool = False) -> ValidationResult:
        """Validate a field value against its expected type."""
        # Handle optional fields
        if value is None or value == "":
            if is_optional:
                return ValidationResult(True)
            else:
                return ValidationResult(False, "Field is required")
        
        # Type-specific validation
        if field_type == bool:
            return ValidationService._validate_bool(value)
        elif field_type == int:
            return ValidationService._validate_int(value)
        elif field_type == float:
            return ValidationService._validate_float(value)
        elif field_type == str:
            return ValidationService._validate_str(value)
        elif field_type == Path:
            return ValidationService._validate_path(value)
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            return ValidationService._validate_enum(value, field_type)
        else:
            # Unknown type - accept as string
            return ValidationResult(True)
    
    @staticmethod
    def _validate_bool(value: Any) -> ValidationResult:
        """Validate boolean value."""
        if isinstance(value, bool):
            return ValidationResult(True)
        return ValidationResult(False, "Must be true or false")
    
    @staticmethod
    def _validate_int(value: Any) -> ValidationResult:
        """Validate integer value."""
        try:
            int(value)
            return ValidationResult(True)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be a valid integer")
    
    @staticmethod
    def _validate_float(value: Any) -> ValidationResult:
        """Validate float value."""
        try:
            float(value)
            return ValidationResult(True)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be a valid number")
    
    @staticmethod
    def _validate_str(value: Any) -> ValidationResult:
        """Validate string value."""
        if isinstance(value, str):
            return ValidationResult(True)
        return ValidationResult(False, "Must be text")
    
    @staticmethod
    def _validate_path(value: Any) -> ValidationResult:
        """Validate path value."""
        try:
            Path(str(value))
            return ValidationResult(True)
        except Exception:
            return ValidationResult(False, "Must be a valid path")
    
    @staticmethod
    def _validate_enum(value: Any, enum_type: type) -> ValidationResult:
        """Validate enum value."""
        try:
            if value in [member.value for member in enum_type]:
                return ValidationResult(True)
            return ValidationResult(False, f"Must be one of: {[m.value for m in enum_type]}")
        except Exception:
            return ValidationResult(False, "Invalid enum value")
    
    @staticmethod
    def validate_form(field_specs: List[Any], field_values: Dict[str, Any]) -> Dict[str, str]:
        """Validate entire form and return error messages."""
        errors = {}
        
        for spec in field_specs:
            field_name = spec.name
            field_value = field_values.get(field_name)
            
            result = ValidationService.validate_field(
                spec.actual_type, 
                field_value, 
                spec.is_optional
            )
            
            if not result.is_valid:
                errors[field_name] = result.error_message
        
        return errors
