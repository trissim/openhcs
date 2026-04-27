"""
Settings Binder - Convert .cppipe settings to function kwargs.

Maps CellProfiler setting strings to typed Python values for OpenHCS functions.
Handles type inference, name normalization, and value parsing.
"""

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class BoundParameter:
    """A parameter with its bound value."""
    name: str  # OpenHCS parameter name (snake_case)
    value: Any  # Typed value
    original_key: str  # Original .cppipe setting key
    original_value: str  # Original .cppipe setting value


class SettingsBinder:
    """
    Bind .cppipe settings to OpenHCS function parameters.
    
    Handles:
    - Name normalization: "Typical diameter of objects" → "typical_diameter"
    - Type inference: "8,80" → (8, 80), "Yes" → True, "0.05" → 0.05
    - Enum mapping: "Otsu" → ThresholdMethod.OTSU
    """
    
    # Common boolean values in CellProfiler
    BOOL_TRUE = {"yes", "true", "1", "on"}
    BOOL_FALSE = {"no", "false", "0", "off"}
    
    # Settings to skip (CellProfiler-specific, not needed for OpenHCS)
    SKIP_SETTINGS = {
        "show_window",
        "notes",
        "batch_state",
        "wants_pause",
        "module_num",
        "svn_version",
        "variable_revision_number",
    }
    
    def __init__(self, enum_mappings: Optional[Dict[str, type]] = None):
        """
        Initialize binder.
        
        Args:
            enum_mappings: Dict mapping setting names to enum types
        """
        self.enum_mappings = enum_mappings or {}
    
    def bind(self, settings: Dict[str, str]) -> Dict[str, Any]:
        """
        Bind .cppipe settings to kwargs dict.
        
        Args:
            settings: Dict of setting key → string value from .cppipe
            
        Returns:
            Dict of parameter name → typed value
        """
        kwargs = {}
        
        for key, value in settings.items():
            # Skip CellProfiler-specific settings
            normalized_key = self._normalize_name(key)
            if normalized_key in self.SKIP_SETTINGS:
                continue
            
            # Parse value
            typed_value = self._parse_value(key, value)
            
            # Add to kwargs
            kwargs[normalized_key] = typed_value
        
        return kwargs
    
    def bind_with_details(self, settings: Dict[str, str]) -> List[BoundParameter]:
        """
        Bind settings and return detailed binding info.
        
        Args:
            settings: Dict of setting key → string value from .cppipe
            
        Returns:
            List of BoundParameter with full binding details
        """
        result = []
        
        for key, value in settings.items():
            normalized_key = self._normalize_name(key)
            if normalized_key in self.SKIP_SETTINGS:
                continue
            
            typed_value = self._parse_value(key, value)
            
            result.append(BoundParameter(
                name=normalized_key,
                value=typed_value,
                original_key=key,
                original_value=value,
            ))
        
        return result
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize setting name to snake_case parameter name.
        
        "Typical diameter of objects, in pixel units (Min,Max)" → "typical_diameter_min_max"
        """
        # Remove parenthetical content
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove question marks
        name = name.replace('?', '')
        
        # Replace special chars with spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Convert to lowercase and split
        words = name.lower().split()
        
        # Join with underscores
        return '_'.join(words)
    
    def _parse_value(self, key: str, value: str) -> Any:
        """
        Parse string value to typed Python value.
        
        Handles:
        - Booleans: "Yes" → True
        - Integers: "10" → 10
        - Floats: "0.05" → 0.05
        - Tuples: "8,80" → (8, 80)
        - Ranges: "0.0,1.0" → (0.0, 1.0)
        - Lists: "DNA, PH3" → ["DNA", "PH3"]
        - Enums: Lookup in enum_mappings
        """
        value = value.strip()
        
        # Check for boolean
        if value.lower() in self.BOOL_TRUE:
            return True
        if value.lower() in self.BOOL_FALSE:
            return False
        
        # Check for enum mapping
        normalized_key = self._normalize_name(key)
        if normalized_key in self.enum_mappings:
            enum_type = self.enum_mappings[normalized_key]
            try:
                return enum_type[value.upper().replace(' ', '_')]
            except KeyError:
                logger.warning(f"Unknown enum value '{value}' for {normalized_key}")
                return value
        
        # Check for comma-separated values
        if ',' in value:
            parts = [p.strip() for p in value.split(',')]
            
            # Try to parse as numeric tuple
            try:
                numeric_parts = []
                for p in parts:
                    if '.' in p:
                        numeric_parts.append(float(p))
                    else:
                        numeric_parts.append(int(p))
                return tuple(numeric_parts)
            except ValueError:
                # Not numeric - return as list of strings
                return parts
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
