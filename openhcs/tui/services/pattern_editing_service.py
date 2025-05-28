"""
Pattern Editing Service - Business Logic for Function Pattern Editing.

Handles the business logic for function pattern editing, validation, and management.
Separates pattern operations from UI concerns.

🔒 Clause 295: Component Boundaries
Clear separation between pattern business logic and UI interaction.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import copy

from openhcs.processing.func_registry import FUNC_REGISTRY
from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator

logger = logging.getLogger(__name__)


class PatternEditingService:
    """
    Service layer for function pattern editing operations.
    
    Handles:
    - Pattern validation and manipulation
    - Function registry interactions
    - Pattern conversion operations
    - Business logic for pattern editing
    """
    
    def __init__(self, state):
        self.state = state
        self.func_registry = FUNC_REGISTRY
        self.validator = FuncStepContractValidator
    
    def create_empty_pattern(self, pattern_type: str = 'list') -> Union[List, Dict]:
        """
        Create an empty pattern of the specified type.
        
        Args:
            pattern_type: 'list' or 'dict'
            
        Returns:
            Empty pattern of the specified type
        """
        if pattern_type == 'dict':
            return {}
        else:
            return []
    
    def clone_pattern(self, pattern: Union[List, Dict, None]) -> Union[List, Dict]:
        """
        Create a deep copy of a pattern.
        
        Args:
            pattern: Pattern to clone
            
        Returns:
            Deep copy of the pattern
        """
        if pattern is None:
            return []
        return copy.deepcopy(pattern)
    
    def is_dict_pattern(self, pattern: Union[List, Dict]) -> bool:
        """Check if a pattern is a dictionary pattern."""
        return isinstance(pattern, dict)
    
    def get_pattern_keys(self, pattern: Union[List, Dict]) -> List[str]:
        """
        Get the keys from a pattern.
        
        Args:
            pattern: Pattern to get keys from
            
        Returns:
            List of keys (empty for list patterns)
        """
        if self.is_dict_pattern(pattern):
            return list(pattern.keys())
        return []
    
    def add_pattern_key(self, pattern: Dict, key: str) -> Dict:
        """
        Add a new key to a dictionary pattern.
        
        Args:
            pattern: Dictionary pattern
            key: Key to add
            
        Returns:
            Updated pattern
        """
        if not self.is_dict_pattern(pattern):
            raise ValueError("Cannot add key to non-dictionary pattern")
        
        if key in pattern:
            raise ValueError(f"Key '{key}' already exists in pattern")
        
        pattern[key] = []
        return pattern
    
    def remove_pattern_key(self, pattern: Dict, key: str) -> Dict:
        """
        Remove a key from a dictionary pattern.
        
        Args:
            pattern: Dictionary pattern
            key: Key to remove
            
        Returns:
            Updated pattern
        """
        if not self.is_dict_pattern(pattern):
            raise ValueError("Cannot remove key from non-dictionary pattern")
        
        if key not in pattern:
            raise ValueError(f"Key '{key}' not found in pattern")
        
        del pattern[key]
        return pattern
    
    def convert_list_to_dict_pattern(self, pattern: List) -> Dict:
        """
        Convert a list pattern to a dictionary pattern.
        
        Args:
            pattern: List pattern to convert
            
        Returns:
            Dictionary pattern with None key containing the original list
        """
        if self.is_dict_pattern(pattern):
            return pattern
        
        # Use None as key for unnamed group (Clause 234)
        return {None: pattern}
    
    def convert_dict_to_list_pattern(self, pattern: Dict) -> List:
        """
        Convert a dictionary pattern to a list pattern.
        
        Args:
            pattern: Dictionary pattern to convert
            
        Returns:
            List pattern (uses None key if available, otherwise first key)
        """
        if not self.is_dict_pattern(pattern):
            return pattern
        
        if None in pattern:
            return pattern[None]
        elif pattern:
            # Use first key's value
            first_key = next(iter(pattern))
            return pattern[first_key]
        else:
            return []
    
    def get_pattern_functions(self, pattern: Union[List, Dict], key: Optional[str] = None) -> List[Tuple[Callable, Dict]]:
        """
        Get functions from a pattern.
        
        Args:
            pattern: Pattern to get functions from
            key: Key for dictionary patterns (None for list patterns)
            
        Returns:
            List of (function, kwargs) tuples
        """
        if self.is_dict_pattern(pattern):
            if key is None:
                # Return all functions from all keys
                all_functions = []
                for k, func_list in pattern.items():
                    all_functions.extend(self._extract_functions_from_list(func_list))
                return all_functions
            else:
                func_list = pattern.get(key, [])
                return self._extract_functions_from_list(func_list)
        else:
            return self._extract_functions_from_list(pattern)
    
    def _extract_functions_from_list(self, func_list: List) -> List[Tuple[Callable, Dict]]:
        """Extract functions from a function list."""
        functions = []
        for item in func_list:
            if isinstance(item, dict) and 'func' in item:
                func = item['func']
                kwargs = item.get('kwargs', {})
                functions.append((func, kwargs))
            elif callable(item):
                functions.append((item, {}))
        return functions
    
    def add_function_to_pattern(self, pattern: Union[List, Dict], func: Callable, kwargs: Dict = None, key: Optional[str] = None) -> Union[List, Dict]:
        """
        Add a function to a pattern.
        
        Args:
            pattern: Pattern to add function to
            func: Function to add
            kwargs: Function arguments
            key: Key for dictionary patterns
            
        Returns:
            Updated pattern
        """
        if kwargs is None:
            kwargs = {}
        
        func_entry = {'func': func, 'kwargs': kwargs}
        
        if self.is_dict_pattern(pattern):
            if key is None:
                raise ValueError("Key required for dictionary pattern")
            if key not in pattern:
                pattern[key] = []
            pattern[key].append(func_entry)
        else:
            pattern.append(func_entry)
        
        return pattern
    
    def remove_function_from_pattern(self, pattern: Union[List, Dict], index: int, key: Optional[str] = None) -> Union[List, Dict]:
        """
        Remove a function from a pattern.
        
        Args:
            pattern: Pattern to remove function from
            index: Index of function to remove
            key: Key for dictionary patterns
            
        Returns:
            Updated pattern
        """
        if self.is_dict_pattern(pattern):
            if key is None:
                raise ValueError("Key required for dictionary pattern")
            if key not in pattern:
                raise ValueError(f"Key '{key}' not found in pattern")
            func_list = pattern[key]
        else:
            func_list = pattern
        
        if 0 <= index < len(func_list):
            del func_list[index]
        else:
            raise ValueError(f"Index {index} out of range")
        
        return pattern
    
    def move_function_in_pattern(self, pattern: Union[List, Dict], from_index: int, to_index: int, key: Optional[str] = None) -> Union[List, Dict]:
        """
        Move a function within a pattern.
        
        Args:
            pattern: Pattern to modify
            from_index: Current index of function
            to_index: Target index for function
            key: Key for dictionary patterns
            
        Returns:
            Updated pattern
        """
        if self.is_dict_pattern(pattern):
            if key is None:
                raise ValueError("Key required for dictionary pattern")
            if key not in pattern:
                raise ValueError(f"Key '{key}' not found in pattern")
            func_list = pattern[key]
        else:
            func_list = pattern
        
        if not (0 <= from_index < len(func_list)) or not (0 <= to_index < len(func_list)):
            raise ValueError("Invalid index for move operation")
        
        # Move the function
        func_entry = func_list.pop(from_index)
        func_list.insert(to_index, func_entry)
        
        return pattern
    
    def update_function_kwargs(self, pattern: Union[List, Dict], index: int, kwargs: Dict, key: Optional[str] = None) -> Union[List, Dict]:
        """
        Update function arguments in a pattern.
        
        Args:
            pattern: Pattern to update
            index: Index of function to update
            kwargs: New arguments
            key: Key for dictionary patterns
            
        Returns:
            Updated pattern
        """
        if self.is_dict_pattern(pattern):
            if key is None:
                raise ValueError("Key required for dictionary pattern")
            if key not in pattern:
                raise ValueError(f"Key '{key}' not found in pattern")
            func_list = pattern[key]
        else:
            func_list = pattern
        
        if 0 <= index < len(func_list):
            if isinstance(func_list[index], dict):
                func_list[index]['kwargs'] = kwargs
            else:
                # Convert simple function to dict format
                func_list[index] = {'func': func_list[index], 'kwargs': kwargs}
        else:
            raise ValueError(f"Index {index} out of range")
        
        return pattern
    
    def validate_pattern(self, pattern: Union[List, Dict]) -> Tuple[bool, Optional[str]]:
        """
        Validate a function pattern.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.validator.validate_function_pattern(pattern, "Pattern Editing Service")
            return True, None
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"
    
    def get_available_functions(self) -> List[Callable]:
        """Get list of available functions from the registry."""
        return list(self.func_registry.values())
    
    def get_function_info(self, func: Callable) -> Dict[str, Any]:
        """
        Get information about a function.
        
        Args:
            func: Function to get info for
            
        Returns:
            Dictionary with function information
        """
        if func is None:
            return {"name": "None", "backend": "", "signature": ""}
        
        # Find function in registry
        for name, registered_func in self.func_registry.items():
            if registered_func == func:
                return {
                    "name": name,
                    "backend": getattr(func, 'backend', ''),
                    "signature": str(func.__signature__ if hasattr(func, '__signature__') else ''),
                    "doc": func.__doc__ or ""
                }
        
        # Function not in registry
        return {
            "name": getattr(func, '__name__', 'Unknown'),
            "backend": getattr(func, 'backend', ''),
            "signature": str(func.__signature__ if hasattr(func, '__signature__') else ''),
            "doc": func.__doc__ or ""
        }
