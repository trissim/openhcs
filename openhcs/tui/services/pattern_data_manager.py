"""
Pattern Data Manager - Pure data operations for function patterns.

This service handles pattern data structure operations and transformations
with order determinism and immutable operations.
"""

import copy
from typing import Union, List, Dict, Tuple, Optional, Callable, Any


class PatternDataManager:
    """
    Pure data operations for function patterns.
    
    Handles List↔Dict conversions, cloning, and data transformations
    with order determinism and immutable operations.
    """
    
    @staticmethod
    def clone_pattern(pattern: Union[List, Dict]) -> Union[List, Dict]:
        """
        Deep clone preserving callable references exactly.
        
        Args:
            pattern: Pattern to clone (List or Dict)
            
        Returns:
            Deep cloned pattern with preserved callable references
        """
        if pattern is None:
            return []
        return copy.deepcopy(pattern)
    
    @staticmethod
    def convert_list_to_dict(pattern: List) -> Dict:
        """
        Convert List pattern to empty Dict - user must add component keys manually.

        Args:
            pattern: List pattern to convert (will be discarded)

        Returns:
            Empty dict for user to populate with experimental component identifiers
        """
        if not isinstance(pattern, list):
            raise ValueError(f"Expected list, got {type(pattern)}")

        # Return empty dict - user will add experimental component keys manually
        return {}
    
    @staticmethod
    def convert_dict_to_list(pattern: Dict) -> Union[List, Dict]:
        """
        Convert Dict pattern to List when empty.

        Args:
            pattern: Dict pattern to potentially convert

        Returns:
            Empty list if dict is empty, otherwise returns original dict
        """
        if not isinstance(pattern, dict):
            raise ValueError(f"Expected dict, got {type(pattern)}")

        # Convert to empty list if dict is empty
        if not pattern:
            return []

        # Keep as dict if it has keys
        return pattern
    
    @staticmethod
    def extract_func_and_kwargs(func_item) -> Tuple[Optional[Callable], Dict]:
        """
        Parse (func, kwargs) tuples and bare callables.
        
        Handles both tuple format and bare callable format exactly as current logic.
        
        Args:
            func_item: Either (callable, kwargs) tuple or bare callable
            
        Returns:
            Tuple of (callable, kwargs_dict)
        """
        # EXACT current logic preservation
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            return func_item[0], func_item[1]
        elif callable(func_item):
            return func_item, {}
        else:
            return None, {}
    
    @staticmethod
    def validate_pattern_structure(pattern: Union[List, Dict]) -> bool:
        """
        Basic structural validation of pattern.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        if pattern is None:
            return True
        
        if isinstance(pattern, list):
            # Validate list items are callables or (callable, dict) tuples
            for item in pattern:
                func, kwargs = PatternDataManager.extract_func_and_kwargs(item)
                if func is None:
                    return False
                if not isinstance(kwargs, dict):
                    return False
            return True
        
        elif isinstance(pattern, dict):
            # Validate dict values are lists of callables
            for key, value in pattern.items():
                if not isinstance(value, list):
                    return False
                # Recursively validate the list
                if not PatternDataManager.validate_pattern_structure(value):
                    return False
            return True
        
        else:
            return False
    
    @staticmethod
    def get_current_functions(pattern: Union[List, Dict], key: Any, is_dict: bool) -> List:
        """
        Extract function list for current context.
        
        Args:
            pattern: Full pattern (List or Dict)
            key: Current key (for Dict patterns)
            is_dict: Whether pattern is currently in dict mode
            
        Returns:
            List of functions for current context
        """
        if is_dict and isinstance(pattern, dict):
            return pattern.get(key, [])
        elif not is_dict and isinstance(pattern, list):
            return pattern
        else:
            return []
    
    @staticmethod
    def update_pattern_functions(pattern: Union[List, Dict], key: Any, is_dict: bool, 
                               new_functions: List) -> Union[List, Dict]:
        """
        Update functions in pattern for current context.
        
        Returns new pattern object (immutable operation).
        
        Args:
            pattern: Original pattern
            key: Current key (for Dict patterns)
            is_dict: Whether pattern is in dict mode
            new_functions: New function list
            
        Returns:
            New pattern with updated functions
        """
        if is_dict and isinstance(pattern, dict):
            new_pattern = copy.deepcopy(pattern)
            new_pattern[key] = new_functions
            return new_pattern
        elif not is_dict and isinstance(pattern, list):
            return copy.deepcopy(new_functions)
        else:
            # Fallback - return original pattern
            return copy.deepcopy(pattern)
    
    @staticmethod
    def add_new_key(pattern: Dict, new_key: str) -> Dict:
        """
        Add new key to dict pattern.
        
        Args:
            pattern: Dict pattern
            new_key: Key to add
            
        Returns:
            New dict with added key
        """
        new_pattern = copy.deepcopy(pattern)
        if new_key not in new_pattern:
            new_pattern[new_key] = []
        return new_pattern
    
    @staticmethod
    def remove_key(pattern: Dict, key_to_remove: Any) -> Union[List, Dict]:
        """
        Remove key from dict pattern.

        If dict becomes empty after removal, converts back to list.

        Args:
            pattern: Dict pattern
            key_to_remove: Key to remove

        Returns:
            New pattern (List if dict becomes empty, Dict otherwise)
        """
        new_pattern = copy.deepcopy(pattern)
        if key_to_remove in new_pattern:
            del new_pattern[key_to_remove]

        # Check if should convert back to list (when empty)
        return PatternDataManager.convert_dict_to_list(new_pattern)
