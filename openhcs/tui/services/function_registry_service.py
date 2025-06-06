"""
Function Registry Service - Enhanced function discovery and metadata.

This service extends existing func_registry functionality with enhanced metadata
extraction and UI-specific formatting while avoiding import collisions.
"""

from typing import Dict, List, Tuple, Callable, Any, Optional
import logging

# Import from func_registry to avoid circular imports
from openhcs.processing.func_registry import FUNC_REGISTRY, get_function_info, get_functions_by_memory_type

logger = logging.getLogger(__name__)


class FunctionRegistryService:
    """
    Stateless service for function registry integration and metadata extraction.
    
    Extends existing FUNC_REGISTRY functionality with enhanced metadata
    and UI-specific formatting.
    """
    
    @staticmethod
    def get_functions_by_backend() -> Dict[str, List[Tuple[Callable, str]]]:
        """
        Group FUNC_REGISTRY functions by backend with memory type display names.
        
        EXACT backend grouping logic - preserves current behavior.
        
        Returns:
            Dict mapping backend names to lists of (function, display_name) tuples
        """
        functions_by_backend = {}

        # CORRECT: Iterate over memory types, then functions within each type
        for memory_type in FUNC_REGISTRY:
            for func in FUNC_REGISTRY[memory_type]:
                try:
                    # Get basic function info using existing registry
                    func_info = get_function_info(func)
                
                    backend = func_info.get('backend', 'unknown')
                    input_type = func_info.get('input_memory_type', 'unknown')
                    output_type = func_info.get('output_memory_type', 'unknown')

                    # Clean display format - show only function name, backend is shown separately
                    display_name = func.__name__

                    if backend not in functions_by_backend:
                        functions_by_backend[backend] = []

                    functions_by_backend[backend].append((func, display_name))

                except Exception as e:
                    logger.warning(f"Failed to get info for function {func}: {e}")
                    # Add to unknown backend as fallback
                    if 'unknown' not in functions_by_backend:
                        functions_by_backend['unknown'] = []
                    functions_by_backend['unknown'].append((func, func.__name__))
        
        return functions_by_backend
    
    @staticmethod
    def get_enhanced_function_metadata(func: Callable) -> Dict[str, Any]:
        """
        Enhanced version of get_function_info with validation and special inputs/outputs.
        
        RENAMED to avoid import collision with existing get_function_info.
        
        Args:
            func: Function to analyze
            
        Returns:
            Enhanced metadata dict with same structure as get_function_info plus validation
        """
        try:
            # Start with existing function info
            base_info = get_function_info(func)
            
            # Enhance with additional metadata
            enhanced_info = {
                'name': base_info.get('name', func.__name__),
                'backend': base_info.get('backend', 'unknown'),
                'input_memory_type': base_info.get('input_memory_type', 'unknown'),
                'output_memory_type': base_info.get('output_memory_type', 'unknown'),
                'special_inputs': base_info.get('special_inputs', []),
                'special_outputs': base_info.get('special_outputs', []),
            }
            
            # Add validation status
            enhanced_info['is_valid'] = True
            enhanced_info['validation_errors'] = []
            
            # Basic validation checks
            if enhanced_info['backend'] == 'unknown':
                enhanced_info['validation_errors'].append("Unknown backend")
            
            if enhanced_info['input_memory_type'] == 'unknown':
                enhanced_info['validation_errors'].append("Unknown input memory type")
            
            if enhanced_info['output_memory_type'] == 'unknown':
                enhanced_info['validation_errors'].append("Unknown output memory type")
            
            if enhanced_info['validation_errors']:
                enhanced_info['is_valid'] = False
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Failed to get enhanced metadata for {func}: {e}")
            # Return minimal fallback info
            return {
                'name': getattr(func, '__name__', 'unknown'),
                'backend': 'unknown',
                'input_memory_type': 'unknown',
                'output_memory_type': 'unknown',
                'special_inputs': [],
                'special_outputs': [],
                'is_valid': False,
                'validation_errors': [f"Metadata extraction failed: {e}"]
            }
    
    @staticmethod
    def create_dropdown_options(functions_by_backend: Dict[str, List[Tuple[Callable, str]]]) -> List[Tuple[str, List[Tuple[Callable, str]]]]:
        """
        Format function groups for GroupedDropdown component.
        
        Args:
            functions_by_backend: Output from get_functions_by_backend()
            
        Returns:
            List of (group_name, options) tuples for GroupedDropdown
        """
        dropdown_options = []
        
        # Sort backends for consistent ordering
        sorted_backends = sorted(functions_by_backend.keys())
        
        for backend in sorted_backends:
            functions = functions_by_backend[backend]
            # Sort functions by display name for consistent ordering
            sorted_functions = sorted(functions, key=lambda x: x[1])
            dropdown_options.append((backend, sorted_functions))
        
        return dropdown_options
    
    @staticmethod
    def find_default_function() -> Optional[Callable]:
        """
        Get first available function for new items.

        Returns:
            First function from FUNC_REGISTRY, or None if empty
        """
        # CORRECT: Iterate through memory types to find first function
        for memory_type in FUNC_REGISTRY:
            if FUNC_REGISTRY[memory_type]:  # Check if list is not empty
                return FUNC_REGISTRY[memory_type][0]
        return None
    
    @staticmethod
    def get_functions_by_memory_type_wrapper(memory_type: str) -> List[Callable]:
        """
        Wrapper around existing get_functions_by_memory_type for consistency.
        
        Args:
            memory_type: Memory type to filter by
            
        Returns:
            List of functions matching the memory type
        """
        try:
            return get_functions_by_memory_type(memory_type)
        except Exception as e:
            logger.error(f"Failed to get functions by memory type {memory_type}: {e}")
            return []
    
    @staticmethod
    def validate_function_in_registry(func: Callable) -> bool:
        """
        Check if function exists in FUNC_REGISTRY.

        Args:
            func: Function to check

        Returns:
            True if function is in registry, False otherwise
        """
        # CORRECT: Check if function exists in any memory type list
        for memory_type in FUNC_REGISTRY:
            if func in FUNC_REGISTRY[memory_type]:
                return True
        return False
    
    @staticmethod
    def get_function_display_name(func: Callable) -> str:
        """
        Get formatted display name for function.
        
        Args:
            func: Function to format
            
        Returns:
            Formatted display name with memory types
        """
        try:
            # Clean display format - just return function name
            # Backend information is displayed separately in the UI
            return func.__name__
        except Exception:
            return func.__name__
