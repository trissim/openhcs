"""
File operation utilities for hybrid TUI.

Provides functions for loading and saving function patterns, steps, and pipelines
with proper error handling and type validation.
"""

import pickle
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Callable

logger = logging.getLogger(__name__)

async def load_func_pattern(file_path: Path) -> Optional[Any]:
    """
    Load function pattern from .func file.
    
    Args:
        file_path: Path to .func file
        
    Returns:
        Loaded pattern or None if failed
    """
    try:
        if not file_path.exists():
            logger.error(f"Function pattern file not found: {file_path}")
            return None
            
        if file_path.suffix != '.func':
            logger.warning(f"Expected .func file, got: {file_path.suffix}")
            
        with open(file_path, 'rb') as f:
            pattern = pickle.load(f)
            
        logger.info(f"Loaded function pattern from {file_path}")
        return pattern
        
    except Exception as e:
        logger.error(f"Failed to load function pattern from {file_path}: {e}")
        return None

async def save_func_pattern(pattern: Any, file_path: Path) -> bool:
    """
    Save function pattern to .func file.
    
    Args:
        pattern: Function pattern to save
        file_path: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure .func extension
        if file_path.suffix != '.func':
            file_path = file_path.with_suffix('.func')
            
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(pattern, f)
            
        logger.info(f"Saved function pattern to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save function pattern to {file_path}: {e}")
        return False

async def load_step_file(file_path: Path) -> Optional[Any]:
    """
    Load step object from .step file.
    
    Args:
        file_path: Path to .step file
        
    Returns:
        Loaded step or None if failed
    """
    try:
        if not file_path.exists():
            logger.error(f"Step file not found: {file_path}")
            return None
            
        if file_path.suffix != '.step':
            logger.warning(f"Expected .step file, got: {file_path.suffix}")
            
        with open(file_path, 'rb') as f:
            step = pickle.load(f)
            
        logger.info(f"Loaded step from {file_path}")
        return step
        
    except Exception as e:
        logger.error(f"Failed to load step from {file_path}: {e}")
        return None

async def save_step_file(step: Any, file_path: Path) -> bool:
    """
    Save step object to .step file.
    
    Args:
        step: Step object to save
        file_path: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure .step extension
        if file_path.suffix != '.step':
            file_path = file_path.with_suffix('.step')
            
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(step, f)
            
        logger.info(f"Saved step to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save step to {file_path}: {e}")
        return False

async def load_pipeline_file(file_path: Path) -> Optional[Any]:
    """
    Load pipeline from .pipeline file.
    
    Args:
        file_path: Path to .pipeline file
        
    Returns:
        Loaded pipeline or None if failed
    """
    try:
        if not file_path.exists():
            logger.error(f"Pipeline file not found: {file_path}")
            return None
            
        if file_path.suffix != '.pipeline':
            logger.warning(f"Expected .pipeline file, got: {file_path.suffix}")
            
        with open(file_path, 'rb') as f:
            pipeline = pickle.load(f)
            
        logger.info(f"Loaded pipeline from {file_path}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load pipeline from {file_path}: {e}")
        return None

async def save_pipeline_file(pipeline: Any, file_path: Path) -> bool:
    """
    Save pipeline to .pipeline file.
    
    Args:
        pipeline: Pipeline object to save
        file_path: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure .pipeline extension
        if file_path.suffix != '.pipeline':
            file_path = file_path.with_suffix('.pipeline')
            
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline, f)
            
        logger.info(f"Saved pipeline to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save pipeline to {file_path}: {e}")
        return False

def validate_pattern_structure(pattern: Any) -> Dict[str, Any]:
    """
    Validate function pattern structure.
    
    Args:
        pattern: Pattern to validate
        
    Returns:
        Dict with validation results
    """
    try:
        if callable(pattern):
            return {'valid': True, 'type': 'single_function'}
        elif isinstance(pattern, tuple) and len(pattern) == 2:
            func, kwargs = pattern
            if callable(func) and isinstance(kwargs, dict):
                return {'valid': True, 'type': 'function_with_kwargs'}
            else:
                return {'valid': False, 'error': 'Invalid tuple structure'}
        elif isinstance(pattern, list):
            for i, item in enumerate(pattern):
                if not (callable(item) or (isinstance(item, tuple) and len(item) == 2)):
                    return {'valid': False, 'error': f'Invalid item at index {i}'}
            return {'valid': True, 'type': 'function_list'}
        elif isinstance(pattern, dict):
            for key, value in pattern.items():
                if not isinstance(key, str):
                    return {'valid': False, 'error': f'Non-string key: {key}'}
                sub_result = validate_pattern_structure(value)
                if not sub_result['valid']:
                    return {'valid': False, 'error': f'Invalid value for key {key}: {sub_result["error"]}'}
            return {'valid': True, 'type': 'function_dict'}
        else:
            return {'valid': False, 'error': f'Unsupported pattern type: {type(pattern)}'}
            
    except Exception as e:
        return {'valid': False, 'error': f'Validation error: {e}'}
