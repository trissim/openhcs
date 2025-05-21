"""
Configuration module for agent-driven analysis.

This module provides configuration management for the agent-driven analyzer,
allowing customization of defaults, primitives, and custom functions.
"""

import importlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from semantic_matrix_analyzer.agent.models import FileComplexity, FileRelevance


@dataclass
class AgentConfig:
    """Configuration for agent-driven analysis."""
    
    # Selection thresholds
    selection_threshold: float = 0.5
    min_relevance_score: float = 0.1
    min_information_value: float = 0.1
    max_effort_multiplier: float = 3.0
    
    # Relevance weights
    explicit_mention_weight: float = 1.0
    component_match_weight: float = 0.8
    central_file_weight: float = 0.7
    historical_usefulness_weight: float = 0.6
    
    # Information value weights
    complexity_weight: float = 0.7
    dependency_weight: float = 0.6
    change_frequency_weight: float = 0.5
    
    # Effort multipliers
    large_file_multiplier: float = 1.5
    very_large_file_multiplier: float = 2.0
    complex_file_multiplier: float = 1.5
    very_complex_file_multiplier: float = 2.0
    many_dependencies_multiplier: float = 1.5
    
    # Thresholds
    large_file_threshold: int = 500
    very_large_file_threshold: int = 1000
    many_dependencies_threshold: int = 10
    high_change_frequency: float = 5.0
    medium_change_frequency: float = 2.0
    
    # Custom functions
    calculate_relevance_func: Optional[Callable] = None
    calculate_information_value_func: Optional[Callable] = None
    calculate_effort_func: Optional[Callable] = None
    
    # File extensions to analyze
    file_extensions: Set[str] = field(default_factory=lambda: {".py"})
    
    # Maximum files to analyze
    max_files: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            An AgentConfig instance.
        """
        # Create a default config
        config = cls()
        
        # Update primitive values
        for key, value in config_dict.items():
            if key in cls.__dataclass_fields__ and not key.endswith('_func'):
                if isinstance(value, (int, float, bool, str)) or value is None:
                    setattr(config, key, value)
                elif key == 'file_extensions' and isinstance(value, list):
                    config.file_extensions = set(value)
        
        # Handle custom functions
        if 'custom_functions' in config_dict:
            custom_funcs = config_dict['custom_functions']
            
            # Load calculate_relevance function
            if 'calculate_relevance' in custom_funcs:
                config.calculate_relevance_func = cls._load_function(
                    custom_funcs['calculate_relevance']
                )
            
            # Load calculate_information_value function
            if 'calculate_information_value' in custom_funcs:
                config.calculate_information_value_func = cls._load_function(
                    custom_funcs['calculate_information_value']
                )
            
            # Load calculate_effort function
            if 'calculate_effort' in custom_funcs:
                config.calculate_effort_func = cls._load_function(
                    custom_funcs['calculate_effort']
                )
        
        return config
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'AgentConfig':
        """Load configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file.
            
        Returns:
            An AgentConfig instance.
            
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the JSON file is invalid.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            A dictionary representation of the configuration.
        """
        config_dict = {}
        
        # Add primitive values
        for key, field_info in self.__dataclass_fields__.items():
            if not key.endswith('_func'):
                value = getattr(self, key)
                
                # Convert sets to lists for JSON serialization
                if isinstance(value, set):
                    value = list(value)
                
                config_dict[key] = value
        
        # Add custom function information
        custom_funcs = {}
        
        if self.calculate_relevance_func:
            custom_funcs['calculate_relevance'] = self._get_function_path(
                self.calculate_relevance_func
            )
        
        if self.calculate_information_value_func:
            custom_funcs['calculate_information_value'] = self._get_function_path(
                self.calculate_information_value_func
            )
        
        if self.calculate_effort_func:
            custom_funcs['calculate_effort'] = self._get_function_path(
                self.calculate_effort_func
            )
        
        if custom_funcs:
            config_dict['custom_functions'] = custom_funcs
        
        return config_dict
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save the configuration to a JSON file.
        
        Args:
            json_path: Path to save the JSON configuration file.
            
        Raises:
            IOError: If the JSON file cannot be written.
        """
        config_dict = self.to_dict()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def _load_function(function_path: str) -> Callable:
        """Load a function from a module path.
        
        Args:
            function_path: Path to the function in the format 'module.submodule:function_name'.
            
        Returns:
            The loaded function.
            
        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the function cannot be found in the module.
        """
        try:
            module_path, function_name = function_path.split(':')
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error loading function {function_path}: {e}")
            raise
    
    @staticmethod
    def _get_function_path(func: Callable) -> str:
        """Get the module path for a function.
        
        Args:
            func: The function to get the path for.
            
        Returns:
            The module path in the format 'module.submodule:function_name'.
        """
        module = func.__module__
        name = func.__name__
        return f"{module}:{name}"


def generate_default_config(output_path: Union[str, Path]) -> None:
    """Generate a default configuration file.
    
    Args:
        output_path: Path to save the default configuration file.
        
    Raises:
        IOError: If the configuration file cannot be written.
    """
    config = AgentConfig()
    config.to_json(output_path)
    logging.info(f"Default configuration saved to {output_path}")


def load_config(config_path: Union[str, Path]) -> AgentConfig:
    """Load a configuration from a file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        The loaded configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file is invalid.
    """
    return AgentConfig.from_json(config_path)
