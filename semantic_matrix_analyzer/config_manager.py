"""
Configuration Manager for Semantic Matrix Analyzer.

This module provides a dynamic configuration interface for adjusting weights,
patterns, keys, and tokens used by the Semantic Matrix Analyzer.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default_config.yaml"

class ConfigManager:
    """
    Dynamic configuration manager for Semantic Matrix Analyzer.
    
    This class provides an interface for loading, modifying, and saving
    configuration settings for the Semantic Matrix Analyzer.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self._ensure_config_directory()
        
    def _ensure_config_directory(self) -> None:
        """Ensure the configuration directory exists."""
        config_dir = self.config_path.parent
        if not config_dir.exists():
            config_dir.mkdir(parents=True)
            
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            return self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
                
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            "analysis": {
                "weights": {
                    "name_similarity": 0.7,
                    "type_compatibility": 0.8,
                    "structural_coherence": 0.9,
                    "docstring_relevance": 0.6,
                    "import_dependency": 0.5
                },
                "patterns": {
                    "naming_conventions": {
                        "class": r"^[A-Z][a-zA-Z0-9]*$",
                        "function": r"^[a-z][a-zA-Z0-9_]*$",
                        "constant": r"^[A-Z][A-Z0-9_]*$",
                        "variable": r"^[a-z][a-zA-Z0-9_]*$"
                    },
                    "code_structure": {
                        "max_function_length": 100,
                        "max_class_length": 500,
                        "max_line_length": 88,
                        "max_complexity": 10
                    }
                },
                "tokens": {
                    "special_markers": ["TODO", "FIXME", "NOTE", "WARNING"],
                    "docstring_tags": ["param", "return", "raises", "yields", "example"]
                },
                "keys": {
                    "intent_indicators": ["purpose", "goal", "objective", "aim", "intent"],
                    "error_indicators": ["error", "exception", "fail", "invalid", "wrong"]
                }
            },
            "visualization": {
                "colors": {
                    "high_confidence": "#4CAF50",
                    "medium_confidence": "#FFC107",
                    "low_confidence": "#F44336"
                },
                "thresholds": {
                    "high_confidence": 0.8,
                    "medium_confidence": 0.5
                }
            },
            "output": {
                "formats": ["text", "markdown", "json"],
                "default_format": "markdown",
                "verbosity": 2
            }
        }
        
        # Save default config
        self.save_config(default_config)
        return default_config
        
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save. If None, saves the current config.
        """
        config_to_save = config if config is not None else self.config
        
        with open(self.config_path, 'w') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_to_save, f, default_flow_style=False)
            else:
                json.dump(config_to_save, f, indent=2)
                
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._recursive_update(self.config, updates)
        self.save_config()
        
    def _recursive_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionaries.
        
        Args:
            target: Target dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value
                
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._create_default_config()
        self.save_config()
        
    def get_weight(self, weight_name: str) -> float:
        """
        Get a specific weight value.
        
        Args:
            weight_name: Name of the weight
            
        Returns:
            Weight value
        """
        return self.config["analysis"]["weights"].get(weight_name, 0.5)
        
    def set_weight(self, weight_name: str, value: float) -> None:
        """
        Set a specific weight value.
        
        Args:
            weight_name: Name of the weight
            value: New weight value
        """
        self.config["analysis"]["weights"][weight_name] = value
        self.save_config()
        
    def get_pattern(self, category: str, pattern_name: str) -> Any:
        """
        Get a specific pattern.
        
        Args:
            category: Pattern category
            pattern_name: Pattern name
            
        Returns:
            Pattern value
        """
        return self.config["analysis"]["patterns"][category].get(pattern_name)
        
    def set_pattern(self, category: str, pattern_name: str, value: Any) -> None:
        """
        Set a specific pattern.
        
        Args:
            category: Pattern category
            pattern_name: Pattern name
            value: New pattern value
        """
        if category not in self.config["analysis"]["patterns"]:
            self.config["analysis"]["patterns"][category] = {}
        self.config["analysis"]["patterns"][category][pattern_name] = value
        self.save_config()
        
    def get_tokens(self, token_category: str) -> List[str]:
        """
        Get tokens for a specific category.
        
        Args:
            token_category: Token category
            
        Returns:
            List of tokens
        """
        return self.config["analysis"]["tokens"].get(token_category, [])
        
    def add_token(self, token_category: str, token: str) -> None:
        """
        Add a token to a category.
        
        Args:
            token_category: Token category
            token: Token to add
        """
        if token_category not in self.config["analysis"]["tokens"]:
            self.config["analysis"]["tokens"][token_category] = []
        
        if token not in self.config["analysis"]["tokens"][token_category]:
            self.config["analysis"]["tokens"][token_category].append(token)
            self.save_config()
            
    def remove_token(self, token_category: str, token: str) -> None:
        """
        Remove a token from a category.
        
        Args:
            token_category: Token category
            token: Token to remove
        """
        if (token_category in self.config["analysis"]["tokens"] and 
            token in self.config["analysis"]["tokens"][token_category]):
            self.config["analysis"]["tokens"][token_category].remove(token)
            self.save_config()
            
    def get_keys(self, key_category: str) -> List[str]:
        """
        Get keys for a specific category.
        
        Args:
            key_category: Key category
            
        Returns:
            List of keys
        """
        return self.config["analysis"]["keys"].get(key_category, [])
        
    def add_key(self, key_category: str, key: str) -> None:
        """
        Add a key to a category.
        
        Args:
            key_category: Key category
            key: Key to add
        """
        if key_category not in self.config["analysis"]["keys"]:
            self.config["analysis"]["keys"][key_category] = []
        
        if key not in self.config["analysis"]["keys"][key_category]:
            self.config["analysis"]["keys"][key_category].append(key)
            self.save_config()
            
    def remove_key(self, key_category: str, key: str) -> None:
        """
        Remove a key from a category.
        
        Args:
            key_category: Key category
            key: Key to remove
        """
        if (key_category in self.config["analysis"]["keys"] and 
            key in self.config["analysis"]["keys"][key_category]):
            self.config["analysis"]["keys"][key_category].remove(key)
            self.save_config()
