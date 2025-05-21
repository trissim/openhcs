"""
Configuration system for the Structural Intent Analysis.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class Configuration:
    """Configuration for the Structural Intent Analysis system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the configuration.
        
        Args:
            config_dict: A dictionary of configuration options (optional).
        """
        # Default configuration
        self.config = {
            "name_analysis": {
                "enabled": True,
                "tokenization": {
                    "separators": ["_", "-", " "],
                    "normalize_tokens": True
                },
                "semantic_extraction": {
                    "action_verbs": {
                        "get": "Retrieve or access",
                        "set": "Modify or update",
                        "create": "Create or instantiate",
                        "delete": "Remove or destroy",
                        "update": "Modify or change",
                        "validate": "Check or verify",
                        "process": "Handle or transform",
                        "calculate": "Compute or determine",
                        "find": "Search or locate",
                        "check": "Verify or test",
                        "is": "Test condition",
                        "has": "Test possession",
                        "can": "Test capability",
                        "should": "Test recommendation",
                        "will": "Indicate future action",
                        "do": "Perform action"
                    },
                    "design_patterns": {
                        "factory": "Create objects",
                        "builder": "Construct complex objects",
                        "singleton": "Ensure single instance",
                        "adapter": "Convert interface",
                        "decorator": "Add responsibilities",
                        "observer": "Notify of changes",
                        "strategy": "Define algorithm family",
                        "command": "Encapsulate request",
                        "iterator": "Access elements",
                        "composite": "Treat objects uniformly",
                        "proxy": "Control access",
                        "facade": "Simplify interface"
                    },
                    "domain_objects": {
                        "user": "User or account",
                        "customer": "Client or buyer",
                        "order": "Purchase or request",
                        "product": "Item or good",
                        "service": "Functionality or offering",
                        "transaction": "Exchange or operation",
                        "payment": "Financial transaction",
                        "account": "User profile or financial account",
                        "message": "Communication or notification",
                        "event": "Occurrence or happening",
                        "request": "Ask or demand",
                        "response": "Answer or reply",
                        "data": "Information or content",
                        "config": "Configuration or settings",
                        "manager": "Controller or supervisor",
                        "handler": "Processor or responder",
                        "provider": "Supplier or source",
                        "consumer": "User or recipient"
                    }
                },
                "confidence": {
                    "base_confidence": 0.5,
                    "compound_name_bonus": 0.1,
                    "meaningful_token_bonus": 0.2,
                    "class_name_bonus": 0.1,
                    "method_name_bonus": 0.1
                }
            },
            "type_analysis": {
                "enabled": True,
                "type_mappings": {
                    "str": ["String", "Textual data", "ENTITY"],
                    "int": ["Integer", "Numeric data", "ENTITY"],
                    "float": ["Float", "Numeric data with decimal precision", "ENTITY"],
                    "bool": ["Boolean", "True/False condition", "STATE"],
                    "list": ["List", "Collection of items", "ENTITY"],
                    "dict": ["Dictionary", "Key-value mapping", "ENTITY"],
                    "set": ["Set", "Unique collection of items", "ENTITY"],
                    "tuple": ["Tuple", "Immutable collection of items", "ENTITY"],
                    "None": ["None", "No value", "STATE"],
                    "Any": ["Any", "Any type", "ENTITY"],
                    "Optional": ["Optional", "May be None", "STATE"],
                    "Union": ["Union", "One of several types", "ENTITY"],
                    "Callable": ["Callable", "Function or method", "ACTION"],
                    "Iterator": ["Iterator", "Sequence that can be iterated", "ENTITY"],
                    "Iterable": ["Iterable", "Can be iterated over", "ENTITY"],
                    "Generator": ["Generator", "Generates values on demand", "ACTION"],
                    "Type": ["Type", "Class or type object", "ENTITY"],
                    "Path": ["Path", "File system path", "ENTITY"],
                    "datetime": ["Datetime", "Date and time", "ENTITY"],
                    "date": ["Date", "Calendar date", "ENTITY"],
                    "time": ["Time", "Time of day", "ENTITY"],
                    "timedelta": ["Timedelta", "Duration", "ENTITY"],
                    "Exception": ["Exception", "Error condition", "STATE"],
                    "Pattern": ["Pattern", "Regular expression pattern", "ENTITY"],
                    "Match": ["Match", "Regular expression match", "ENTITY"]
                },
                "confidence": {
                    "base_confidence": 0.6,
                    "union_optional_bonus": 0.1,
                    "collection_bonus": 0.1,
                    "custom_type_bonus": 0.2
                }
            },
            "structural_analysis": {
                "enabled": True,
                "patterns": {
                    "layered_architecture": {
                        "enabled": True,
                        "layer_names": ["presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"],
                        "confidence": 0.7
                    },
                    "microservices_architecture": {
                        "enabled": True,
                        "service_indicators": ["service", "api", "client", "server"],
                        "confidence": 0.6
                    },
                    "event_driven_architecture": {
                        "enabled": True,
                        "event_indicators": ["event", "listener", "handler", "subscriber", "publisher"],
                        "confidence": 0.6
                    },
                    "mvc_architecture": {
                        "enabled": True,
                        "model_indicators": ["model", "entity", "domain"],
                        "view_indicators": ["view", "template", "page", "screen"],
                        "controller_indicators": ["controller", "handler"],
                        "confidence": 0.7
                    },
                    "repository_pattern": {
                        "enabled": True,
                        "repository_indicators": ["repository", "repo", "dao", "data_access"],
                        "confidence": 0.6
                    },
                    "factory_pattern": {
                        "enabled": True,
                        "factory_indicators": ["factory", "creator", "builder"],
                        "confidence": 0.6
                    },
                    "singleton_pattern": {
                        "enabled": True,
                        "singleton_indicators": ["singleton", "instance"],
                        "confidence": 0.6
                    }
                }
            },
            "integration": {
                "combine_intents": True,
                "build_hierarchy": True,
                "report_format": "text",
                "min_confidence": 0.3,
                "max_results": 100
            }
        }
        
        # Update with provided configuration
        if config_dict:
            self._update_config(config_dict)
    
    def _update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update the configuration with a dictionary.
        
        Args:
            config_dict: A dictionary of configuration options.
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        update_nested_dict(self.config, config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Configuration':
        """Load configuration from a file.
        
        Args:
            file_path: The path to the configuration file.
            
        Returns:
            A Configuration object.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return cls()
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                except ImportError:
                    logger.error("PyYAML is not installed. Please install it to use YAML configuration files.")
                    return cls()
            elif file_path.suffix.lower() == '.py':
                # Load Python module
                import importlib.util
                spec = importlib.util.spec_from_file_location("config_module", file_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                # Get configuration dictionary
                if hasattr(config_module, 'CONFIG'):
                    config_dict = config_module.CONFIG
                else:
                    logger.warning(f"No CONFIG variable found in Python configuration file: {file_path}")
                    config_dict = {}
            else:
                logger.warning(f"Unsupported configuration file format: {file_path}")
                return cls()
            
            return cls(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return cls()
    
    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """Save configuration to a file.
        
        Args:
            file_path: The path to save the configuration to.
            
        Returns:
            True if the configuration was saved successfully, False otherwise.
        """
        file_path = Path(file_path)
        
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                except ImportError:
                    logger.error("PyYAML is not installed. Please install it to use YAML configuration files.")
                    return False
            elif file_path.suffix.lower() == '.py':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# Configuration for Structural Intent Analysis\n\n")
                    f.write("CONFIG = ")
                    f.write(repr(self.config))
                    f.write("\n")
            else:
                logger.warning(f"Unsupported configuration file format: {file_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key (dot-separated for nested keys).
            default: The default value to return if the key is not found.
            
        Returns:
            The configuration value, or the default value if not found.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key (dot-separated for nested keys).
            value: The value to set.
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate(self) -> List[str]:
        """Validate the configuration.
        
        Returns:
            A list of validation errors, or an empty list if the configuration is valid.
        """
        errors = []
        
        # Check required sections
        required_sections = ["name_analysis", "type_analysis", "structural_analysis", "integration"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Check enabled flags
        for section in ["name_analysis", "type_analysis", "structural_analysis"]:
            if section in self.config and "enabled" not in self.config[section]:
                errors.append(f"Missing 'enabled' flag in section: {section}")
        
        # Check integration settings
        if "integration" in self.config:
            integration = self.config["integration"]
            
            if "min_confidence" in integration:
                min_confidence = integration["min_confidence"]
                if not isinstance(min_confidence, (int, float)) or min_confidence < 0 or min_confidence > 1:
                    errors.append(f"Invalid min_confidence: {min_confidence} (must be between 0 and 1)")
            
            if "max_results" in integration:
                max_results = integration["max_results"]
                if not isinstance(max_results, int) or max_results <= 0:
                    errors.append(f"Invalid max_results: {max_results} (must be a positive integer)")
        
        return errors


class ConfigurableAnalyzer:
    """Base class for configurable analyzers."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable analyzer.
        
        Args:
            config: The configuration to use (optional).
        """
        self.config = config or Configuration()
    
    def is_enabled(self) -> bool:
        """Check if the analyzer is enabled.
        
        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        return True  # Override in subclasses
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key (relative to the analyzer's section).
            default: The default value to return if the key is not found.
            
        Returns:
            The configuration value, or the default value if not found.
        """
        section = self._get_config_section()
        full_key = f"{section}.{key}" if section else key
        return self.config.get(full_key, default)
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return ""  # Override in subclasses
