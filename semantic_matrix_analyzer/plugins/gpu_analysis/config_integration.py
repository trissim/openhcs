"""
Configuration Integration Module for GPU Analysis Plugin.

This module provides integration with SMA's configuration system, ensuring
consistent configuration management and enabling the implementation of
unimplemented SMA methods.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Define configuration schema
GPU_ANALYSIS_CONFIG_SCHEMA = {
    "gpu_analysis": {
        "device": {
            "type": "string",
            "default": "cuda",
            "description": "Device to use for analysis ('cuda' or 'cpu')",
            "enum": ["cuda", "cpu"]
        },
        "batch_size": {
            "type": "integer",
            "default": 32,
            "description": "Batch size for GPU operations",
            "minimum": 1,
            "maximum": 1024
        },
        "precision": {
            "type": "string",
            "default": "float32",
            "description": "Precision for GPU operations",
            "enum": ["float16", "float32", "float64"]
        },
        "cache_size": {
            "type": "integer",
            "default": 1024,
            "description": "Maximum number of items to cache",
            "minimum": 0,
            "maximum": 10000
        },
        "analyzers": {
            "type": "object",
            "default": {
                "complexity": {
                    "enabled": True,
                    "weights": {
                        "cyclomatic": 1.0,
                        "cognitive": 1.0,
                        "halstead": 0.5
                    }
                },
                "dependency": {
                    "enabled": True,
                    "max_depth": 3
                },
                "semantic": {
                    "enabled": True,
                    "embedding_model": "default",
                    "similarity_threshold": 0.7
                },
                "pattern": {
                    "enabled": True,
                    "confidence_threshold": 0.6
                }
            },
            "description": "Configuration for analyzers"
        },
        "patterns": {
            "type": "array",
            "default": [],
            "description": "Custom patterns to match",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "pattern_type": {"type": "string"},
                    "pattern": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["name", "pattern"]
            }
        },
        "intents": {
            "type": "array",
            "default": [],
            "description": "Custom intents to align with",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "patterns"]
            }
        }
    }
}

def get_gpu_config_from_sma(sma_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract GPU Analysis Plugin configuration from SMA configuration.

    This function extracts the GPU Analysis Plugin configuration from the SMA
    configuration, applying defaults where necessary.

    Args:
        sma_config: SMA configuration dictionary

    Returns:
        GPU Analysis Plugin configuration dictionary
    """
    try:
        # Extract GPU Analysis Plugin configuration
        gpu_config = sma_config.get("gpu_analysis", {})

        # Apply defaults for missing values
        defaults = {
            "device": "cuda",
            "batch_size": 32,
            "precision": "float32",
            "cache_size": 1024,
            "analyzers": {
                "complexity": {
                    "enabled": True,
                    "weights": {
                        "cyclomatic": 1.0,
                        "cognitive": 1.0,
                        "halstead": 0.5
                    }
                },
                "dependency": {
                    "enabled": True,
                    "max_depth": 3
                },
                "semantic": {
                    "enabled": True,
                    "embedding_model": "default",
                    "similarity_threshold": 0.7
                },
                "pattern": {
                    "enabled": True,
                    "confidence_threshold": 0.6
                }
            },
            "patterns": [],
            "intents": []
        }

        # Apply defaults recursively
        def apply_defaults(config, defaults):
            result = {}
            for key, default_value in defaults.items():
                if key in config:
                    if isinstance(default_value, dict) and isinstance(config[key], dict):
                        result[key] = apply_defaults(config[key], default_value)
                    else:
                        result[key] = config[key]
                else:
                    result[key] = default_value
            return result

        gpu_config = apply_defaults(gpu_config, defaults)

        # Override device if CUDA is not available
        if gpu_config["device"] == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            gpu_config["device"] = "cpu"

        return gpu_config
    except Exception as e:
        logger.error(f"Error extracting GPU configuration from SMA configuration: {e}")
        # Return default configuration as fallback
        return {
            "device": "cpu",
            "batch_size": 32,
            "precision": "float32",
            "cache_size": 1024,
            "analyzers": {
                "complexity": {"enabled": True},
                "dependency": {"enabled": True},
                "semantic": {"enabled": True},
                "pattern": {"enabled": True}
            },
            "patterns": [],
            "intents": []
        }

def register_config_schema_with_sma(config_registry: Any) -> None:
    """
    Register the GPU Analysis Plugin configuration schema with SMA.

    This function registers the GPU Analysis Plugin configuration schema
    with SMA's configuration registry, allowing it to be validated and
    included in SMA's configuration.

    Args:
        config_registry: SMA configuration registry
    """
    try:
        # Register configuration schema
        if hasattr(config_registry, 'register_schema'):
            config_registry.register_schema("gpu_analysis", GPU_ANALYSIS_CONFIG_SCHEMA)
            logger.info("GPU Analysis Plugin configuration schema registered with SMA")
        else:
            logger.warning("SMA configuration registry does not support schema registration")
    except Exception as e:
        logger.error(f"Error registering configuration schema with SMA: {e}")

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the GPU Analysis Plugin configuration.

    This function validates the GPU Analysis Plugin configuration against
    the schema, returning a tuple of (is_valid, error_messages).

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        errors = []

        # Validate device
        if "device" in config:
            if config["device"] not in ["cuda", "cpu"]:
                errors.append(f"Invalid device: {config['device']}. Must be 'cuda' or 'cpu'.")

        # Validate batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int):
                errors.append(f"Invalid batch_size: {config['batch_size']}. Must be an integer.")
            elif config["batch_size"] < 1 or config["batch_size"] > 1024:
                errors.append(f"Invalid batch_size: {config['batch_size']}. Must be between 1 and 1024.")

        # Validate precision
        if "precision" in config:
            if config["precision"] not in ["float16", "float32", "float64"]:
                errors.append(f"Invalid precision: {config['precision']}. Must be 'float16', 'float32', or 'float64'.")

        # Validate cache_size
        if "cache_size" in config:
            if not isinstance(config["cache_size"], int):
                errors.append(f"Invalid cache_size: {config['cache_size']}. Must be an integer.")
            elif config["cache_size"] < 0 or config["cache_size"] > 10000:
                errors.append(f"Invalid cache_size: {config['cache_size']}. Must be between 0 and 10000.")

        # Validate analyzers
        if "analyzers" in config:
            if not isinstance(config["analyzers"], dict):
                errors.append(f"Invalid analyzers: {config['analyzers']}. Must be an object.")
            else:
                # Validate complexity analyzer
                if "complexity" in config["analyzers"]:
                    complexity = config["analyzers"]["complexity"]
                    if not isinstance(complexity, dict):
                        errors.append(f"Invalid complexity analyzer: {complexity}. Must be an object.")
                    elif "enabled" in complexity and not isinstance(complexity["enabled"], bool):
                        errors.append(f"Invalid complexity.enabled: {complexity['enabled']}. Must be a boolean.")
                    elif "weights" in complexity:
                        weights = complexity["weights"]
                        if not isinstance(weights, dict):
                            errors.append(f"Invalid complexity.weights: {weights}. Must be an object.")
                        else:
                            for weight_name, weight_value in weights.items():
                                if not isinstance(weight_value, (int, float)):
                                    errors.append(f"Invalid complexity.weights.{weight_name}: {weight_value}. Must be a number.")

                # Validate dependency analyzer
                if "dependency" in config["analyzers"]:
                    dependency = config["analyzers"]["dependency"]
                    if not isinstance(dependency, dict):
                        errors.append(f"Invalid dependency analyzer: {dependency}. Must be an object.")
                    elif "enabled" in dependency and not isinstance(dependency["enabled"], bool):
                        errors.append(f"Invalid dependency.enabled: {dependency['enabled']}. Must be a boolean.")
                    elif "max_depth" in dependency:
                        max_depth = dependency["max_depth"]
                        if not isinstance(max_depth, int):
                            errors.append(f"Invalid dependency.max_depth: {max_depth}. Must be an integer.")
                        elif max_depth < 1:
                            errors.append(f"Invalid dependency.max_depth: {max_depth}. Must be at least 1.")

                # Validate semantic analyzer
                if "semantic" in config["analyzers"]:
                    semantic = config["analyzers"]["semantic"]
                    if not isinstance(semantic, dict):
                        errors.append(f"Invalid semantic analyzer: {semantic}. Must be an object.")
                    elif "enabled" in semantic and not isinstance(semantic["enabled"], bool):
                        errors.append(f"Invalid semantic.enabled: {semantic['enabled']}. Must be a boolean.")
                    elif "similarity_threshold" in semantic:
                        threshold = semantic["similarity_threshold"]
                        if not isinstance(threshold, (int, float)):
                            errors.append(f"Invalid semantic.similarity_threshold: {threshold}. Must be a number.")
                        elif threshold < 0 or threshold > 1:
                            errors.append(f"Invalid semantic.similarity_threshold: {threshold}. Must be between 0 and 1.")

                # Validate pattern analyzer
                if "pattern" in config["analyzers"]:
                    pattern = config["analyzers"]["pattern"]
                    if not isinstance(pattern, dict):
                        errors.append(f"Invalid pattern analyzer: {pattern}. Must be an object.")
                    elif "enabled" in pattern and not isinstance(pattern["enabled"], bool):
                        errors.append(f"Invalid pattern.enabled: {pattern['enabled']}. Must be a boolean.")
                    elif "confidence_threshold" in pattern:
                        threshold = pattern["confidence_threshold"]
                        if not isinstance(threshold, (int, float)):
                            errors.append(f"Invalid pattern.confidence_threshold: {threshold}. Must be a number.")
                        elif threshold < 0 or threshold > 1:
                            errors.append(f"Invalid pattern.confidence_threshold: {threshold}. Must be between 0 and 1.")

        # Validate patterns
        if "patterns" in config:
            if not isinstance(config["patterns"], list):
                errors.append(f"Invalid patterns: {config['patterns']}. Must be an array.")
            else:
                for i, pattern in enumerate(config["patterns"]):
                    if not isinstance(pattern, dict):
                        errors.append(f"Invalid pattern at index {i}: {pattern}. Must be an object.")
                    elif "name" not in pattern:
                        errors.append(f"Invalid pattern at index {i}: {pattern}. Missing required property 'name'.")
                    elif "pattern" not in pattern:
                        errors.append(f"Invalid pattern at index {i}: {pattern}. Missing required property 'pattern'.")
                    elif "weight" in pattern and not isinstance(pattern["weight"], (int, float)):
                        errors.append(f"Invalid pattern.weight at index {i}: {pattern['weight']}. Must be a number.")

        # Validate intents
        if "intents" in config:
            if not isinstance(config["intents"], list):
                errors.append(f"Invalid intents: {config['intents']}. Must be an array.")
            else:
                for i, intent in enumerate(config["intents"]):
                    if not isinstance(intent, dict):
                        errors.append(f"Invalid intent at index {i}: {intent}. Must be an object.")
                    elif "name" not in intent:
                        errors.append(f"Invalid intent at index {i}: {intent}. Missing required property 'name'.")
                    elif "patterns" not in intent:
                        errors.append(f"Invalid intent at index {i}: {intent}. Missing required property 'patterns'.")
                    elif not isinstance(intent["patterns"], list):
                        errors.append(f"Invalid intent.patterns at index {i}: {intent['patterns']}. Must be an array.")

        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False, [f"Error validating configuration: {e}"]
