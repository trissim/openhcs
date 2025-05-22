# Plan 04a: Configuration Schema and Integration Module

## Objective

Create a configuration integration module for the GPU Analysis Plugin that properly integrates with SMA's configuration system, ensuring consistent configuration management and enabling the implementation of unimplemented SMA methods.

## Background

SMA uses a configuration system to manage settings for the core application and its plugins. The GPU Analysis Plugin needs to integrate with this system to ensure consistent configuration management and to enable the implementation of unimplemented SMA methods that rely on configuration settings.

## Current State

The current GPU Analysis Plugin likely uses a simple dictionary for configuration:

```python
def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
    """
    Initialize the GPU analysis plugin.

    Args:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration dictionary
    """
    self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    self.config = config or {}

    # Initialize components
    self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
    self.ast_adapter = ASTAdapter(device=self.device, config=self.config)
```

SMA's configuration system is more sophisticated, with schema validation, default values, and environment variable overrides.

## Unimplemented SMA Methods

The following SMA methods are currently unimplemented and will be implemented using GPU acceleration, requiring proper configuration integration:

1. `SemanticMatrixBuilder.analyze_component` in the core module:
   ```python
   # TODO: Extract dependencies
   # TODO: Analyze component
   # TODO: Detect patterns
   # TODO: Calculate intent alignments
   ```

2. CLI command handlers in `sma_cli.py`:
   ```python
   def handle_analyze_command(args: argparse.Namespace) -> None:
       """Handle the analyze command."""
       print_header("CODE ANALYSIS")
       print("Analyzing code for intent extraction...")
       # Implementation would go here
       print(color_text("Not yet implemented", "YELLOW"))
   ```

3. Semantic analysis placeholders in `generate_project_snapshot`:
   ```python
   # Placeholder for semantic analysis
   if focus in ["semantics", "all"] and depth >= 2:
       snapshot["semantics"] = {
           "status": "placeholder",
           "message": "Semantic analysis would analyze code patterns, naming conventions, and code quality."
       }
   ```

## Implementation Plan

### 1. Define Configuration Schema

Create a configuration schema for the GPU Analysis Plugin:

```python
# Create a new file: brain/gpu_analysis/config_integration.py

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

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
```

### 2. Create Configuration Integration Module

Create a module to integrate with SMA's configuration system:

```python
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
        import torch
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
```

### 3. Update Plugin Initialization

Update the plugin's `initialize` method to use SMA's configuration system:

```python
def initialize(self, context: PluginContext) -> None:
    """Initialize the plugin with the given context.

    Args:
        context: The plugin context.
    """
    self.context = context

    try:
        # Get SMA's configuration
        sma_config = context.get_config()

        # Integrate GPU configuration with SMA's configuration
        from gpu_analysis.config_integration import get_gpu_config_from_sma
        self.config = get_gpu_config_from_sma(sma_config)

        # Update device based on configuration
        self.device = self.config["device"]

        # Initialize components with updated configuration
        self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
        self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

        # Log initialization
        context.log("info", f"GPU Analysis Plugin initialized with device: {self.device}")
        if self.device == "cuda":
            context.log("info", f"CUDA device: {torch.cuda.get_device_name(0)}")
            context.log("info", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    except Exception as e:
        context.log("error", f"Error initializing GPU Analysis Plugin: {e}")
        # Fall back to default configuration
        self.config = {
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
        self.device = "cpu"

        # Initialize components with default configuration
        self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
        self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

        # Log fallback initialization
        context.log("info", f"GPU Analysis Plugin initialized with fallback configuration")
```

### 4. Add Configuration Validation

Add a method to validate configuration:

```python
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

        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False, [f"Error validating configuration: {e}"]
```

## Implementation Focus

The implementation should focus on:

1. **Schema Definition**: Defining a proper configuration schema for the GPU Analysis Plugin.

2. **Registration Mechanism**: Implementing registration of the schema with SMA.

3. **Configuration Extraction**: Implementing extraction of configuration from SMA's configuration.

4. **Basic Validation**: Implementing essential validation of configuration against the schema.

5. **Default Values**: Implementing fallback to default configuration values when necessary.

## Success Criteria

1. The GPU Analysis Plugin configuration schema is properly defined and registered with SMA.

2. The plugin correctly extracts its configuration from SMA's configuration.

3. The plugin has basic validation of its configuration.

4. The plugin falls back to default configuration values when necessary.

5. The configuration integration enables the implementation of unimplemented SMA methods.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Configuration System: `semantic_matrix_analyzer/semantic_matrix_analyzer/config/__init__.py`

2. GPU Analysis Plugin: `brain/gpu_analysis/plugin.py`

3. SMA Plugin Context: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`
