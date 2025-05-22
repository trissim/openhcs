"""
GPU Configuration Manager Module

This module provides a configuration manager for GPU-accelerated semantic analysis.
It allows users to control GPU-specific parameters through a unified interface.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch

from gpu_analysis.batch_processor import DynamicBatchSizeManager

# Set up logging
logger = logging.getLogger(__name__)

class GPUConfigManager:
    """
    GPU configuration manager.

    This class manages GPU-specific configuration options and integrates
    with SMA's configuration system.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU configuration manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._set_defaults()

    def _set_defaults(self):
        """Set default values for GPU configuration options."""
        # Device configuration
        self.config.setdefault("gpu.device", "cuda")
        self.config.setdefault("gpu.enabled", True)

        # Memory configuration
        self.config.setdefault("gpu.memory.max_fraction", 0.9)
        self.config.setdefault("gpu.memory.cache_size", 100)
        self.config.setdefault("gpu.memory.keep_in_vram", True)

        # Batch processing configuration
        self.config.setdefault("gpu.batch.enabled", True)
        self.config.setdefault("gpu.batch.initial_size", 10)
        self.config.setdefault("gpu.batch.min_size", 1)
        self.config.setdefault("gpu.batch.max_size", 100)
        self.config.setdefault("gpu.batch.dynamic_sizing", True)

        # Performance configuration
        self.config.setdefault("gpu.performance.dynamic_dispatch", True)
        self.config.setdefault("gpu.performance.cpu_threshold", 1000)  # AST nodes

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from a dictionary.

        Args:
            config_dict: Dictionary of configuration values
        """
        self.config.update(config_dict)

    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory configuration.

        Returns:
            Dictionary of memory configuration
        """
        return {
            "max_fraction": self.get("gpu.memory.max_fraction"),
            "keep_in_vram": self.get("gpu.memory.keep_in_vram")
        }

    def create_batch_size_manager(self) -> DynamicBatchSizeManager:
        """
        Create a dynamic batch size manager based on configuration.

        Returns:
            Dynamic batch size manager
        """
        device = self.get("gpu.device")
        initial_size = self.get("gpu.batch.initial_size")
        min_size = self.get("gpu.batch.min_size")
        max_size = self.get("gpu.batch.max_size")
        target_memory_usage = self.get("gpu.memory.max_fraction") * 0.9

        return DynamicBatchSizeManager(
            device=device,
            initial_batch_size=initial_size,
            min_batch_size=min_size,
            max_batch_size=max_size,
            target_memory_usage=target_memory_usage
        )

    def is_gpu_enabled(self) -> bool:
        """
        Check if GPU acceleration is enabled.

        Returns:
            True if GPU acceleration is enabled, False otherwise
        """
        return self.get("gpu.enabled") and torch.cuda.is_available()

    def is_batch_processing_enabled(self) -> bool:
        """
        Check if batch processing is enabled.

        Returns:
            True if batch processing is enabled, False otherwise
        """
        return self.get("gpu.batch.enabled")

    def is_dynamic_dispatch_enabled(self) -> bool:
        """
        Check if dynamic dispatch is enabled.

        Returns:
            True if dynamic dispatch is enabled, False otherwise
        """
        return self.get("gpu.performance.dynamic_dispatch")

    def get_cpu_threshold(self) -> int:
        """
        Get the CPU threshold for dynamic dispatch.

        Returns:
            CPU threshold
        """
        return self.get("gpu.performance.cpu_threshold")

    def get_device(self) -> str:
        """
        Get the device to use for GPU acceleration.

        Returns:
            Device name
        """
        device = self.get("gpu.device")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return device

class DynamicDispatcher:
    """
    Dynamic dispatcher between CPU and GPU.

    This class dynamically dispatches analysis tasks to either CPU or GPU
    based on input size and configuration.

    Attributes:
        config_manager: GPU configuration manager
        cpu_threshold: Input size threshold for using CPU instead of GPU
        dynamic_dispatch: Whether to dynamically dispatch between CPU and GPU
    """

    def __init__(self, config_manager: GPUConfigManager):
        """
        Initialize the dynamic dispatcher.

        Args:
            config_manager: GPU configuration manager
        """
        self.config_manager = config_manager
        self.cpu_threshold = config_manager.get_cpu_threshold()
        self.dynamic_dispatch = config_manager.is_dynamic_dispatch_enabled()

        # Initialize CPU and GPU analyzers
        self.gpu_analyzer = None
        self.cpu_analyzer = None

    def initialize_analyzers(self):
        """Initialize CPU and GPU analyzers."""
        from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer

        # Initialize GPU analyzer
        self.gpu_analyzer = SemanticAnalyzer(
            device=self.config_manager.get_device(),
            config=self.config_manager.config
        )

        # Initialize CPU analyzer
        self.cpu_analyzer = SemanticAnalyzer(
            device="cpu",
            config=self.config_manager.config
        )

    def dispatch(self, code: str, file_path: Optional[str] = None,
                analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Dispatch analysis to either CPU or GPU based on input size.

        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        # Initialize analyzers if needed
        if self.gpu_analyzer is None or self.cpu_analyzer is None:
            self.initialize_analyzers()

        if not self.dynamic_dispatch:
            # Always use GPU if dynamic dispatch is disabled
            return self.gpu_analyzer.analyze(code, file_path, analysis_types)

        # Estimate input size
        input_size = self._estimate_input_size(code)

        # Dispatch based on input size
        if input_size < self.cpu_threshold:
            # Use CPU for small inputs
            logger.debug(f"Using CPU for input size {input_size} (threshold: {self.cpu_threshold})")
            return self.cpu_analyzer.analyze(code, file_path, analysis_types)
        else:
            # Use GPU for large inputs
            logger.debug(f"Using GPU for input size {input_size} (threshold: {self.cpu_threshold})")
            return self.gpu_analyzer.analyze(code, file_path, analysis_types)

    def _estimate_input_size(self, code: str) -> int:
        """
        Estimate the size of the input code.

        Args:
            code: Python code to analyze

        Returns:
            Estimated input size
        """
        # Simple heuristic: count the number of AST nodes
        import ast
        try:
            tree = ast.parse(code)
            return sum(1 for _ in ast.walk(tree))
        except:
            # If parsing fails, use code length as a fallback
            return len(code)

def get_gpu_config_schema() -> Dict[str, Dict[str, Any]]:
    """
    Get GPU configuration schema for SMA's configuration system.

    Returns:
        Dictionary defining the configuration schema
    """
    return {
        "gpu.device": {
            "type": "string",
            "default": "cuda",
            "description": "GPU device to use for acceleration"
        },
        "gpu.enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable GPU acceleration"
        },
        "gpu.memory.max_fraction": {
            "type": "float",
            "default": 0.9,
            "min": 0.1,
            "max": 1.0,
            "description": "Maximum fraction of GPU memory to use"
        },
        "gpu.memory.cache_size": {
            "type": "integer",
            "default": 100,
            "min": 1,
            "description": "Maximum number of items to cache in GPU memory"
        },
        "gpu.memory.keep_in_vram": {
            "type": "boolean",
            "default": True,
            "description": "Whether to keep data in GPU memory between operations"
        },
        "gpu.batch.enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable batch processing"
        },
        "gpu.batch.initial_size": {
            "type": "integer",
            "default": 10,
            "min": 1,
            "description": "Initial batch size for processing"
        },
        "gpu.batch.min_size": {
            "type": "integer",
            "default": 1,
            "min": 1,
            "description": "Minimum batch size for processing"
        },
        "gpu.batch.max_size": {
            "type": "integer",
            "default": 100,
            "min": 1,
            "description": "Maximum batch size for processing"
        },
        "gpu.batch.dynamic_sizing": {
            "type": "boolean",
            "default": True,
            "description": "Whether to dynamically adjust batch size based on memory usage"
        },
        "gpu.performance.dynamic_dispatch": {
            "type": "boolean",
            "default": True,
            "description": "Whether to dynamically dispatch between CPU and GPU based on input size"
        },
        "gpu.performance.cpu_threshold": {
            "type": "integer",
            "default": 1000,
            "min": 1,
            "description": "Input size threshold for using CPU instead of GPU"
        }
    }

def register_gpu_module(sma_config: Dict[str, Any]) -> Any:
    """
    Register GPU-accelerated module with SMA's plugin system.

    Args:
        sma_config: SMA configuration object

    Returns:
        GPU-accelerated analysis module
    """
    from gpu_analysis.plugin import GPUAnalysisPlugin

    # Create GPU analysis plugin
    gpu_plugin = GPUAnalysisPlugin(
        device=sma_config.get("gpu.device", "cuda"),
        config=sma_config
    )

    # Register with SMA's plugin system
    # Implementation details depend on SMA's plugin architecture

    return gpu_plugin
