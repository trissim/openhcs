"""
GPU Analysis Plugin Module

This module provides a plugin interface for integrating GPU-accelerated analysis
with the Semantic Matrix Analyzer. It follows proper separation of concerns,
focusing only on GPU acceleration functionality and leaving file management to
the existing systems.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch

from gpu_analysis.ast_adapter import ASTAdapter, GPULanguageParser
from gpu_analysis.ast_tensor import GPUASTTensorizer
from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer
from gpu_analysis.pattern_matcher import GPUPatternMatcherRegistry
from gpu_analysis.dynamic_config import DynamicConfigManager, LearningRateManager
from gpu_analysis.pattern_extraction import PatternExtractor
from gpu_analysis.intent_extraction import IntentExtractor
from gpu_analysis.feedback_processor import FeedbackProcessor
from gpu_analysis.logging_integration import get_logger, set_context
from gpu_analysis.error_handling import handle_error, check_gpu_available, with_error_handling
from gpu_analysis.error_handling import GPUAnalysisError, GPUNotAvailableError, GPUMemoryError

# Set up logging
logger = get_logger("gpu_analysis_plugin")

class GPUAnalysisPlugin:
    """
    Plugin for GPU-accelerated analysis.

    This class provides a plugin interface for integrating GPU-accelerated analysis
    with the Semantic Matrix Analyzer. It follows proper separation of concerns,
    focusing only on GPU acceleration functionality.

    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration dictionary
        semantic_analyzer: GPU-accelerated semantic analyzer
        ast_adapter: Adapter for converting between AST representations
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None, context: Optional[Any] = None):
        """
        Initialize the GPU analysis plugin.

        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration dictionary
            context: SMA plugin context
        """
        try:
            # Set up logging with context if available
            if context:
                set_context(context)
                self.context = context
            else:
                self.context = None

            # Initialize dynamic configuration manager
            self.config_manager = DynamicConfigManager(config)

            # Get configuration from manager
            self.config = self.config_manager.get_config()

            # Set device from configuration or parameter
            self.device = self.config.get("device", device)

            # Override device if CUDA is not available
            if self.device == "cuda":
                try:
                    check_gpu_available()
                except GPUNotAvailableError as e:
                    logger.warning(f"GPU not available: {e}")
                    self.device = "cpu"
                    self.config_manager.update_config({"device": "cpu"}, "system")
                    self.config = self.config_manager.get_config()

            # Initialize learning rate manager
            self.learning_rate_manager = LearningRateManager(self.config_manager)

            # Initialize pattern extractor
            self.pattern_extractor = PatternExtractor(self.config_manager)

            # Initialize intent extractor
            self.intent_extractor = IntentExtractor(self.config_manager, self.pattern_extractor)

            # Initialize feedback processor
            self.feedback_processor = FeedbackProcessor(
                self.config_manager,
                self.pattern_extractor,
                self.intent_extractor,
                self.learning_rate_manager
            )

            # Initialize components
            self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
            self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

            # Log initialization
            logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
            if self.device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception as e:
            # Handle initialization error
            error = handle_error(e, self.context if hasattr(self, 'context') else None)
            logger.error(f"Error initializing GPU Analysis Plugin: {error}")
            raise error

    @with_error_handling
    def analyze_code(self, code: str, file_path: Optional[Union[str, Path]] = None,
                    analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze code using GPU acceleration.

        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        logger.debug(f"Analyzing code from {file_path or 'string'}")
        return self.semantic_analyzer.analyze(code, file_path, analysis_types)

    @with_error_handling
    def analyze_file(self, file_path: Union[str, Path],
                    analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a file using GPU acceleration.

        Args:
            file_path: Path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        logger.debug(f"Analyzing file: {file_path}")
        return self.semantic_analyzer.analyze_file(file_path, analysis_types)

    @with_error_handling
    def analyze_batch(self, file_paths: List[Union[str, Path]],
                     analysis_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple files in batch using GPU acceleration.

        Args:
            file_paths: List of file paths
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary mapping file paths to analysis results
        """
        logger.debug(f"Analyzing batch of {len(file_paths)} files")

        # Read files
        codes = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    codes.append(f.read())
            except Exception as e:
                error = handle_error(e, self.context)
                logger.error(f"Error reading file {file_path}: {error}")
                codes.append("")

        # Analyze in batch
        results = self.semantic_analyzer.batch_analyze(codes, file_paths, analysis_types)

        # Convert to dictionary
        return {str(path): result for path, result in zip(file_paths, results)}

    @with_error_handling
    def tensorize_ast(self, ast_node) -> Dict[str, torch.Tensor]:
        """
        Convert AST to GPU-friendly tensor representation.

        Args:
            ast_node: AST node

        Returns:
            Dictionary of tensors
        """
        logger.debug("Converting AST to GPU-friendly tensor representation")
        return self.ast_adapter.convert_to_gpu_format(ast_node)

    @with_error_handling
    def add_pattern(self, pattern) -> None:
        """
        Add a pattern for matching.

        Args:
            pattern: Pattern to add
        """
        logger.debug(f"Adding pattern: {pattern}")
        self.semantic_analyzer.add_pattern(pattern)

    @with_error_handling
    def clear_patterns(self) -> None:
        """Clear all patterns."""
        logger.debug("Clearing all patterns")
        self.semantic_analyzer.clear_patterns()

    @with_error_handling
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the GPU device.

        Returns:
            Dictionary of device information
        """
        logger.debug("Getting device information")

        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }

        if self.device == "cuda":
            try:
                check_gpu_available()
                info.update({
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "total_memory": torch.cuda.get_device_properties(0).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved()
                })
            except GPUNotAvailableError as e:
                logger.warning(f"GPU not available when getting device info: {e}")
                info["cuda_available"] = False
                info["error"] = str(e)

        return info

    @with_error_handling
    def process_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback and update configuration accordingly.

        This method processes feedback from the user and updates the configuration
        accordingly, enabling adaptive analysis and continuous improvement.

        Args:
            feedback: Feedback dictionary containing:
                - intent_name: Name of the intent
                - is_correct: Whether the analysis was correct
                - corrections: Dictionary of corrections
                - confidence: Confidence in the feedback (0.0-1.0)

        Returns:
            Dictionary of processing results
        """
        logger.debug(f"Processing feedback for intent: {feedback.get('intent_name', 'unknown')}")

        # Process feedback
        result = self.feedback_processor.process_feedback(feedback)

        # Get updated configuration
        self.config = self.config_manager.get_config()

        # Update components with new configuration
        self.semantic_analyzer.update_config(self.config)
        self.ast_adapter.update_config(self.config)

        return {
            "success": True,
            "message": f"Feedback processed successfully for intent: {feedback.get('intent_name', 'unknown')}",
            "config_version": self.config_manager.get_version()
        }

    @with_error_handling
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback history.

        Returns:
            Dictionary of feedback statistics
        """
        logger.debug("Getting feedback statistics")
        return self.feedback_processor.get_feedback_stats()

    @with_error_handling
    def auto_configure(self, project_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Auto-configure the plugin based on project analysis.

        This method analyzes a project and generates an initial configuration
        based on the project's characteristics, which can then be refined through
        human feedback.

        Args:
            project_dir: Path to the project directory

        Returns:
            Dictionary of auto-configuration results
        """
        logger.debug(f"Auto-configuring for project: {project_dir}")

        # Generate auto-configuration
        from gpu_analysis.auto_config import generate_auto_config
        auto_config = generate_auto_config(project_dir, self)

        # Update configuration
        success, errors = self.update_config(auto_config, "auto")

        if not success:
            logger.warning(f"Error applying auto-configuration: {errors}")
            return {
                "success": False,
                "message": f"Error applying auto-configuration: {errors}",
                "config": auto_config
            }

        logger.info("Auto-configuration applied successfully")
        return {
            "success": True,
            "message": "Auto-configuration applied successfully",
            "config": auto_config,
            "config_version": self.config_manager.get_version()
        }

    @with_error_handling
    def save_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Save the current configuration to a file.

        Args:
            file_path: Path to save the configuration to

        Returns:
            Dictionary of save results
        """
        logger.debug(f"Saving configuration to {file_path}")

        # Save configuration
        success = self.config_manager.save_config(file_path)

        if not success:
            logger.warning(f"Error saving configuration to {file_path}")
            return {
                "success": False,
                "message": f"Error saving configuration to {file_path}"
            }

        logger.info(f"Configuration saved to {file_path}")
        return {
            "success": True,
            "message": f"Configuration saved to {file_path}",
            "config_version": self.config_manager.get_version()
        }

    @with_error_handling
    def load_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            file_path: Path to load the configuration from

        Returns:
            Dictionary of load results
        """
        logger.debug(f"Loading configuration from {file_path}")

        # Load configuration
        success = self.config_manager.load_config(file_path)

        if not success:
            logger.warning(f"Error loading configuration from {file_path}")
            return {
                "success": False,
                "message": f"Error loading configuration from {file_path}"
            }

        # Get updated configuration
        self.config = self.config_manager.get_config()

        # Update components with new configuration
        self.semantic_analyzer.update_config(self.config)
        self.ast_adapter.update_config(self.config)

        logger.info(f"Configuration loaded from {file_path}")
        return {
            "success": True,
            "message": f"Configuration loaded from {file_path}",
            "config_version": self.config_manager.get_version()
        }

    @with_error_handling
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dictionary of configuration settings
        """
        logger.debug("Getting current configuration")
        return self.config

    @with_error_handling
    def update_config(self, new_config: Dict[str, Any], source: str = "user") -> Tuple[bool, List[str]]:
        """
        Update the configuration.

        This method updates the configuration with new settings, validating
        the new configuration before applying it.

        Args:
            new_config: New configuration settings
            source: Source of the update (e.g., "user", "system", "feedback")

        Returns:
            Tuple of (success, error_messages)
        """
        logger.debug(f"Updating configuration from source: {source}")

        # Validate new configuration
        from gpu_analysis.config_integration import validate_config
        is_valid, errors = validate_config(new_config)

        if not is_valid:
            logger.warning(f"Invalid configuration: {errors}")
            return False, errors

        # Update device if changed
        if "device" in new_config:
            new_device = new_config["device"]
            if new_device == "cuda":
                try:
                    check_gpu_available()
                    self.device = "cuda"
                except GPUNotAvailableError as e:
                    logger.warning(f"CUDA is not available, falling back to CPU: {e}")
                    new_config["device"] = "cpu"
                    self.device = "cpu"
            else:
                self.device = new_device

        # Update configuration using dynamic configuration manager
        self.config_manager.update_config(new_config, source)

        # Get updated configuration
        self.config = self.config_manager.get_config()

        # Update components
        self.semantic_analyzer.update_config(self.config)
        self.ast_adapter.update_config(self.config)

        logger.info("Configuration updated successfully")
        return True, []

@with_error_handling
def register_plugin(sma_registry):
    """
    Register the GPU analysis plugin with SMA's registry.

    This function registers the GPU analysis plugin with SMA's registry,
    including the configuration schema and language parsers.

    Args:
        sma_registry: SMA registry

    Returns:
        The registered plugin instance
    """
    logger.debug("Registering GPU Analysis Plugin with SMA")

    # Get plugin context if available
    context = None
    if hasattr(sma_registry, 'get_plugin_context'):
        context = sma_registry.get_plugin_context("gpu_analysis")
        set_context(context)
        logger.debug("Using SMA plugin context for logging")

    try:
        # Register configuration schema
        from gpu_analysis.config_integration import register_config_schema_with_sma
        register_config_schema_with_sma(sma_registry.config_registry)
        logger.debug("Configuration schema registered with SMA")

        # Get SMA's configuration
        sma_config = {}
        if hasattr(sma_registry, 'get_config'):
            sma_config = sma_registry.get_config()
            logger.debug("Got SMA configuration")

        # Extract GPU configuration from SMA configuration
        from gpu_analysis.config_integration import get_gpu_config_from_sma
        gpu_config = get_gpu_config_from_sma(sma_config)
        logger.debug("Extracted GPU configuration from SMA configuration")

        # Create plugin with extracted configuration
        plugin = GPUAnalysisPlugin(config=gpu_config, context=context)
        logger.debug("Created GPU Analysis Plugin instance")

        # Register plugin
        sma_registry.register_plugin("gpu_analysis", plugin)
        logger.debug("Registered GPU Analysis Plugin with SMA")

        # Register language parsers
        from gpu_analysis.ast_adapter import register_gpu_parser
        register_gpu_parser(sma_registry.language_registry)
        logger.debug("Registered GPU language parsers with SMA")

        # Log registration
        logger.info("GPU Analysis Plugin registered with SMA")

        return plugin
    except Exception as e:
        error = handle_error(e, context)
        logger.error(f"Error registering GPU Analysis Plugin: {error}")

        # Fall back to default initialization
        logger.warning("Falling back to default initialization")

        try:
            # Create plugin with default configuration
            plugin = GPUAnalysisPlugin(context=context)
            logger.debug("Created GPU Analysis Plugin instance with default configuration")

            # Register plugin
            sma_registry.register_plugin("gpu_analysis", plugin)
            logger.debug("Registered GPU Analysis Plugin with SMA")

            # Register language parsers
            from gpu_analysis.ast_adapter import register_gpu_parser
            register_gpu_parser(sma_registry.language_registry)
            logger.debug("Registered GPU language parsers with SMA")

            return plugin
        except Exception as reg_error:
            error = handle_error(reg_error, context)
            logger.error(f"Error registering plugin: {error}")
            raise error
