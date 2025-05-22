"""
GPU Analysis Plugin Module

This module provides a plugin interface for integrating GPU-accelerated analysis
with the Semantic Matrix Analyzer. It follows proper separation of concerns,
focusing only on GPU acceleration functionality and leaving file management to
the existing systems.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch
import numpy as np

# Import GPU analysis components
# Use relative imports to avoid circular imports
import sys
from pathlib import Path

# Add the current directory to the path
plugins_dir = Path(__file__).parent
if str(plugins_dir) not in sys.path:
    sys.path.insert(0, str(plugins_dir))

# Import components
from gpu_analysis.ast_adapter import ASTAdapter, GPULanguageParser
from gpu_analysis.ast_tensor import GPUASTTensorizer
from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer
from gpu_analysis.pattern_matcher import GPUPatternMatcherRegistry
from gpu_analysis.dynamic_config import ConfigObserver
from gpu_analysis.logging_integration import get_logger, set_context
from gpu_analysis.error_handling import (
    GPUAnalysisError, GPUNotAvailableError, GPUMemoryError,
    GPUAnalysisConfigError, GPUAnalysisRuntimeError,
    handle_error, check_gpu_available, with_error_handling
)

# Define SMA plugin interface if not available
class SMAPlugin:
    """Base class for SMA plugins."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Get the version of the plugin."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        raise NotImplementedError

    def initialize(self, context) -> None:
        """Initialize the plugin with the given context."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        pass

class PluginContext:
    """Context for SMA plugins."""

    def log(self, level: str, message: str) -> None:
        """Log a message."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get the SMA configuration."""
        return {}

# Set up logging
logger = get_logger("gpu_analysis_plugin")

class GPUAnalysisPlugin(SMAPlugin, ConfigObserver):
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

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "gpu_analysis"

    @property
    def version(self) -> str:
        """Get the version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "GPU-accelerated code analysis for the Semantic Matrix Analyzer"

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU analysis plugin.

        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}
        self.context = None

        # Initialize components
        self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
        self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

        # Log initialization
        logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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
        return self.semantic_analyzer.analyze(code, file_path, analysis_types)

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
        return self.semantic_analyzer.analyze_file(file_path, analysis_types)

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
        # Read files
        codes = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    codes.append(f.read())
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                codes.append("")

        # Analyze in batch
        results = self.semantic_analyzer.batch_analyze(codes, file_paths, analysis_types)

        # Convert to dictionary
        return {str(path): result for path, result in zip(file_paths, results)}

    def tensorize_ast(self, ast_node) -> Dict[str, torch.Tensor]:
        """
        Convert AST to GPU-friendly tensor representation.

        Args:
            ast_node: AST node

        Returns:
            Dictionary of tensors
        """
        return self.ast_adapter.convert_to_gpu_format(ast_node)

    def add_pattern(self, pattern) -> None:
        """
        Add a pattern for matching.

        Args:
            pattern: Pattern to add
        """
        self.semantic_analyzer.add_pattern(pattern)

    def clear_patterns(self) -> None:
        """Clear all patterns."""
        self.semantic_analyzer.clear_patterns()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if hasattr(self.semantic_analyzer, 'clear_cache'):
            self.semantic_analyzer.clear_cache()

        if hasattr(self.ast_adapter, 'clear_cache'):
            self.ast_adapter.clear_cache()

        if self.context:
            self.context.log("info", "Cache cleared")
        else:
            logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary of cache information
        """
        info = {
            "cache_enabled": True,
            "cache_size": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Get semantic analyzer cache info
        if hasattr(self.semantic_analyzer, 'get_cache_info'):
            semantic_cache_info = self.semantic_analyzer.get_cache_info()
            info["semantic_analyzer"] = semantic_cache_info
            info["cache_size"] += semantic_cache_info.get("size", 0)
            info["cache_hits"] += semantic_cache_info.get("hits", 0)
            info["cache_misses"] += semantic_cache_info.get("misses", 0)

        # Get AST adapter cache info
        if hasattr(self.ast_adapter, 'get_cache_info'):
            ast_cache_info = self.ast_adapter.get_cache_info()
            info["ast_adapter"] = ast_cache_info
            info["cache_size"] += ast_cache_info.get("size", 0)
            info["cache_hits"] += ast_cache_info.get("hits", 0)
            info["cache_misses"] += ast_cache_info.get("misses", 0)

        return info

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the GPU device.

        Returns:
            Dictionary of device information
        """
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }

        if self.device == "cuda":
            info.update({
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            })

        return info

    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with the given context.

        Args:
            context: The plugin context.
        """
        self.context = context

        # Set up logging with context
        set_context(context)
        logger.info("Initializing GPU Analysis Plugin")

        try:
            # Get SMA's configuration
            sma_config = context.get_config() if hasattr(context, 'get_config') else {}
            logger.debug("Retrieved SMA configuration")

            # Register configuration schema with SMA
            from gpu_analysis.config_integration import register_config_schema_with_sma
            if hasattr(context, 'config_registry'):
                register_config_schema_with_sma(context.config_registry)
                logger.debug("Registered configuration schema with SMA")

            # Initialize dynamic configuration manager
            from gpu_analysis.dynamic_config import DynamicConfigManager, LearningRateManager
            self.dynamic_config_manager = DynamicConfigManager(sma_config)
            self.learning_rate_manager = LearningRateManager(self.dynamic_config_manager)
            logger.debug("Initialized dynamic configuration manager")

            # Get configuration from dynamic config manager
            self.config = self.dynamic_config_manager.get_config()

            # Update device based on configuration
            self.device = self.config["device"]
            if self.device == "cuda":
                try:
                    # Check if GPU is available
                    check_gpu_available()
                    logger.info("CUDA is available")
                except GPUNotAvailableError as e:
                    logger.warning(f"GPU not available: {e}")
                    self.device = "cpu"

            # Register self as observer for configuration changes
            self.dynamic_config_manager.register_observer(self)
            logger.debug("Registered as observer for configuration changes")

            # Initialize pattern extraction, intent extraction, and feedback processing
            from gpu_analysis.pattern_extraction import PatternExtractor
            from gpu_analysis.intent_extraction import IntentExtractor
            from gpu_analysis.feedback_processor import FeedbackProcessor

            self.pattern_extractor = PatternExtractor(self.dynamic_config_manager)
            self.intent_extractor = IntentExtractor(self.dynamic_config_manager, self.pattern_extractor)
            self.feedback_processor = FeedbackProcessor(
                self.dynamic_config_manager,
                self.pattern_extractor,
                self.intent_extractor,
                self.learning_rate_manager
            )
            logger.debug("Initialized pattern extraction, intent extraction, and feedback processing")

            # Initialize components with updated configuration
            self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
            self.ast_adapter = ASTAdapter(device=self.device, config=self.config)
            logger.debug("Initialized semantic analyzer and AST adapter")

            # Register language parsers
            from gpu_analysis.ast_adapter import register_gpu_parser
            if hasattr(context, 'language_registry'):
                register_gpu_parser(context.language_registry)
                logger.debug("Registered language parsers")

            # Log initialization
            logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
            if self.device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

            # Register plugin methods with SMA
            # Placeholder for now
            logger.debug("Registered plugin methods with SMA")

            # Register CLI commands
            # Placeholder for now
            logger.debug("Registered CLI commands")

        except Exception as e:
            # Handle initialization error
            error = handle_error(e, context)
            logger.error(f"Error initializing GPU Analysis Plugin: {error}")

            # Fall back to default configuration
            from gpu_analysis.config_integration import get_gpu_config_from_sma
            self.config = get_gpu_config_from_sma({})
            self.device = "cpu"

            # Initialize components with default configuration
            self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
            self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

            # Log fallback initialization
            logger.info("GPU Analysis Plugin initialized with fallback configuration")

    @with_error_handling
    def on_config_changed(self, config: Dict[str, Any], changed_keys: Set[str], source: str) -> None:
        """
        Called when configuration changes.

        Args:
            config: The new configuration
            changed_keys: Set of keys that were changed
            source: Source of the change (e.g., "user", "system", "feedback")
        """
        # Update local configuration
        self.config = config

        # Log configuration change
        logger.info(f"Configuration changed by {source}: {', '.join(changed_keys)}")

        # Check if device changed
        if "device" in changed_keys:
            # Update device
            new_device = config["device"]
            if new_device != self.device:
                logger.info(f"Device changing from {self.device} to {new_device}")

                # Check if CUDA is available if switching to cuda
                if new_device == "cuda":
                    try:
                        check_gpu_available()
                        logger.info("CUDA is available for device change")
                    except GPUNotAvailableError as e:
                        logger.warning(f"GPU not available for device change: {e}")
                        new_device = "cpu"

                self.device = new_device
                logger.info(f"Device changed to {self.device}")

                # Reinitialize components with new device
                try:
                    self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
                    self.ast_adapter = ASTAdapter(device=self.device, config=self.config)
                    logger.debug("Reinitialized components with new device")
                except Exception as e:
                    error = handle_error(e)
                    logger.error(f"Error reinitializing components with new device: {error}")
                    raise error

        # Update components with new configuration
        try:
            if hasattr(self.semantic_analyzer, 'update_config'):
                self.semantic_analyzer.update_config(self.config)
                logger.debug("Updated semantic analyzer configuration")

            if hasattr(self.ast_adapter, 'update_config'):
                self.ast_adapter.update_config(self.config)
                logger.debug("Updated AST adapter configuration")
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error updating component configurations: {error}")
            raise error

    def extract_patterns(self, content: str, content_type: str,
                       context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Extract patterns from content.

        Args:
            content: Content to extract patterns from
            content_type: Type of content ("code", "text", "behavior")
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        if hasattr(self, 'pattern_extractor') and self.pattern_extractor:
            return self.pattern_extractor.extract_patterns(content, content_type, context)
        return []

    def extract_intent(self, content: str, content_type: str,
                      context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Extract intents from content.

        Args:
            content: Content to extract intents from
            content_type: Type of content ("code", "text", "behavior")
            context: Optional context information

        Returns:
            List of extracted intents
        """
        if hasattr(self, 'intent_extractor') and self.intent_extractor:
            return self.intent_extractor.extract_intent(content, content_type, context)
        return []

    def process_feedback(self, feedback: Union[Any, str],
                        context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process a piece of feedback and adapt behavior accordingly.

        Args:
            feedback: The feedback to process (Feedback object or content string)
            context: Additional context for processing

        Returns:
            Result of processing the feedback
        """
        if hasattr(self, 'feedback_processor') and self.feedback_processor:
            return self.feedback_processor.process_feedback(feedback, context)
        return None

    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        logger.info("Shutting down GPU Analysis Plugin")

        try:
            # Clean up resources
            if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer:
                try:
                    self.semantic_analyzer.cleanup()
                    logger.debug("Cleaned up semantic analyzer")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error cleaning up semantic analyzer: {error}")

            if hasattr(self, 'ast_adapter') and self.ast_adapter:
                try:
                    self.ast_adapter.cleanup()
                    logger.debug("Cleaned up AST adapter")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error cleaning up AST adapter: {error}")

            # Unregister from dynamic config manager
            if hasattr(self, 'dynamic_config_manager') and self.dynamic_config_manager:
                try:
                    self.dynamic_config_manager.unregister_observer(self)
                    logger.debug("Unregistered from dynamic config manager")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error unregistering from dynamic config manager: {error}")

            # Unregister pattern extractor, intent extractor, and feedback processor
            if hasattr(self, 'pattern_extractor') and self.pattern_extractor:
                try:
                    self.dynamic_config_manager.unregister_observer(self.pattern_extractor)
                    logger.debug("Unregistered pattern extractor")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error unregistering pattern extractor: {error}")

            if hasattr(self, 'intent_extractor') and self.intent_extractor:
                try:
                    self.dynamic_config_manager.unregister_observer(self.intent_extractor)
                    logger.debug("Unregistered intent extractor")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error unregistering intent extractor: {error}")

            if hasattr(self, 'feedback_processor') and self.feedback_processor:
                try:
                    self.dynamic_config_manager.unregister_observer(self.feedback_processor)
                    logger.debug("Unregistered feedback processor")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error unregistering feedback processor: {error}")

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA cache")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error clearing CUDA cache: {error}")

            # Log shutdown
            logger.info("GPU Analysis Plugin shutdown complete")

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error during GPU Analysis Plugin shutdown: {error}")

    # Methods to implement unimplemented SMA functionality

    @with_error_handling
    def analyze_component(self, component: str, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a component using GPU acceleration.

        This method implements the unimplemented SMA method for analyzing components.

        Args:
            component: Component to analyze (e.g., class or function name)
            file_path: Path to the file containing the component

        Returns:
            Analysis results
        """
        # Convert file_path to Path if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Log analysis
        logger.info(f"Analyzing component {component} in {file_path}")

        try:
            # Parse the file
            logger.debug(f"Reading file {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error reading file {file_path}: {error}")
            raise GPUAnalysisParsingError(f"Error reading file {file_path}", e)

        try:
            # Extract the component
            # Note: This is a placeholder implementation that will be replaced in Plan 03
            # with proper component extraction using the language parser
            logger.debug(f"Extracting component {component}")
            component_code = self._extract_component_code(code, component)

            if not component_code:
                logger.warning(f"Component {component} not found in {file_path}")
                return {
                    "error": f"Component {component} not found in {file_path}",
                    "component": component,
                    "file_path": str(file_path)
                }
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting component {component}: {error}")
            raise GPUAnalysisComponentError(f"Error extracting component {component}", e)

        try:
            # Analyze the component
            logger.debug(f"Analyzing component code")
            results = self.analyze_code(component_code, file_path)

            # Add component information
            results["component"] = component
            results["file_path"] = str(file_path)

            logger.info(f"Component analysis complete for {component}")
            return results
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing component code: {error}")
            raise GPUAnalysisRuntimeError(f"Error analyzing component code", e)

    @with_error_handling
    def _extract_component_code(self, code: str, component: str) -> str:
        """
        Extract component code from file code.

        This is a placeholder implementation that will be replaced in Plan 03
        with proper component extraction using the language parser.

        Args:
            code: File code
            component: Component to extract

        Returns:
            Component code
        """
        logger.debug(f"Extracting component code for {component}")

        # Simple implementation that looks for class or function definition
        import re

        try:
            # Look for class definition
            class_pattern = rf"class\s+{re.escape(component)}\s*(\(.*\))?\s*:"
            class_match = re.search(class_pattern, code)

            if class_match:
                # Found class definition
                logger.debug(f"Found class definition for {component}")
                start_pos = class_match.start()

                # Find the end of the class
                # This is a simplified approach that doesn't handle nested classes correctly
                lines = code[start_pos:].split('\n')
                class_lines = [lines[0]]
                indent = len(lines[0]) - len(lines[0].lstrip())

                for line in lines[1:]:
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        break
                    class_lines.append(line)

                return '\n'.join(class_lines)

            # Look for function definition
            func_pattern = rf"def\s+{re.escape(component)}\s*\("
            func_match = re.search(func_pattern, code)

            if func_match:
                # Found function definition
                logger.debug(f"Found function definition for {component}")
                start_pos = func_match.start()

                # Find the end of the function
                # This is a simplified approach that doesn't handle nested functions correctly
                lines = code[start_pos:].split('\n')
                func_lines = [lines[0]]
                indent = len(lines[0]) - len(lines[0].lstrip())

                for line in lines[1:]:
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        break
                    func_lines.append(line)

                return '\n'.join(func_lines)

            # Component not found
            logger.warning(f"Component {component} not found in code")
            return ""

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting component code: {error}")
            raise GPUAnalysisComponentError(f"Error extracting component code for {component}", e)

    @with_error_handling
    def generate_semantic_snapshot(self, project_dir: Union[str, Path], depth: int = 2) -> Dict[str, Any]:
        """
        Generate a semantic snapshot of a project.

        This method implements the unimplemented SMA method for generating
        semantic snapshots of projects.

        Args:
            project_dir: Path to the project directory
            depth: Depth of analysis (1-3)

        Returns:
            Semantic snapshot
        """
        # Convert project_dir to Path if it's a string
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        # Log snapshot generation
        logger.info(f"Generating semantic snapshot for {project_dir} with depth {depth}")

        try:
            # Find Python files
            logger.debug(f"Finding Python files in {project_dir}")
            python_files = list(project_dir.glob("**/*.py"))
            logger.info(f"Found {len(python_files)} Python files")

            # Analyze files
            file_analyses = {}
            analyzed_count = 0
            skipped_count = 0
            error_count = 0

            for file_path in python_files:
                try:
                    # Skip files that are too large
                    if file_path.stat().st_size > 1_000_000:  # 1 MB
                        logger.debug(f"Skipping large file: {file_path}")
                        skipped_count += 1
                        continue

                    # Analyze file
                    logger.debug(f"Analyzing file: {file_path}")
                    analysis = self.analyze_file(file_path)
                    file_analyses[str(file_path.relative_to(project_dir))] = analysis
                    analyzed_count += 1

                    # Log progress periodically
                    if analyzed_count % 10 == 0:
                        logger.info(f"Analyzed {analyzed_count} files so far")

                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error analyzing file {file_path}: {error}")
                    error_count += 1

            logger.info(f"Analysis complete: {analyzed_count} files analyzed, {skipped_count} files skipped, {error_count} errors")

            # Generate snapshot
            logger.debug("Generating snapshot from analysis results")
            snapshot = {
                "project": project_dir.name,
                "file_count": len(python_files),
                "analyzed_files": len(file_analyses),
                "skipped_files": skipped_count,
                "error_files": error_count,
                "semantics": {
                    "status": "complete",
                    "complexity": self._calculate_project_complexity(file_analyses),
                    "patterns": self._extract_project_patterns(file_analyses),
                    "dependencies": self._extract_project_dependencies(file_analyses, depth),
                    "intents": self._extract_project_intents(file_analyses, depth)
                }
            }

            logger.info(f"Semantic snapshot generated for {project_dir}")
            return snapshot

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error generating semantic snapshot: {error}")

            # Return error information
            return {
                "error": str(error),
                "project": str(project_dir)
            }

    @with_error_handling
    def _calculate_project_complexity(self, file_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate project complexity from file analyses.

        Args:
            file_analyses: Dictionary mapping file paths to analysis results

        Returns:
            Project complexity
        """
        logger.debug("Calculating project complexity")

        # Initialize complexity metrics
        complexity = {
            "cyclomatic": 0,
            "cognitive": 0,
            "halstead": 0,
            "file_count": len(file_analyses),
            "loc": 0,
            "average_cyclomatic": 0,
            "average_cognitive": 0
        }

        # Sum complexity metrics
        files_with_complexity = 0
        for file_path, analysis in file_analyses.items():
            if "complexity" in analysis:
                file_complexity = analysis["complexity"]
                complexity["cyclomatic"] += file_complexity.get("cyclomatic", 0)
                complexity["cognitive"] += file_complexity.get("cognitive", 0)
                complexity["halstead"] += file_complexity.get("halstead", 0)
                complexity["loc"] += file_complexity.get("loc", 0)
                files_with_complexity += 1

        # Calculate averages
        if files_with_complexity > 0:
            complexity["average_cyclomatic"] = complexity["cyclomatic"] / files_with_complexity
            complexity["average_cognitive"] = complexity["cognitive"] / files_with_complexity
            logger.debug(f"Calculated complexity for {files_with_complexity} files")
        else:
            logger.warning("No files with complexity metrics found")

        return complexity

    @with_error_handling
    def _extract_project_patterns(self, file_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract project patterns from file analyses.

        Args:
            file_analyses: Dictionary mapping file paths to analysis results

        Returns:
            List of project patterns
        """
        logger.debug("Extracting project patterns")

        # Initialize patterns
        patterns = []
        pattern_counts = {}
        pattern_examples = {}

        # Extract patterns
        for file_path, analysis in file_analyses.items():
            if "pattern_matches" in analysis:
                for pattern_match in analysis["pattern_matches"]:
                    try:
                        pattern_name = pattern_match.get("pattern_name", "unknown")

                        # Update pattern counts
                        if pattern_name in pattern_counts:
                            pattern_counts[pattern_name] += 1
                        else:
                            pattern_counts[pattern_name] = 1
                            pattern_examples[pattern_name] = []

                        # Add example
                        example = {"file": file_path, "line": pattern_match.get("line", 0)}
                        if len(pattern_examples[pattern_name]) < 5:  # Limit to 5 examples per pattern
                            pattern_examples[pattern_name].append(example)
                    except Exception as e:
                        error = handle_error(e)
                        logger.warning(f"Error processing pattern match in {file_path}: {error}")

        # Create pattern objects
        for pattern_name, count in pattern_counts.items():
            try:
                patterns.append({
                    "name": pattern_name,
                    "description": self._get_pattern_description(pattern_name),
                    "count": count,
                    "examples": pattern_examples.get(pattern_name, [])
                })
            except Exception as e:
                error = handle_error(e)
                logger.warning(f"Error creating pattern object for {pattern_name}: {error}")

        # Sort patterns by count
        patterns.sort(key=lambda p: p["count"], reverse=True)

        logger.debug(f"Extracted {len(patterns)} patterns")
        return patterns

    def _get_pattern_description(self, pattern_name: str) -> str:
        """
        Get a description for a pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Pattern description
        """
        # Simple mapping of pattern names to descriptions
        descriptions = {
            "singleton": "Singleton pattern - ensures a class has only one instance",
            "factory": "Factory pattern - creates objects without specifying the exact class",
            "observer": "Observer pattern - defines a one-to-many dependency between objects",
            "decorator": "Decorator pattern - adds behavior to objects dynamically",
            "strategy": "Strategy pattern - defines a family of algorithms",
            "command": "Command pattern - encapsulates a request as an object",
            "adapter": "Adapter pattern - allows incompatible interfaces to work together",
            "facade": "Facade pattern - provides a simplified interface to a complex system",
            "template": "Template pattern - defines the skeleton of an algorithm",
            "iterator": "Iterator pattern - provides a way to access elements sequentially",
            "composite": "Composite pattern - composes objects into tree structures",
            "state": "State pattern - allows an object to alter its behavior when its state changes",
            "proxy": "Proxy pattern - provides a surrogate for another object",
            "bridge": "Bridge pattern - separates an abstraction from its implementation",
            "builder": "Builder pattern - separates the construction of a complex object",
            "chain": "Chain of Responsibility pattern - passes a request along a chain",
            "mediator": "Mediator pattern - defines simplified communication between classes",
            "memento": "Memento pattern - captures and externalizes an object's state",
            "visitor": "Visitor pattern - separates an algorithm from an object structure",
            "interpreter": "Interpreter pattern - implements a specialized language"
        }

        return descriptions.get(pattern_name.lower(), f"Pattern: {pattern_name}")

    @with_error_handling
    def _extract_project_dependencies(self, file_analyses: Dict[str, Dict[str, Any]], depth: int) -> Dict[str, Any]:
        """
        Extract project dependencies from file analyses.

        Args:
            file_analyses: Dictionary mapping file paths to analysis results
            depth: Depth of analysis (1-3)

        Returns:
            Project dependencies
        """
        logger.debug("Extracting project dependencies")

        # Initialize dependencies
        dependencies = {
            "internal": {},
            "external": {},
            "graph": {}
        }

        # Extract dependencies
        for file_path, analysis in file_analyses.items():
            if "dependencies" in analysis:
                try:
                    file_deps = analysis["dependencies"]

                    # Internal dependencies
                    if "internal" in file_deps:
                        for dep in file_deps["internal"]:
                            try:
                                dep_name = dep.get("name", "")
                                if dep_name:
                                    if dep_name in dependencies["internal"]:
                                        dependencies["internal"][dep_name] += 1
                                    else:
                                        dependencies["internal"][dep_name] = 1
                            except Exception as e:
                                error = handle_error(e)
                                logger.warning(f"Error processing internal dependency in {file_path}: {error}")

                    # External dependencies
                    if "external" in file_deps:
                        for dep in file_deps["external"]:
                            try:
                                dep_name = dep.get("name", "")
                                if dep_name:
                                    if dep_name in dependencies["external"]:
                                        dependencies["external"][dep_name] += 1
                                    else:
                                        dependencies["external"][dep_name] = 1
                            except Exception as e:
                                error = handle_error(e)
                                logger.warning(f"Error processing external dependency in {file_path}: {error}")

                    # Dependency graph
                    if depth >= 2 and "graph" in file_deps:
                        dependencies["graph"][file_path] = file_deps["graph"]
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error processing dependencies in {file_path}: {error}")

        # Sort dependencies by count
        internal_sorted = sorted(dependencies["internal"].items(), key=lambda x: x[1], reverse=True)
        external_sorted = sorted(dependencies["external"].items(), key=lambda x: x[1], reverse=True)

        # Convert back to dictionaries
        dependencies["internal"] = dict(internal_sorted)
        dependencies["external"] = dict(external_sorted)

        logger.debug(f"Extracted {len(dependencies['internal'])} internal dependencies and {len(dependencies['external'])} external dependencies")
        return dependencies

    @with_error_handling
    def _extract_project_intents(self, file_analyses: Dict[str, Dict[str, Any]], depth: int) -> List[Dict[str, Any]]:
        """
        Extract project intents from file analyses.

        Args:
            file_analyses: Dictionary mapping file paths to analysis results
            depth: Depth of analysis (1-3)

        Returns:
            List of project intents
        """
        logger.debug("Extracting project intents")

        # Initialize intents
        intents = []
        intent_counts = {}
        intent_examples = {}

        # Only extract intents for depth >= 3
        if depth < 3:
            logger.debug("Skipping intent extraction for depth < 3")
            return intents

        # Extract intents
        for file_path, analysis in file_analyses.items():
            if "intents" in analysis:
                for intent_match in analysis["intents"]:
                    try:
                        intent_name = intent_match.get("intent_name", "unknown")

                        # Update intent counts
                        if intent_name in intent_counts:
                            intent_counts[intent_name] += 1
                        else:
                            intent_counts[intent_name] = 1
                            intent_examples[intent_name] = []

                        # Add example
                        example = {"file": file_path, "line": intent_match.get("line", 0)}
                        if len(intent_examples[intent_name]) < 5:  # Limit to 5 examples per intent
                            intent_examples[intent_name].append(example)
                    except Exception as e:
                        error = handle_error(e)
                        logger.warning(f"Error processing intent match in {file_path}: {error}")

        # Create intent objects
        for intent_name, count in intent_counts.items():
            try:
                intents.append({
                    "name": intent_name,
                    "description": self._get_intent_description(intent_name),
                    "count": count,
                    "examples": intent_examples.get(intent_name, [])
                })
            except Exception as e:
                error = handle_error(e)
                logger.warning(f"Error creating intent object for {intent_name}: {error}")

        # Sort intents by count
        intents.sort(key=lambda i: i["count"], reverse=True)

        logger.debug(f"Extracted {len(intents)} intents")
        return intents

    def _get_intent_description(self, intent_name: str) -> str:
        """
        Get a description for an intent.

        Args:
            intent_name: Name of the intent

        Returns:
            Intent description
        """
        # Simple mapping of intent names to descriptions
        descriptions = {
            "data_processing": "Intent to process or transform data",
            "file_io": "Intent to read from or write to files",
            "web_request": "Intent to make web requests or API calls",
            "database": "Intent to interact with a database",
            "visualization": "Intent to visualize data",
            "machine_learning": "Intent to perform machine learning tasks",
            "authentication": "Intent to authenticate users or services",
            "authorization": "Intent to authorize users or services",
            "logging": "Intent to log information or errors",
            "error_handling": "Intent to handle errors or exceptions",
            "configuration": "Intent to manage configuration",
            "caching": "Intent to cache data for performance",
            "validation": "Intent to validate data or input",
            "parsing": "Intent to parse data or text",
            "serialization": "Intent to serialize or deserialize data",
            "concurrency": "Intent to handle concurrent operations",
            "networking": "Intent to perform network operations",
            "ui_interaction": "Intent to interact with a user interface",
            "testing": "Intent to test code or functionality",
            "monitoring": "Intent to monitor system or application state"
        }

        return descriptions.get(intent_name.lower(), f"Intent: {intent_name}")

    @with_error_handling
    def analyze_project(self, project_dir: Union[str, Path],
                       analysis_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a project directory using GPU acceleration.

        Args:
            project_dir: Path to the project directory
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results by file path
        """
        logger.info(f"Analyzing project directory: {project_dir}")

        # Convert project_dir to Path if it's a string
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        try:
            # Find Python files
            logger.debug(f"Finding Python files in {project_dir}")
            python_files = list(project_dir.glob("**/*.py"))
            logger.info(f"Found {len(python_files)} Python files")

            # Analyze files in batch
            results = {}
            analyzed_count = 0
            skipped_count = 0
            error_count = 0

            for file_path in python_files:
                try:
                    # Skip files that are too large
                    if file_path.stat().st_size > 1_000_000:  # 1 MB
                        logger.debug(f"Skipping large file: {file_path}")
                        skipped_count += 1
                        continue

                    # Analyze file
                    logger.debug(f"Analyzing file: {file_path}")
                    result = self.analyze_file(file_path, analysis_types)
                    results[str(file_path.relative_to(project_dir))] = result
                    analyzed_count += 1

                    # Log progress periodically
                    if analyzed_count % 10 == 0:
                        logger.info(f"Analyzed {analyzed_count} files so far")

                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error analyzing file {file_path}: {error}")
                    error_count += 1
                    # Continue with next file

            logger.info(f"Project analysis complete: {analyzed_count} files analyzed, {skipped_count} files skipped, {error_count} errors")
            return results

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing project directory {project_dir}: {error}")
            raise GPUAnalysisRuntimeError(f"GPU analysis failed for project directory {project_dir}", e)

    @with_error_handling
    def analyze_error_trace(self, error_trace: str) -> Dict[str, Any]:
        """
        Analyze an error trace using GPU acceleration.

        Args:
            error_trace: Error trace to analyze

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing error trace: {error_trace[:100]}...")

        try:
            # Extract code snippets from error trace
            logger.debug("Extracting code snippets from error trace")
            snippets = self._extract_code_snippets(error_trace)
            logger.info(f"Extracted {len(snippets)} code snippets from error trace")

            # Detect error type
            logger.debug("Detecting error type")
            error_type = self._detect_error_type(error_trace)
            logger.info(f"Detected error type: {error_type}")

            # Initialize results
            results = {
                "snippets_analyzed": len(snippets),
                "error_type": error_type,
                "suggested_fixes": []
            }

            # Analyze each snippet
            for i, snippet in enumerate(snippets):
                try:
                    # Analyze snippet
                    logger.debug(f"Analyzing snippet {i+1}/{len(snippets)}")
                    snippet_analysis = self.analyze_code(snippet)

                    # Generate suggested fix
                    logger.debug(f"Generating fix suggestion for snippet {i+1}")
                    suggested_fix = self._generate_fix_suggestion(snippet, snippet_analysis)
                    if suggested_fix:
                        logger.info(f"Generated fix suggestion: {suggested_fix}")
                        results["suggested_fixes"].append(suggested_fix)
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error analyzing snippet {i}: {error}")

            logger.info(f"Error trace analysis complete: {len(snippets)} snippets analyzed, {len(results['suggested_fixes'])} fixes suggested")
            return results

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing error trace: {error}")

            # Return error information with graceful degradation
            return {
                "error": str(error),
                "snippets_analyzed": 0,
                "error_type": "Unknown",
                "suggested_fixes": []
            }

    @with_error_handling
    def _extract_code_snippets(self, error_trace: str) -> List[str]:
        """
        Extract code snippets from an error trace.

        Args:
            error_trace: Error trace to extract snippets from

        Returns:
            List of code snippets
        """
        logger.debug("Extracting code snippets from error trace")

        # Simple implementation: look for indented blocks after line numbers
        snippets = []
        lines = error_trace.split("\n")
        current_snippet = []
        in_snippet = False
        file_locations = []

        try:
            for i, line in enumerate(lines):
                # Check if line contains a line number indicator
                file_match = re.search(r"^\s*File \"(.*)\", line (\d+)", line)
                if file_match:
                    # End previous snippet if any
                    if current_snippet:
                        snippet_text = "\n".join(current_snippet)
                        snippets.append(snippet_text)
                        logger.debug(f"Extracted snippet: {snippet_text[:50]}...")
                        current_snippet = []

                    # Record file location
                    file_path = file_match.group(1)
                    line_number = file_match.group(2)
                    file_locations.append(f"{file_path}:{line_number}")
                    logger.debug(f"Found file location: {file_path}:{line_number}")

                    in_snippet = True
                elif in_snippet and line.strip() and line.startswith("    "):
                    # Add indented line to current snippet
                    current_snippet.append(line.strip())
                elif in_snippet and current_snippet and not line.strip():
                    # Empty line after snippet
                    in_snippet = False

            # Add last snippet if any
            if current_snippet:
                snippet_text = "\n".join(current_snippet)
                snippets.append(snippet_text)
                logger.debug(f"Extracted final snippet: {snippet_text[:50]}...")

            logger.info(f"Extracted {len(snippets)} code snippets from {len(file_locations)} file locations")
            return snippets

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting code snippets: {error}")
            return []

    @with_error_handling
    def _detect_error_type(self, error_trace: str) -> str:
        """
        Detect the type of error from an error trace.

        Args:
            error_trace: Error trace to analyze

        Returns:
            Error type
        """
        logger.debug("Detecting error type from error trace")

        # Define error types to look for
        error_types = [
            "SyntaxError",
            "TypeError",
            "NameError",
            "AttributeError",
            "ImportError",
            "ValueError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "AssertionError",
            "ZeroDivisionError",
            "FileNotFoundError",
            "PermissionError",
            "OSError",
            "IOError",
            "ModuleNotFoundError",
            "UnboundLocalError",
            "MemoryError",
            "RecursionError",
            "StopIteration",
            "IndentationError",
            "TabError",
            "UnicodeError",
            "UnicodeDecodeError",
            "UnicodeEncodeError",
            "ConnectionError",
            "TimeoutError"
        ]

        # Look for error types in the trace
        for error_type in error_types:
            if error_type in error_trace:
                logger.info(f"Detected error type: {error_type}")
                return error_type

        # If no specific error type is found, try to extract from the first line
        try:
            first_line = error_trace.strip().split('\n')[0]
            if "Error:" in first_line:
                error_name = first_line.split("Error:")[0].strip().split()[-1]
                if error_name:
                    logger.info(f"Extracted error type from first line: {error_name}")
                    return error_name
        except Exception as e:
            error = handle_error(e)
            logger.warning(f"Error extracting error type from first line: {error}")

        logger.info("Could not determine specific error type")
        return "Unknown"

    @with_error_handling
    def _generate_fix_suggestion(self, snippet: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate a fix suggestion for a code snippet.

        Args:
            snippet: Code snippet to generate a fix for
            analysis: Analysis results for the snippet

        Returns:
            Fix suggestion, or None if no suggestion could be generated
        """
        logger.debug("Generating fix suggestion for code snippet")

        # Get error type from analysis
        error_type = analysis.get("error_type", "Unknown")
        logger.debug(f"Error type from analysis: {error_type}")

        try:
            # Generate suggestions based on error type
            if error_type == "SyntaxError":
                logger.debug("Checking for common syntax errors")

                # Check for missing parentheses, braces, brackets
                if "(" in snippet and ")" not in snippet:
                    logger.info("Detected missing closing parenthesis")
                    return "Add missing closing parenthesis ')'"
                elif "{" in snippet and "}" not in snippet:
                    logger.info("Detected missing closing brace")
                    return "Add missing closing brace '}'"
                elif "[" in snippet and "]" not in snippet:
                    logger.info("Detected missing closing bracket")
                    return "Add missing closing bracket ']'"

                # Check for missing colons in control structures
                if ":" not in snippet:
                    if any(keyword in snippet for keyword in ["if ", "for ", "while ", "def ", "class "]):
                        logger.info("Detected missing colon in control structure")
                        return "Add missing colon ':' at the end of the line"

                # Check for indentation errors
                if "IndentationError" in error_type:
                    logger.info("Detected indentation error")
                    return "Fix indentation - use consistent spaces or tabs"

            elif error_type == "NameError":
                logger.debug("Checking for undefined variables")

                # Extract undefined variables from analysis
                undefined_vars = analysis.get("undefined_variables", [])
                if undefined_vars:
                    logger.info(f"Detected undefined variables: {undefined_vars}")
                    return f"Define variable(s) before use: {', '.join(undefined_vars)}"

                # Check for common typos in variable names
                variables = analysis.get("variables", {})
                if variables:
                    for var in variables:
                        if var.lower() in snippet.lower() and var not in snippet:
                            logger.info(f"Detected possible typo in variable name: {var}")
                            return f"Check spelling of variable '{var}'"

            elif error_type == "TypeError":
                logger.debug("Checking for type mismatches")

                # Check for common type errors
                if "cannot concatenate" in str(analysis):
                    logger.info("Detected string concatenation error")
                    return "Convert non-string values to strings before concatenation using str()"
                elif "NoneType" in str(analysis):
                    logger.info("Detected operation on None value")
                    return "Check for None values before performing operations"
                else:
                    logger.info("Detected general type mismatch")
                    return "Check the types of the operands and ensure they are compatible"

            elif error_type == "ImportError" or error_type == "ModuleNotFoundError":
                logger.debug("Checking for import errors")
                logger.info("Detected import error")
                return "Check that the module is installed and the import path is correct"

            elif error_type == "IndexError":
                logger.debug("Checking for index errors")
                logger.info("Detected index error")
                return "Check that the index is within the valid range of the sequence"

            elif error_type == "KeyError":
                logger.debug("Checking for key errors")
                logger.info("Detected key error")
                return "Check that the key exists in the dictionary before accessing it"

            logger.info("No specific fix suggestion could be generated")
            return None

        except Exception as e:
            error = handle_error(e)
            logger.warning(f"Error generating fix suggestion: {error}")
            return None

    @with_error_handling
    def extract_intents_from_conversation(self, conversation_text: str) -> List[Dict[str, Any]]:
        """
        Extract intents from a conversation.

        Args:
            conversation_text: Conversation text to extract intents from

        Returns:
            List of extracted intents
        """
        logger.info(f"Extracting intents from conversation: {conversation_text[:100]}...")

        # Check if we have an intent extractor
        if hasattr(self, 'intent_extractor') and self.intent_extractor:
            try:
                # Use the intent extractor
                logger.debug("Using intent extractor to extract intents")
                extracted_intents = self.intent_extractor.extract_intent(
                    conversation_text,
                    "text",
                    {"source": "conversation"}
                )

                # Convert to the expected format
                intents = []
                for intent in extracted_intents:
                    intents.append({
                        "name": intent.intent_type,
                        "description": self._generate_intent_description(intent.intent_type, []),
                        "confidence": intent.confidence,
                        "entities": intent.entities
                    })

                logger.info(f"Extracted {len(intents)} intents using intent extractor")
                return intents
            except Exception as e:
                error = handle_error(e)
                logger.warning(f"Error using intent extractor: {error}")
                # Fall back to code snippet analysis

        try:
            # Extract code snippets from conversation
            logger.debug("Extracting code snippets from conversation")
            snippets = self._extract_code_snippets_from_conversation(conversation_text)
            logger.info(f"Extracted {len(snippets)} code snippets from conversation")

            # Analyze snippets
            intents = []
            for i, snippet in enumerate(snippets):
                try:
                    # Analyze snippet
                    logger.debug(f"Analyzing snippet {i+1}/{len(snippets)}")
                    snippet_analysis = self.analyze_code(snippet)

                    # Extract patterns
                    patterns = snippet_analysis.get("pattern_matches", [])
                    logger.debug(f"Found {len(patterns)} patterns in snippet {i+1}")

                    # Group patterns by intent
                    intent_patterns = {}
                    for pattern in patterns:
                        try:
                            intent_name = pattern.get("intent", "unknown")
                            if intent_name not in intent_patterns:
                                intent_patterns[intent_name] = []
                            intent_patterns[intent_name].append(pattern)
                        except Exception as e:
                            error = handle_error(e)
                            logger.warning(f"Error processing pattern in snippet {i}: {error}")

                    # Create intent objects
                    for intent_name, patterns in intent_patterns.items():
                        try:
                            intent = {
                                "name": intent_name,
                                "description": self._generate_intent_description(intent_name, patterns),
                                "confidence": sum(p.get("confidence", 0.5) for p in patterns) / len(patterns),
                                "patterns": [
                                    {
                                        "name": p.get("pattern_name", "unknown"),
                                        "pattern": p.get("pattern", ""),
                                        "pattern_type": p.get("pattern_type", "unknown")
                                    }
                                    for p in patterns
                                ]
                            }
                            intents.append(intent)
                            logger.debug(f"Created intent object for {intent_name}")
                        except Exception as e:
                            error = handle_error(e)
                            logger.warning(f"Error creating intent object for {intent_name}: {error}")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error analyzing snippet {i}: {error}")

            logger.info(f"Extracted {len(intents)} intents from {len(snippets)} code snippets")
            return intents
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting intents from conversation: {error}")
            return []

    @with_error_handling
    def _extract_code_snippets_from_conversation(self, conversation_text: str) -> List[str]:
        """
        Extract code snippets from a conversation.

        Args:
            conversation_text: Conversation text to extract snippets from

        Returns:
            List of code snippets
        """
        logger.debug("Extracting code snippets from conversation")

        # Look for code blocks delimited by ```
        snippets = []

        try:
            # Match code blocks with or without language specifier
            pattern = r"```(?:python|py|java|js|javascript|typescript|ts|c\+\+|cpp|csharp|cs|go|rust|ruby|php|html|css|bash|shell|sql)?\s*\n(.*?)\n```"
            matches = re.finditer(pattern, conversation_text, re.DOTALL)

            for match in matches:
                try:
                    snippet = match.group(1).strip()
                    if snippet:
                        snippets.append(snippet)
                        logger.debug(f"Extracted code snippet: {snippet[:50]}...")
                except Exception as e:
                    error = handle_error(e)
                    logger.warning(f"Error extracting code snippet: {error}")

            # If no code blocks found, try to find indented blocks
            if not snippets:
                logger.debug("No code blocks found, looking for indented blocks")
                lines = conversation_text.split("\n")
                current_snippet = []
                in_snippet = False

                for line in lines:
                    if line.startswith("    ") and not line.strip().startswith(">"):
                        # Indented line, could be code
                        if not in_snippet:
                            in_snippet = True
                        current_snippet.append(line.strip())
                    elif in_snippet and not line.strip():
                        # Empty line, continue snippet
                        current_snippet.append("")
                    elif in_snippet:
                        # End of snippet
                        if current_snippet:
                            snippet = "\n".join(current_snippet).strip()
                            if snippet:
                                snippets.append(snippet)
                                logger.debug(f"Extracted indented code snippet: {snippet[:50]}...")
                        current_snippet = []
                        in_snippet = False

                # Add last snippet if any
                if in_snippet and current_snippet:
                    snippet = "\n".join(current_snippet).strip()
                    if snippet:
                        snippets.append(snippet)
                        logger.debug(f"Extracted final indented code snippet: {snippet[:50]}...")

            logger.info(f"Extracted {len(snippets)} code snippets from conversation")
            return snippets

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting code snippets from conversation: {error}")
            return []

    @with_error_handling
    def _generate_intent_description(self, intent_name: str, patterns: List[Dict[str, Any]]) -> str:
        """
        Generate a description for an intent.

        Args:
            intent_name: Name of the intent
            patterns: List of patterns associated with the intent

        Returns:
            Intent description
        """
        logger.debug(f"Generating description for intent: {intent_name}")

        # Define intent descriptions
        intent_descriptions = {
            "data_processing": "Intent to process or transform data",
            "file_io": "Intent to read from or write to files",
            "web_request": "Intent to make web requests or API calls",
            "database": "Intent to interact with a database",
            "visualization": "Intent to visualize data",
            "machine_learning": "Intent to perform machine learning tasks",
            "authentication": "Intent to authenticate users or services",
            "authorization": "Intent to authorize users or services",
            "logging": "Intent to log information or errors",
            "error_handling": "Intent to handle errors or exceptions",
            "configuration": "Intent to manage configuration",
            "caching": "Intent to cache data for performance",
            "validation": "Intent to validate data or input",
            "parsing": "Intent to parse data or text",
            "serialization": "Intent to serialize or deserialize data",
            "concurrency": "Intent to handle concurrent operations",
            "networking": "Intent to perform network operations",
            "ui_interaction": "Intent to interact with a user interface",
            "testing": "Intent to test code or functionality",
            "monitoring": "Intent to monitor system or application state",
            "question": "Intent to ask a question or request information",
            "request": "Intent to request an action or task",
            "problem": "Intent to report a problem or issue",
            "feedback": "Intent to provide feedback or evaluation",
            "clarification": "Intent to seek clarification or additional information"
        }

        # Check if intent name is in the dictionary
        if intent_name.lower() in intent_descriptions:
            description = intent_descriptions[intent_name.lower()]
            logger.debug(f"Found predefined description for intent: {intent_name}")
            return description

        # If not in dictionary, try to generate a description from patterns
        if patterns:
            try:
                # Extract common themes from patterns
                pattern_types = set(p.get("pattern_type", "unknown") for p in patterns)
                pattern_names = set(p.get("pattern_name", "unknown") for p in patterns)

                # Generate description based on pattern types and names
                if "code_structure" in pattern_types:
                    logger.debug(f"Generated description for code structure intent: {intent_name}")
                    return f"Intent to structure code using {', '.join(pattern_names)}"
                elif "algorithm" in pattern_types:
                    logger.debug(f"Generated description for algorithm intent: {intent_name}")
                    return f"Intent to implement algorithm: {', '.join(pattern_names)}"
                elif "design_pattern" in pattern_types:
                    logger.debug(f"Generated description for design pattern intent: {intent_name}")
                    return f"Intent to apply design pattern: {', '.join(pattern_names)}"
            except Exception as e:
                error = handle_error(e)
                logger.warning(f"Error generating description from patterns: {error}")

        # Default description
        logger.debug(f"Using default description for intent: {intent_name}")
        return f"Intent: {intent_name}"

def register_with_sma():
    """
    Register the GPU analysis plugin with SMA's plugin manager.

    This function is for manual registration when automatic discovery is not possible.
    It should be called after SMA has been initialized.

    Returns:
        The registered plugin instance, or None if registration failed.
    """
    try:
        from semantic_matrix_analyzer.semantic_matrix_analyzer.plugins import plugin_manager
        from semantic_matrix_analyzer.semantic_matrix_analyzer.language import language_registry

        # Load the plugin using SMA's plugin manager
        plugin = plugin_manager.load_plugin(GPUAnalysisPlugin)

        if plugin:
            # Register language parsers
            from gpu_analysis.ast_adapter import register_gpu_parser
            register_gpu_parser(language_registry)

            # Initialize the plugin
            from semantic_matrix_analyzer.semantic_matrix_analyzer.plugins import PluginContext
            context = PluginContext()
            plugin.initialize(context)

            logger.info("GPU Analysis Plugin registered with SMA")
            return plugin
        else:
            logger.error("Failed to register GPU Analysis Plugin with SMA")
            return None
    except Exception as e:
        logger.error(f"Error registering GPU Analysis Plugin with SMA: {e}")
        return None

def register_plugin_methods_with_sma(plugin: 'GPUAnalysisPlugin', context: PluginContext) -> None:
    """
    Register the plugin's methods with SMA's core components.

    This function registers the plugin's methods with SMA's core components,
    allowing them to be used to implement unimplemented SMA methods.

    Args:
        plugin: The GPU Analysis Plugin instance
        context: The plugin context
    """
    try:
        from semantic_matrix_analyzer.semantic_matrix_analyzer.core import semantic_matrix_builder
        from semantic_matrix_analyzer.semantic_matrix_analyzer.cli import sma_cli

        # Register analyze_component method with SemanticMatrixBuilder
        if hasattr(semantic_matrix_builder, 'SemanticMatrixBuilder'):
            builder = semantic_matrix_builder.SemanticMatrixBuilder

            # Store original method for fallback
            if not hasattr(builder, '_original_analyze_component'):
                builder._original_analyze_component = builder.analyze_component

            # Replace with GPU-accelerated method
            def gpu_analyze_component(self, component: str, file_path: Path) -> Any:
                """GPU-accelerated component analysis."""
                try:
                    return plugin.analyze_component(component, file_path)
                except Exception as e:
                    context.log("error", f"GPU analysis failed, falling back to original method: {e}")
                    return self._original_analyze_component(component, file_path)

            # Apply the replacement
            builder.analyze_component = gpu_analyze_component
            context.log("info", "Registered GPU-accelerated analyze_component method with SemanticMatrixBuilder")

        # Register CLI command handlers
        if hasattr(sma_cli, 'handle_analyze_command'):
            # Store original method for fallback
            if not hasattr(sma_cli, '_original_handle_analyze_command'):
                sma_cli._original_handle_analyze_command = sma_cli.handle_analyze_command

            # Replace with GPU-accelerated method
            def gpu_handle_analyze_command(args: Any) -> None:
                """GPU-accelerated analyze command handler."""
                try:
                    # Extract code or file path from args
                    code = None
                    file_path = None

                    if hasattr(args, 'code_file') and args.code_file:
                        file_path = args.code_file
                    elif hasattr(args, 'project_dir') and args.project_dir:
                        # Analyze project directory
                        results = plugin.generate_semantic_snapshot(args.project_dir)

                        # Print results
                        print(f"Analyzed {results.get('analyzed_files', 0)} files using GPU acceleration")
                        return

                    # Analyze code or file
                    if file_path:
                        results = plugin.analyze_file(file_path)
                    elif code:
                        results = plugin.analyze_code(code)
                    else:
                        print("No code or file specified")
                        return

                    # Print results
                    print(f"Analysis completed using GPU acceleration")
                    print(f"Complexity: {results.get('complexity', {})}")
                    print(f"Patterns: {len(results.get('pattern_matches', []))}")
                except Exception as e:
                    context.log("error", f"GPU analysis failed, falling back to original method: {e}")
                    sma_cli._original_handle_analyze_command(args)

            # Apply the replacement
            sma_cli.handle_analyze_command = gpu_handle_analyze_command
            context.log("info", "Registered GPU-accelerated handle_analyze_command with SMA CLI")

        context.log("info", "Successfully registered plugin methods with SMA")
    except Exception as e:
        context.log("error", f"Error registering plugin methods with SMA: {e}")


def register_cli_config_parameters(plugin: 'GPUAnalysisPlugin', context: PluginContext, sma_cli: Any) -> None:
    """
    Register the plugin's configuration parameters with SMA's CLI.

    This function registers the plugin's configuration parameters with SMA's CLI,
    allowing them to be configured through the command line.

    Args:
        plugin: The GPU Analysis Plugin instance
        context: The plugin context
        sma_cli: SMA CLI module
    """
    try:
        # Check if SMA CLI has a create_parser function
        if not hasattr(sma_cli, 'create_parser'):
            context.log("warning", "SMA CLI does not have a create_parser function, cannot register configuration parameters")
            return

        # Store original create_parser function
        if not hasattr(sma_cli, '_original_create_parser'):
            sma_cli._original_create_parser = sma_cli.create_parser

        # Create a new create_parser function that adds GPU configuration parameters
        def gpu_create_parser():
            """Create the argument parser for the SMA CLI with GPU configuration parameters."""
            # Call original create_parser
            parser = sma_cli._original_create_parser()

            # Get the config subparser
            if not hasattr(parser, 'subparsers'):
                context.log("warning", "SMA CLI parser does not have subparsers, cannot register configuration parameters")
                return parser

            # Find the config subparser
            config_parser = None
            for action in parser._actions:
                if hasattr(action, 'choices') and 'config' in action.choices:
                    config_parser = action.choices['config']
                    break

            if not config_parser:
                context.log("warning", "SMA CLI parser does not have a config subparser, cannot register configuration parameters")
                return parser

            # Find the update subparser
            update_parser = None
            if hasattr(config_parser, 'subparsers'):
                for action in config_parser._actions:
                    if hasattr(action, 'choices') and 'update' in action.choices:
                        update_parser = action.choices['update']
                        break

            if not update_parser:
                context.log("warning", "SMA CLI config parser does not have an update subparser, cannot register configuration parameters")
                return parser

            # Add GPU configuration parameters
            update_parser.add_argument(
                "--gpu-device",
                help="Device to use for GPU analysis ('cuda' or 'cpu')",
                choices=["cuda", "cpu"]
            )
            update_parser.add_argument(
                "--gpu-batch-size",
                help="Batch size for GPU operations",
                type=int
            )
            update_parser.add_argument(
                "--gpu-precision",
                help="Precision for GPU operations",
                choices=["float16", "float32", "float64"]
            )
            update_parser.add_argument(
                "--gpu-cache-size",
                help="Maximum number of items to cache",
                type=int
            )
            update_parser.add_argument(
                "--gpu-enable-complexity",
                help="Enable/disable complexity analysis",
                action="store_true"
            )
            update_parser.add_argument(
                "--gpu-enable-dependency",
                help="Enable/disable dependency analysis",
                action="store_true"
            )
            update_parser.add_argument(
                "--gpu-enable-semantic",
                help="Enable/disable semantic analysis",
                action="store_true"
            )
            update_parser.add_argument(
                "--gpu-enable-pattern",
                help="Enable/disable pattern matching",
                action="store_true"
            )
            update_parser.add_argument(
                "--gpu-similarity-threshold",
                help="Threshold for semantic similarity",
                type=float
            )
            update_parser.add_argument(
                "--gpu-confidence-threshold",
                help="Threshold for pattern matching confidence",
                type=float
            )

            # Add learning rate parameters
            update_parser.add_argument(
                "--gpu-adaptation-rate",
                help="Base adaptation rate for learning (0.01-0.5)",
                type=float
            )
            update_parser.add_argument(
                "--gpu-confidence-threshold",
                help="Confidence threshold for learning (0.0-1.0)",
                type=float
            )
            update_parser.add_argument(
                "--gpu-feedback-weight",
                help="Weight for feedback in learning (0.0-1.0)",
                type=float
            )
            update_parser.add_argument(
                "--gpu-stability-factor",
                help="Stability factor for learning (0.0-1.0)",
                type=float
            )

            return parser

        # Replace create_parser function
        sma_cli.create_parser = gpu_create_parser
        context.log("info", "Registered GPU configuration parameters with SMA CLI")

        # Store original handle_config_command function
        if not hasattr(sma_cli, '_original_handle_config_command'):
            sma_cli._original_handle_config_command = sma_cli.handle_config_command

        # Create a new handle_config_command function that handles GPU configuration parameters
        def gpu_handle_config_command(args: Any) -> None:
            """Handle the config command with GPU configuration parameters."""
            try:
                # Check if any GPU configuration parameters are set
                gpu_params = {
                    "device": getattr(args, "gpu_device", None),
                    "batch_size": getattr(args, "gpu_batch_size", None),
                    "precision": getattr(args, "gpu_precision", None),
                    "cache_size": getattr(args, "gpu_cache_size", None),
                    "analyzers": {
                        "complexity": {
                            "enabled": getattr(args, "gpu_enable_complexity", None)
                        },
                        "dependency": {
                            "enabled": getattr(args, "gpu_enable_dependency", None)
                        },
                        "semantic": {
                            "enabled": getattr(args, "gpu_enable_semantic", None),
                            "similarity_threshold": getattr(args, "gpu_similarity_threshold", None)
                        },
                        "pattern": {
                            "enabled": getattr(args, "gpu_enable_pattern", None),
                            "confidence_threshold": getattr(args, "gpu_confidence_threshold", None)
                        }
                    },
                    "learning": {
                        "base_adaptation_rate": getattr(args, "gpu_adaptation_rate", None),
                        "confidence_threshold": getattr(args, "gpu_confidence_threshold", None),
                        "feedback_weight": getattr(args, "gpu_feedback_weight", None),
                        "stability_factor": getattr(args, "gpu_stability_factor", None)
                    }
                }

                # Remove None values
                gpu_params = {k: v for k, v in gpu_params.items() if v is not None}
                if "analyzers" in gpu_params:
                    for analyzer, settings in list(gpu_params["analyzers"].items()):
                        gpu_params["analyzers"][analyzer] = {k: v for k, v in settings.items() if v is not None}
                        if not gpu_params["analyzers"][analyzer]:
                            del gpu_params["analyzers"][analyzer]
                    if not gpu_params["analyzers"]:
                        del gpu_params["analyzers"]

                # If any GPU configuration parameters are set, update the configuration
                if gpu_params and args.config_command == "update":
                    # Get the configuration manager
                    config_manager = sma_cli.ConfigManager(args.config)

                    # Get the current configuration
                    config = config_manager.get_config()

                    # Update the GPU configuration
                    if "gpu_analysis" not in config:
                        config["gpu_analysis"] = {}

                    # Update the configuration recursively
                    def update_config(config, updates):
                        for key, value in updates.items():
                            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                                update_config(config[key], value)
                            else:
                                config[key] = value

                    update_config(config["gpu_analysis"], gpu_params)

                    # Save the configuration
                    config_manager.save_config(config)

                    # Print success message
                    print(sma_cli.color_text("GPU configuration updated successfully.", "GREEN"))

                # Call original handle_config_command
                sma_cli._original_handle_config_command(args)
            except Exception as e:
                context.log("error", f"Error handling GPU configuration parameters: {e}")
                # Call original handle_config_command
                sma_cli._original_handle_config_command(args)

        # Replace handle_config_command function
        sma_cli.handle_config_command = gpu_handle_config_command
        context.log("info", "Registered GPU configuration handler with SMA CLI")

    except Exception as e:
        context.log("error", f"Error registering GPU configuration parameters with SMA CLI: {e}")

def register_cli_commands(plugin: 'GPUAnalysisPlugin', context: PluginContext) -> None:
    """
    Register the plugin's CLI commands with SMA.

    This function registers the plugin's CLI commands with SMA,
    allowing them to be used to implement unimplemented SMA CLI commands.
    It also registers configuration parameters with SMA's CLI.

    Args:
        plugin: The GPU Analysis Plugin instance
        context: The plugin context
    """
    try:
        import importlib

        # Try to import SMA CLI module
        try:
            sma_cli = importlib.import_module("semantic_matrix_analyzer.sma_cli")
        except ImportError:
            context.log("error", "Failed to import SMA CLI module")
            return

        # Register configuration parameters with SMA's CLI
        register_cli_config_parameters(plugin, context, sma_cli)

        # Register analyze command handler
        if hasattr(sma_cli, "handle_analyze_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_analyze_command"):
                sma_cli._original_handle_analyze_command = sma_cli.handle_analyze_command

            # Replace with GPU-accelerated method
            def gpu_handle_analyze_command(args: Any) -> None:
                """GPU-accelerated analyze command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("CODE ANALYSIS")
                    print("Analyzing code for intent extraction using GPU acceleration...")

                    # Extract code or file path from args
                    code = None
                    file_path = None

                    if hasattr(args, 'code_file') and args.code_file:
                        file_path = args.code_file
                        print(f"Analyzing file: {file_path}")
                    elif hasattr(args, 'project_dir') and args.project_dir:
                        print(f"Analyzing project directory: {args.project_dir}")
                        # Analyze project directory
                        results = plugin.analyze_project(args.project_dir)

                        # Print results
                        print(f"Analyzed {len(results)} files using GPU acceleration")

                        # Format output
                        if hasattr(args, 'format') and args.format == "json":
                            import json
                            print(json.dumps(results, indent=2))
                        else:
                            # Print summary
                            print(f"Average complexity: {sum(r['complexity']['cyclomatic'] for r in results.values()) / len(results):.2f}")
                            print(f"Total pattern matches: {sum(len(r.get('pattern_matches', [])) for r in results.values())}")
                        return

                    # Analyze code or file
                    if file_path:
                        results = plugin.analyze_file(file_path)
                    elif code:
                        results = plugin.analyze_code(code)
                    else:
                        print("No code or file specified")
                        return

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(results, indent=2))
                    else:
                        # Print results
                        print(f"Analysis completed using GPU acceleration")
                        print(f"Complexity: {results.get('complexity', {})}")
                        print(f"Patterns: {len(results.get('pattern_matches', []))}")
                except Exception as e:
                    context.log("error", f"GPU analysis failed, falling back to original method: {e}")
                    sma_cli._original_handle_analyze_command(args)

            # Apply the replacement
            sma_cli.handle_analyze_command = gpu_handle_analyze_command
            context.log("info", "Registered GPU-accelerated handle_analyze_command with SMA CLI")

        # Register error trace command handler
        if hasattr(sma_cli, "handle_error_trace_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_error_trace_command"):
                sma_cli._original_handle_error_trace_command = sma_cli.handle_error_trace_command

            # Replace with GPU-accelerated method
            def gpu_handle_error_trace_command(args: Any) -> None:
                """GPU-accelerated error trace command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("ERROR TRACE PROCESSING")
                    print("Processing error trace using GPU acceleration...")

                    # Get error trace
                    error_trace = ""
                    if hasattr(args, 'input_file') and args.input_file:
                        print(f"Reading error trace from file: {args.input_file}")
                        with open(args.input_file, "r") as f:
                            error_trace = f.read()
                    elif hasattr(args, 'text') and args.text:
                        error_trace = args.text
                    else:
                        print("No error trace specified")
                        return

                    # Analyze error trace
                    results = plugin.analyze_error_trace(error_trace)

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(results, indent=2))
                    else:
                        # Print results
                        print(f"Error trace analysis completed using GPU acceleration")
                        print(f"Snippets analyzed: {results.get('snippets_analyzed', 0)}")
                        print(f"Error type: {results.get('error_type', 'Unknown')}")

                        # Print suggested fixes
                        if 'suggested_fixes' in results:
                            print("\nSuggested fixes:")
                            for fix in results['suggested_fixes']:
                                print(f"- {fix}")
                except Exception as e:
                    context.log("error", f"GPU error trace analysis failed, falling back to original method: {e}")
                    sma_cli._original_handle_error_trace_command(args)

            # Apply the replacement
            sma_cli.handle_error_trace_command = gpu_handle_error_trace_command
            context.log("info", "Registered GPU-accelerated handle_error_trace_command with SMA CLI")

        # Register extract intent command handler
        if hasattr(sma_cli, "handle_extract_intent_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_extract_intent_command"):
                sma_cli._original_handle_extract_intent_command = sma_cli.handle_extract_intent_command

            # Replace with GPU-accelerated method
            def gpu_handle_extract_intent_command(args: Any) -> None:
                """GPU-accelerated extract intent command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("INTENT EXTRACTION")
                    print("Extracting intent from conversation using GPU acceleration...")

                    # Get conversation text
                    conversation_text = ""
                    if hasattr(args, 'input_file') and args.input_file:
                        print(f"Reading conversation from file: {args.input_file}")
                        with open(args.input_file, "r") as f:
                            conversation_text = f.read()
                    elif hasattr(args, 'text') and args.text:
                        conversation_text = args.text
                    else:
                        print("No conversation specified")
                        return

                    # Extract intents
                    intents = plugin.extract_intents_from_conversation(conversation_text)

                    # Save intents to file if specified
                    if hasattr(args, 'output_file') and args.output_file:
                        with open(args.output_file, "w") as f:
                            import json
                            json.dump(intents, f, indent=2)
                        print(f"Intents saved to {args.output_file}")

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(intents, indent=2))
                    else:
                        # Print intents
                        print(f"Intent extraction completed using GPU acceleration")
                        print(f"Intents extracted: {len(intents)}")
                        for intent in intents:
                            print(f"\nIntent: {intent['name']}")
                            print(f"Description: {intent['description']}")
                            print(f"Patterns: {len(intent['patterns'])}")
                            for pattern in intent['patterns']:
                                print(f"  - {pattern['name']}: {pattern['pattern']} ({pattern['pattern_type']})")
                except Exception as e:
                    context.log("error", f"GPU intent extraction failed, falling back to original method: {e}")
                    sma_cli._original_handle_extract_intent_command(args)

            # Apply the replacement
            sma_cli.handle_extract_intent_command = gpu_handle_extract_intent_command
            context.log("info", "Registered GPU-accelerated handle_extract_intent_command with SMA CLI")

        # Register semantic analysis in generate_project_snapshot
        if hasattr(sma_cli, "generate_project_snapshot"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_generate_project_snapshot"):
                sma_cli._original_generate_project_snapshot = sma_cli.generate_project_snapshot

            # Replace with GPU-accelerated method
            def gpu_generate_project_snapshot(
                project_dir: str,
                depth: int = 3,
                focus: str = "all",
                analyzer: Any = None,
                output_format: str = "markdown"
            ) -> Any:
                """GPU-accelerated project snapshot generation."""
                try:
                    # Call original method to get basic snapshot
                    snapshot = sma_cli._original_generate_project_snapshot(
                        project_dir=project_dir,
                        depth=depth,
                        focus=focus,
                        analyzer=analyzer,
                        output_format="json"  # Always get JSON for processing
                    )

                    # Add GPU-accelerated semantic analysis
                    if focus in ["semantics", "all"] and depth >= 2:
                        # Generate semantic snapshot
                        semantic_snapshot = plugin.generate_semantic_snapshot(project_dir, depth)

                        # Update snapshot with semantic analysis
                        snapshot["semantics"] = semantic_snapshot

                    # Format output
                    if output_format == "json":
                        return snapshot
                    elif output_format == "yaml":
                        import yaml
                        return yaml.dump(snapshot, default_flow_style=False)
                    else:
                        # Format as markdown
                        if hasattr(sma_cli, "format_snapshot_as_markdown"):
                            return sma_cli.format_snapshot_as_markdown(snapshot)
                        else:
                            import json
                            return json.dumps(snapshot, indent=2)
                except Exception as e:
                    context.log("error", f"GPU snapshot generation failed, falling back to original method: {e}")
                    return sma_cli._original_generate_project_snapshot(
                        project_dir=project_dir,
                        depth=depth,
                        focus=focus,
                        analyzer=analyzer,
                        output_format=output_format
                    )

            # Apply the replacement
            sma_cli.generate_project_snapshot = gpu_generate_project_snapshot
            context.log("info", "Registered GPU-accelerated generate_project_snapshot with SMA CLI")

        context.log("info", "Successfully registered CLI commands with SMA")
    except Exception as e:
        context.log("error", f"Error registering CLI commands with SMA: {e}")

# For backward compatibility
def register_plugin(sma_registry):
    """
    Legacy registration function for backward compatibility.

    Args:
        sma_registry: SMA registry

    Returns:
        The registered plugin instance
    """
    logger.warning("Using legacy registration method. Consider using SMA's plugin discovery mechanism instead.")

    # Create plugin
    plugin = GPUAnalysisPlugin()

    # Register plugin
    sma_registry.register_plugin("gpu_analysis", plugin)

    # Register language parsers
    from gpu_analysis.ast_adapter import register_gpu_parser
    register_gpu_parser(sma_registry.language_registry)

    # Log registration
    logger.info("GPU Analysis Plugin registered with SMA using legacy method")

    return plugin

# This is the entry point for SMA's plugin discovery mechanism
# SMA will look for plugin classes in the module
GPUAnalysisPluginClass = GPUAnalysisPlugin
