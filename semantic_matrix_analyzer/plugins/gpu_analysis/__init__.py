"""
GPU-Accelerated Semantic Analysis Module

This module provides GPU-accelerated implementations of semantic analysis tools,
including AST traversal, token scoring, and pattern weight computation.

It is designed to integrate with the Semantic Matrix Analyzer (SMA) project
and provides GPU-accelerated alternatives to the core analysis components.

Key components:
- AST Tensorizer: Converts Python AST to tensor representation for GPU processing
- Pattern Matchers: GPU-accelerated implementations of pattern matching algorithms
- Semantic Analyzers: Neural network models for analyzing code semantics
- Execution Engine: Parallel execution of analysis tasks on GPU

Usage:
    from gpu_analysis import ASTTensorizer, GPUPatternMatcher, SemanticAnalyzer

    # Initialize components
    tensorizer = ASTTensorizer()
    matcher = GPUPatternMatcher()
    analyzer = SemanticAnalyzer()

    # Analyze code
    ast_tensors = tensorizer.tensorize(code)
    patterns = matcher.match_patterns(ast_tensors)
    results = analyzer.analyze(ast_tensors, patterns)
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Union, Any

# Set up logging
logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    else:
        DEFAULT_DEVICE = "cpu"
        logger.warning("CUDA not available. Using CPU for GPU-accelerated analysis.")
except ImportError:
    HAS_TORCH = False
    DEFAULT_DEVICE = "cpu"
    logger.warning("PyTorch not found. GPU-accelerated analysis will not be available.")

# Import core components
from .ast_tensor import ASTTensorizer
from .formalism import ParseFormalism
from .executor import ParsingExecutor
from .ast_adapter import ASTAdapter, GPULanguageParser, register_gpu_parser

# Import pattern matchers
from .patterns import (
    Pattern, PatternMatch, PatternType,
    GPUPatternMatcher, TokenPattern, RegexPattern, ASTPattern,
    create_token_pattern, create_regex_pattern, create_ast_pattern
)

# Import analyzers
from .analyzers import (
    ComplexityAnalyzer, DependencyAnalyzer,
    SemanticAnalyzer, IntentExtractor
)

# Import configuration integration
from .config_integration import (
    GPU_ANALYSIS_CONFIG_SCHEMA,
    get_gpu_config_from_sma,
    register_config_schema_with_sma,
    validate_config
)

# Import dynamic configuration
from .dynamic_config import (
    ConfigObserver,
    DynamicConfigManager,
    LearningRateManager
)

# Import pattern extraction
from .pattern_extraction import (
    Pattern,
    PatternMatch,
    PatternExtractor
)

# Import intent extraction
from .intent_extraction import (
    Intent,
    IntentExtractor
)

# Import feedback processing
from .feedback_processor import (
    FeedbackType,
    Feedback,
    FeedbackResult,
    FeedbackProcessor
)

# Import logging integration
from .logging_integration import (
    get_logger,
    set_context,
    SMALoggerAdapter
)

# Import error handling
from .error_handling import (
    GPUAnalysisError,
    GPUNotAvailableError,
    GPUMemoryError,
    GPUAnalysisConfigError,
    GPUAnalysisRuntimeError,
    GPUAnalysisParsingError,
    GPUAnalysisComponentError,
    GPUAnalysisPatternError,
    GPUAnalysisIntentError,
    handle_error,
    check_gpu_available,
    with_error_handling
)

# Define exports
__all__ = [
    # Core components
    'ASTTensorizer',
    'ParseFormalism',
    'ParsingExecutor',
    'ASTAdapter',
    'GPULanguageParser',
    'register_gpu_parser',

    # Pattern matchers
    'Pattern',
    'PatternMatch',
    'PatternType',
    'GPUPatternMatcher',
    'TokenPattern',
    'RegexPattern',
    'ASTPattern',
    'create_token_pattern',
    'create_regex_pattern',
    'create_ast_pattern',

    # Analyzers
    'ComplexityAnalyzer',
    'DependencyAnalyzer',
    'SemanticAnalyzer',
    'IntentExtractor',

    # Configuration integration
    'GPU_ANALYSIS_CONFIG_SCHEMA',
    'get_gpu_config_from_sma',
    'register_config_schema_with_sma',
    'validate_config',

    # Dynamic configuration
    'ConfigObserver',
    'DynamicConfigManager',
    'LearningRateManager',

    # Pattern extraction
    'Pattern',
    'PatternMatch',
    'PatternExtractor',

    # Intent extraction
    'Intent',
    'IntentExtractor',

    # Feedback processing
    'FeedbackType',
    'Feedback',
    'FeedbackResult',
    'FeedbackProcessor',

    # Logging integration
    'get_logger',
    'set_context',
    'SMALoggerAdapter',

    # Error handling
    'GPUAnalysisError',
    'GPUNotAvailableError',
    'GPUMemoryError',
    'GPUAnalysisConfigError',
    'GPUAnalysisRuntimeError',
    'GPUAnalysisParsingError',
    'GPUAnalysisComponentError',
    'GPUAnalysisPatternError',
    'GPUAnalysisIntentError',
    'handle_error',
    'check_gpu_available',
    'with_error_handling',

    # Constants
    'HAS_TORCH',
    'DEFAULT_DEVICE'
]

def get_device() -> str:
    """
    Get the default device for GPU-accelerated analysis.

    Returns:
        Device string ("cuda" or "cpu")
    """
    return DEFAULT_DEVICE

def set_device(device: str) -> None:
    """
    Set the default device for GPU-accelerated analysis.

    Args:
        device: Device to use ("cuda" or "cpu")
    """
    global DEFAULT_DEVICE
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'.")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU instead.")
        DEFAULT_DEVICE = "cpu"
    else:
        DEFAULT_DEVICE = device
