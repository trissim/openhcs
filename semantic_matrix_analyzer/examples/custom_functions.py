"""
Custom functions for agent-driven analysis.

This module provides custom functions for the agent-driven analyzer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.agent import AgentDrivenAnalyzer
from semantic_matrix_analyzer.agent.models import FileComplexity


def custom_relevance_function(analyzer: AgentDrivenAnalyzer, file_path: Path) -> float:
    """Custom function for calculating relevance score.
    
    Args:
        analyzer: The agent-driven analyzer.
        file_path: Path to the file.
        
    Returns:
        Relevance score from 0.0 to 1.0.
    """
    # Start with minimum relevance
    relevance = analyzer.config.min_relevance_score
    
    # Check if file is explicitly mentioned
    if file_path in analyzer.user_intent.file_mentions:
        relevance = max(relevance, analyzer.config.explicit_mention_weight)
    
    # Check if file matches mentioned components
    for component in analyzer.user_intent.component_mentions:
        if component.lower() in file_path.name.lower():
            relevance = max(relevance, analyzer.config.component_match_weight)
    
    # Check if file is central to the codebase
    metrics = analyzer.file_metrics.get(file_path)
    if metrics and metrics.is_central:
        relevance = max(relevance, analyzer.config.central_file_weight)
    
    # Check if file has been useful in the past
    history = analyzer.file_history.get(file_path)
    if history and history.usefulness_ratio > 0.5:
        usefulness_boost = analyzer.config.historical_usefulness_weight * history.usefulness_ratio
        relevance = max(relevance, analyzer.config.historical_usefulness_weight + usefulness_boost)
    
    # Custom logic: Prioritize test files
    if "test" in file_path.name.lower():
        relevance = max(relevance, 0.75)
    
    # Custom logic: Prioritize files with specific concerns
    for concern in analyzer.user_intent.primary_concerns:
        # Check if the file content contains the concern
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                if concern.lower() in content:
                    relevance = max(relevance, 0.85)
                    break
        except Exception:
            pass
    
    return relevance


def custom_information_value_function(analyzer: AgentDrivenAnalyzer, file_path: Path) -> float:
    """Custom function for calculating information value score.
    
    Args:
        analyzer: The agent-driven analyzer.
        file_path: Path to the file.
        
    Returns:
        Information value score from 0.0 to 1.0.
    """
    # Start with minimum information value
    info_value = analyzer.config.min_information_value
    
    # Check file complexity
    metrics = analyzer.file_metrics.get(file_path)
    if metrics:
        # Adjust based on complexity
        if metrics.complexity == FileComplexity.VERY_HIGH:
            info_value = max(info_value, 0.9 * analyzer.config.complexity_weight)
        elif metrics.complexity == FileComplexity.HIGH:
            info_value = max(info_value, 0.7 * analyzer.config.complexity_weight)
        elif metrics.complexity == FileComplexity.MEDIUM:
            info_value = max(info_value, 0.5 * analyzer.config.complexity_weight)
        elif metrics.complexity == FileComplexity.LOW:
            info_value = min(info_value, 0.3 * analyzer.config.complexity_weight)
        elif metrics.complexity == FileComplexity.TRIVIAL:
            info_value = min(info_value, 0.1 * analyzer.config.complexity_weight)
        
        # Files with many dependents have higher information value
        if len(metrics.dependents) > analyzer.config.many_dependencies_threshold:
            info_value = max(info_value, 0.8 * analyzer.config.dependency_weight)
        elif len(metrics.dependents) > analyzer.config.many_dependencies_threshold / 2:
            info_value = max(info_value, 0.6 * analyzer.config.dependency_weight)
        
        # Frequently changed files have higher information value
        if metrics.change_frequency > analyzer.config.high_change_frequency:
            info_value = max(info_value, 0.8 * analyzer.config.change_frequency_weight)
        elif metrics.change_frequency > analyzer.config.medium_change_frequency:
            info_value = max(info_value, 0.6 * analyzer.config.change_frequency_weight)
    
    # Custom logic: Interface files have higher information value
    if "interface" in file_path.name.lower() or "api" in file_path.name.lower():
        info_value = max(info_value, 0.9)
    
    # Custom logic: Configuration files have higher information value
    if "config" in file_path.name.lower() or file_path.suffix in [".json", ".yaml", ".yml", ".toml"]:
        info_value = max(info_value, 0.8)
    
    return info_value


def custom_effort_function(analyzer: AgentDrivenAnalyzer, file_path: Path) -> float:
    """Custom function for calculating effort required.
    
    Args:
        analyzer: The agent-driven analyzer.
        file_path: Path to the file.
        
    Returns:
        Effort score (higher means more effort required).
    """
    # Start with default effort
    effort = 1.0
    
    # Check file size and complexity
    metrics = analyzer.file_metrics.get(file_path)
    if metrics:
        # Large files require more effort
        if metrics.line_count > analyzer.config.very_large_file_threshold:
            effort *= analyzer.config.very_large_file_multiplier
        elif metrics.line_count > analyzer.config.large_file_threshold:
            effort *= analyzer.config.large_file_multiplier
        
        # Complex files require more effort
        if metrics.complexity == FileComplexity.VERY_HIGH:
            effort *= analyzer.config.very_complex_file_multiplier
        elif metrics.complexity == FileComplexity.HIGH:
            effort *= analyzer.config.complex_file_multiplier
        
        # Files with many dependencies require more effort
        if len(metrics.dependencies) > analyzer.config.many_dependencies_threshold:
            effort *= analyzer.config.many_dependencies_multiplier
    
    # Custom logic: Test files require less effort
    if "test" in file_path.name.lower():
        effort *= 0.8
    
    # Custom logic: Documentation files require less effort
    if "doc" in file_path.name.lower() or file_path.suffix in [".md", ".rst", ".txt"]:
        effort *= 0.7
    
    # Custom logic: Generated files require more effort
    if "generated" in file_path.name.lower() or "auto" in file_path.name.lower():
        effort *= 1.5
    
    # Ensure effort doesn't exceed the maximum multiplier
    return max(0.1, min(effort, analyzer.config.max_effort_multiplier))
