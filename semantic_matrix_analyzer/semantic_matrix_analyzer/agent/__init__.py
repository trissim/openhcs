"""
Agent-driven analysis module for Semantic Matrix Analyzer.

This module provides functionality for agent-driven file selection and analysis,
leveraging the agent's judgment to reduce human cognitive load.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from semantic_matrix_analyzer.agent.config import AgentConfig
from semantic_matrix_analyzer.agent.models import (
    FileAnalysisHistory, FileAnalysisMetrics, FileComplexity,
    FileRelevance, FileSelectionResult, UserIntent
)


class AgentDrivenAnalyzer:
    """Agent-driven analyzer for intelligent file selection and analysis.

    This class implements the agent-driven approach to code analysis, where the agent
    uses its judgment to select which files are worth analyzing based on relevance,
    information value, and effort required.
    """

    def __init__(
        self,
        codebase_path: Union[str, Path],
        user_intent: UserIntent,
        config: Optional[AgentConfig] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the analyzer.

        Args:
            codebase_path: Path to the codebase.
            user_intent: User's intent for analysis.
            config: Configuration for the analyzer (optional).
            config_path: Path to a configuration file (optional).
                If both config and config_path are provided, config takes precedence.
        """
        self.codebase_path = Path(codebase_path)
        self.user_intent = user_intent
        self.file_history: Dict[Path, FileAnalysisHistory] = {}
        self.file_metrics: Dict[Path, FileAnalysisMetrics] = {}

        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = AgentConfig.from_json(config_path)
        else:
            self.config = AgentConfig()

        # For backward compatibility
        self.selection_threshold = self.config.selection_threshold

    def calculate_relevance(self, file_path: Path) -> float:
        """Calculate relevance score for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Relevance score from 0.0 to 1.0.
        """
        # Use custom function if provided
        if self.config.calculate_relevance_func:
            return self.config.calculate_relevance_func(self, file_path)

        # Start with minimum relevance
        relevance = self.config.min_relevance_score

        # Check if file is explicitly mentioned
        if file_path in self.user_intent.file_mentions:
            relevance = max(relevance, self.config.explicit_mention_weight)

        # Check if file matches mentioned components
        for component in self.user_intent.component_mentions:
            if component.lower() in file_path.name.lower():
                relevance = max(relevance, self.config.component_match_weight)

        # Check if file is central to the codebase
        metrics = self.file_metrics.get(file_path)
        if metrics and metrics.is_central:
            relevance = max(relevance, self.config.central_file_weight)

        # Check if file has been useful in the past
        history = self.file_history.get(file_path)
        if history and history.usefulness_ratio > 0.5:
            usefulness_boost = self.config.historical_usefulness_weight * history.usefulness_ratio
            relevance = max(relevance, self.config.historical_usefulness_weight + usefulness_boost)

        return relevance

    def calculate_information_value(self, file_path: Path) -> float:
        """Calculate information value score for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Information value score from 0.0 to 1.0.
        """
        # Use custom function if provided
        if self.config.calculate_information_value_func:
            return self.config.calculate_information_value_func(self, file_path)

        # Start with minimum information value
        info_value = self.config.min_information_value

        # Check file complexity
        metrics = self.file_metrics.get(file_path)
        if metrics:
            # Adjust based on complexity
            if metrics.complexity == FileComplexity.VERY_HIGH:
                info_value = max(info_value, 0.9 * self.config.complexity_weight)
            elif metrics.complexity == FileComplexity.HIGH:
                info_value = max(info_value, 0.7 * self.config.complexity_weight)
            elif metrics.complexity == FileComplexity.MEDIUM:
                info_value = max(info_value, 0.5 * self.config.complexity_weight)
            elif metrics.complexity == FileComplexity.LOW:
                info_value = min(info_value, 0.3 * self.config.complexity_weight)
            elif metrics.complexity == FileComplexity.TRIVIAL:
                info_value = min(info_value, 0.1 * self.config.complexity_weight)

            # Files with many dependents have higher information value
            if len(metrics.dependents) > self.config.many_dependencies_threshold:
                info_value = max(info_value, 0.8 * self.config.dependency_weight)
            elif len(metrics.dependents) > self.config.many_dependencies_threshold / 2:
                info_value = max(info_value, 0.6 * self.config.dependency_weight)

            # Frequently changed files have higher information value
            if metrics.change_frequency > self.config.high_change_frequency:
                info_value = max(info_value, 0.8 * self.config.change_frequency_weight)
            elif metrics.change_frequency > self.config.medium_change_frequency:
                info_value = max(info_value, 0.6 * self.config.change_frequency_weight)

        return info_value

    def calculate_effort(self, file_path: Path) -> float:
        """Calculate effort required to analyze a file.

        Args:
            file_path: Path to the file.

        Returns:
            Effort score (higher means more effort required).
        """
        # Use custom function if provided
        if self.config.calculate_effort_func:
            return self.config.calculate_effort_func(self, file_path)

        # Start with default effort
        effort = 1.0

        # Check file size and complexity
        metrics = self.file_metrics.get(file_path)
        if metrics:
            # Large files require more effort
            if metrics.line_count > self.config.very_large_file_threshold:
                effort *= self.config.very_large_file_multiplier
            elif metrics.line_count > self.config.large_file_threshold:
                effort *= self.config.large_file_multiplier

            # Complex files require more effort
            if metrics.complexity == FileComplexity.VERY_HIGH:
                effort *= self.config.very_complex_file_multiplier
            elif metrics.complexity == FileComplexity.HIGH:
                effort *= self.config.complex_file_multiplier

            # Files with many dependencies require more effort
            if len(metrics.dependencies) > self.config.many_dependencies_threshold:
                effort *= self.config.many_dependencies_multiplier

        # Ensure effort doesn't exceed the maximum multiplier
        return max(0.1, min(effort, self.config.max_effort_multiplier))

    def select_files_for_analysis(self, threshold: Optional[float] = None) -> List[FileSelectionResult]:
        """Select files worth analyzing based on relevance, value, and effort.

        Args:
            threshold: Selection threshold (optional, uses instance threshold if not provided).

        Returns:
            List of FileSelectionResult objects, sorted by score in descending order.
        """
        if threshold is None:
            threshold = self.selection_threshold

        selected_files = []

        for file_path in self.get_all_files():
            # Skip files that are not relevant to the user's intent
            if not self.user_intent.is_file_relevant(file_path):
                continue

            # Skip files with extensions not in the config
            if file_path.suffix not in self.config.file_extensions:
                continue

            relevance = self.calculate_relevance(file_path)
            info_value = self.calculate_information_value(file_path)
            effort = self.calculate_effort(file_path)

            score = (relevance * info_value) / effort

            # Update history
            if file_path not in self.file_history:
                self.file_history[file_path] = FileAnalysisHistory(file_path=file_path)

            self.file_history[file_path].last_relevance_score = relevance
            self.file_history[file_path].last_information_value = info_value
            self.file_history[file_path].last_effort_score = effort

            if score > threshold:
                selected_files.append(FileSelectionResult(
                    file_path=file_path,
                    score=score,
                    relevance=relevance,
                    information_value=info_value,
                    effort=effort
                ))

        # Sort by score in descending order
        selected_files.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_files if specified
        if self.config.max_files and len(selected_files) > self.config.max_files:
            selected_files = selected_files[:self.config.max_files]

        return selected_files

    def get_all_files(self) -> List[Path]:
        """Get all files in the codebase.

        Returns:
            List of file paths.
        """
        all_files = []

        # Get files for each extension in the config
        for ext in self.config.file_extensions:
            all_files.extend(self.codebase_path.glob(f"**/*{ext}"))

        return all_files

    def update_from_feedback(self, file_path: Path, was_useful: bool) -> None:
        """Update model based on user feedback.

        Args:
            file_path: Path to the file.
            was_useful: Whether the file was useful for analysis.
        """
        if file_path not in self.file_history:
            self.file_history[file_path] = FileAnalysisHistory(file_path=file_path)

        self.file_history[file_path].times_selected += 1
        if was_useful:
            self.file_history[file_path].times_useful += 1

    def collect_file_metrics(self) -> None:
        """Collect metrics for all files in the codebase."""
        import logging

        # Get all files
        all_files = self.get_all_files()
        logging.info(f"Collecting metrics for {len(all_files)} files...")

        # Process each file
        for file_path in all_files:
            try:
                # Skip files that are not relevant to the user's intent
                if not self.user_intent.is_file_relevant(file_path):
                    continue

                # Get file size
                size_bytes = file_path.stat().st_size

                # Count lines
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                except UnicodeDecodeError:
                    # Try with a different encoding
                    try:
                        with open(file_path, "r", encoding="latin-1") as f:
                            line_count = sum(1 for _ in f)
                    except Exception:
                        # Skip binary files
                        logging.warning(f"Skipping binary file: {file_path}")
                        continue

                # Estimate complexity based on file size
                if line_count > self.config.very_large_file_threshold:
                    complexity = FileComplexity.VERY_HIGH
                elif line_count > self.config.large_file_threshold:
                    complexity = FileComplexity.HIGH
                elif line_count > self.config.large_file_threshold / 2.5:
                    complexity = FileComplexity.MEDIUM
                elif line_count > self.config.large_file_threshold / 10:
                    complexity = FileComplexity.LOW
                else:
                    complexity = FileComplexity.TRIVIAL

                # Create metrics
                self.file_metrics[file_path] = FileAnalysisMetrics(
                    file_path=file_path,
                    size_bytes=size_bytes,
                    line_count=line_count,
                    complexity=complexity
                )
            except Exception as e:
                logging.error(f"Error collecting metrics for {file_path}: {e}")

        logging.info(f"Collected metrics for {len(self.file_metrics)} files.")

    def analyze_selected_files(self, selected_files: List[FileSelectionResult]) -> Dict[Path, Dict[str, Any]]:
        """Analyze selected files.

        Args:
            selected_files: List of FileSelectionResult objects.

        Returns:
            Dictionary mapping file paths to analysis results.
        """
        # This is a placeholder implementation
        # In a real implementation, we would use the SemanticMatrixBuilder
        results = {}

        for selection in selected_files:
            file_path = selection.file_path

            # Mark file as selected in history
            if file_path not in self.file_history:
                self.file_history[file_path] = FileAnalysisHistory(file_path=file_path)

            self.file_history[file_path].times_selected += 1

            # Placeholder analysis result
            results[file_path] = selection.to_dict()

        return results

    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save the current configuration to a file.

        Args:
            config_path: Path to save the configuration file.

        Raises:
            IOError: If the configuration file cannot be written.
        """
        self.config.to_json(config_path)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is invalid.
        """
        self.config = AgentConfig.from_json(config_path)
        self.selection_threshold = self.config.selection_threshold
