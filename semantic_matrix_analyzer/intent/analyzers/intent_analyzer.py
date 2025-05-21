"""
Main analyzer for extracting intent from code structure.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentHierarchy
)
from semantic_matrix_analyzer.intent.config.configuration import Configuration
from semantic_matrix_analyzer.intent.analyzers.name_analyzer import ConfigurableNameAnalyzer
from semantic_matrix_analyzer.intent.analyzers.type_analyzer import TypeHintExtractor, ConfigurableTypeHintAnalyzer
from semantic_matrix_analyzer.intent.integration.combiner import ConfigurableIntentCombiner
from semantic_matrix_analyzer.intent.integration.hierarchy import ConfigurableHierarchyBuilder
from semantic_matrix_analyzer.intent.integration.reporter import ConfigurableIntentReporter

logger = logging.getLogger(__name__)


class ConfigurableIntentAnalyzer:
    """Configurable analyzer for extracting intent from code structure."""

    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable intent analyzer.

        Args:
            config: The configuration to use (optional).
        """
        self.config = config or Configuration()
        self.name_analyzer = ConfigurableNameAnalyzer(self.config)
        self.type_analyzer = ConfigurableTypeHintAnalyzer(self.config)
        self.intent_combiner = ConfigurableIntentCombiner(self.config)
        self.hierarchy_builder = ConfigurableHierarchyBuilder(self.config)
        self.intent_reporter = ConfigurableIntentReporter(self.config)

    def analyze_codebase(self, file_paths: List[Path], dependency_graph=None) -> Dict[str, Any]:
        """Analyze a codebase to extract intent.

        Args:
            file_paths: A list of file paths to analyze.
            dependency_graph: A dependency graph (optional).

        Returns:
            A report of the analysis.
        """
        # Analyze names
        name_intents = self.analyze_names(file_paths)

        # Analyze type hints
        type_intents = self.analyze_type_hints(file_paths)

        # Analyze structure
        structural_intents = []
        if dependency_graph:
            structural_analyzer = ConfigurableStructuralAnalyzer(dependency_graph, self.config)
            structural_intents = structural_analyzer.analyze_dependency_graph()

        # Combine intents
        combined_intents = self.intent_combiner.combine_intents([name_intents, type_intents, structural_intents])

        # Build hierarchy
        intent_hierarchy = self.hierarchy_builder.build_hierarchy(combined_intents)

        # Generate report
        report = self.intent_reporter.generate_report(intent_hierarchy)

        return report

    def analyze_names(self, file_paths: List[Path]) -> List[NameIntent]:
        """Analyze names in the codebase.

        Args:
            file_paths: A list of file paths.

        Returns:
            A list of NameIntent objects.
        """
        name_intents = []

        for file_path in file_paths:
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            tree = ast.parse(code, filename=str(file_path))

            # Analyze class names
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    intent = self.name_analyzer.analyze_class_name(
                        node.name,
                        file_path,
                        node.lineno
                    )
                    name_intents.append(intent)

                elif isinstance(node, ast.FunctionDef):
                    # Check if this is a method
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef) and node in parent.body:
                            is_method = True
                            break

                    if is_method:
                        intent = self.name_analyzer.analyze_method_name(
                            node.name,
                            file_path,
                            node.lineno
                        )
                    else:
                        intent = self.name_analyzer.analyze_name(
                            node.name,
                            "function",
                            file_path,
                            node.lineno
                        )
                    name_intents.append(intent)

                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            intent = self.name_analyzer.analyze_variable_name(
                                target.id,
                                file_path,
                                node.lineno
                            )
                            name_intents.append(intent)

        return name_intents

    def analyze_type_hints(self, file_paths: List[Path]) -> List[TypeIntent]:
        """Analyze type hints in the codebase.

        Args:
            file_paths: A list of file paths.

        Returns:
            A list of TypeIntent objects.
        """
        type_intents = []

        # Create a type hint extractor
        type_hint_extractor = TypeHintExtractor()

        for file_path in file_paths:
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            tree = ast.parse(code, filename=str(file_path))

            # Extract type hints from functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    type_hints = type_hint_extractor.extract_type_hints_from_ast(node)

                    # Analyze parameter type hints
                    for param_name, type_hint in type_hints.items():
                        if param_name != "return":
                            intent = self.type_analyzer.analyze_parameter_type(
                                param_name,
                                type_hint,
                                file_path,
                                node.lineno
                            )
                            type_intents.append(intent)

                    # Analyze return type hint
                    if "return" in type_hints:
                        intent = self.type_analyzer.analyze_return_type(
                            type_hints["return"],
                            file_path,
                            node.lineno
                        )
                        type_intents.append(intent)

        return type_intents

    def format_report(self, report: Dict[str, Any], format: Optional[str] = None) -> str:
        """Format a report in the specified format.

        Args:
            report: The report to format.
            format: The format to use ("text", "markdown", "json").

        Returns:
            The formatted report.
        """
        return self.intent_reporter.format_report(report, format)
