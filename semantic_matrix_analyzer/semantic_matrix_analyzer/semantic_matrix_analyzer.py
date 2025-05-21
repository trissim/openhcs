#!/usr/bin/env python3
"""
Semantic Matrix Analyzer

This script analyzes Python codebases to create semantically dense matrices
that correlate intent with AST correctness. It can be used to identify
refactoring opportunities and assess code quality.

Usage:
    python semantic_matrix_analyzer.py analyze --config config.json
    python semantic_matrix_analyzer.py analyze --components comp1,comp2 --intents intent1,intent2
    python semantic_matrix_analyzer.py analyze --project-dir /path/to/project
"""

import ast
import os
import sys
import json
import logging
import argparse
import importlib.util
import inspect
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("semantic_matrix_analyzer")


@dataclass
class ComponentAnalysis:
    """Analysis results for a component."""
    name: str
    file_path: Path
    ast_node: Optional[ast.AST] = None
    source_code: str = ""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    intent_alignments: Dict[str, float] = field(default_factory=dict)  # intent_name -> alignment_score (0.0 to 1.0)


@dataclass
class SemanticMatrix:
    """Semantic matrix correlating intent with AST correctness."""
    components: List[str]
    intents: List[str]
    matrix: np.ndarray  # 2D matrix of shape (len(components), len(intents))
    component_analyses: Dict[str, ComponentAnalysis] = field(default_factory=dict)


@dataclass
class IntentPattern:
    """Pattern for detecting intent in code."""
    name: str  # Name of the pattern
    description: str  # Description of what the pattern detects
    pattern_type: str  # Type of pattern: "string", "regex", "ast", "annotation"
    pattern: Any  # The actual pattern (string, regex, AST node type, etc.)
    weight: float = 1.0  # Weight of this pattern in the overall intent score (0.0 to 1.0)
    is_negative: bool = False  # If True, presence of this pattern reduces the intent score


@dataclass
class Intent:
    """Definition of an intent to detect in code."""
    name: str  # Name of the intent
    description: str  # Description of the intent
    patterns: List[IntentPattern] = field(default_factory=list)  # Patterns to detect this intent

    def add_pattern(self, pattern: IntentPattern) -> None:
        """Add a pattern to this intent."""
        self.patterns.append(pattern)

    def add_string_pattern(self, name: str, description: str, pattern: str,
                          weight: float = 1.0, is_negative: bool = False) -> None:
        """Add a string pattern to this intent."""
        self.patterns.append(IntentPattern(
            name=name,
            description=description,
            pattern_type="string",
            pattern=pattern,
            weight=weight,
            is_negative=is_negative
        ))

    def add_regex_pattern(self, name: str, description: str, pattern: str,
                         weight: float = 1.0, is_negative: bool = False) -> None:
        """Add a regex pattern to this intent."""
        import re
        self.patterns.append(IntentPattern(
            name=name,
            description=description,
            pattern_type="regex",
            pattern=re.compile(pattern),
            weight=weight,
            is_negative=is_negative
        ))

    def add_ast_pattern(self, name: str, description: str, node_type: type,
                       condition: Callable[[ast.AST], bool] = None,
                       weight: float = 1.0, is_negative: bool = False) -> None:
        """Add an AST pattern to this intent."""
        self.patterns.append(IntentPattern(
            name=name,
            description=description,
            pattern_type="ast",
            pattern=(node_type, condition),
            weight=weight,
            is_negative=is_negative
        ))

    def add_annotation_pattern(self, name: str, description: str, annotation: str,
                              weight: float = 1.0, is_negative: bool = False) -> None:
        """Add an annotation pattern to this intent."""
        self.patterns.append(IntentPattern(
            name=name,
            description=description,
            pattern_type="annotation",
            pattern=annotation,
            weight=weight,
            is_negative=is_negative
        ))


class IntentRegistry:
    """Registry of intents to detect in code."""

    def __init__(self):
        """Initialize the registry."""
        self.intents: Dict[str, Intent] = {}

    def register_intent(self, intent: Intent) -> None:
        """Register an intent."""
        self.intents[intent.name] = intent

    def get_intent(self, name: str) -> Optional[Intent]:
        """Get an intent by name."""
        return self.intents.get(name)

    def get_all_intents(self) -> List[Intent]:
        """Get all registered intents."""
        return list(self.intents.values())

    def get_intent_names(self) -> List[str]:
        """Get all registered intent names."""
        return list(self.intents.keys())


class IntentDetector:
    """Detects intents in code."""

    def __init__(self, registry: IntentRegistry):
        """Initialize the detector with an intent registry."""
        self.registry = registry

    def detect_intents(self, analysis: ComponentAnalysis) -> Dict[str, float]:
        """Detect intents in a component analysis."""
        intent_scores = {}

        for intent_name, intent in self.registry.intents.items():
            score = self._calculate_intent_score(intent, analysis)
            intent_scores[intent_name] = score

        return intent_scores

    def _calculate_intent_score(self, intent: Intent, analysis: ComponentAnalysis) -> float:
        """Calculate the score for an intent in a component analysis."""
        if not intent.patterns:
            return 0.0

        total_weight = sum(pattern.weight for pattern in intent.patterns)
        if total_weight == 0.0:
            return 0.0

        weighted_score = 0.0

        for pattern in intent.patterns:
            pattern_score = self._check_pattern(pattern, analysis)
            if pattern.is_negative:
                # Negative patterns reduce the score
                weighted_score -= pattern.weight * pattern_score
            else:
                weighted_score += pattern.weight * pattern_score

        # Normalize to 0.0-1.0 range
        normalized_score = max(0.0, min(1.0, weighted_score / total_weight))

        return normalized_score

    def _check_pattern(self, pattern: IntentPattern, analysis: ComponentAnalysis) -> float:
        """Check if a pattern is present in a component analysis."""
        if pattern.pattern_type == "string":
            return self._check_string_pattern(pattern.pattern, analysis)
        elif pattern.pattern_type == "regex":
            return self._check_regex_pattern(pattern.pattern, analysis)
        elif pattern.pattern_type == "ast":
            return self._check_ast_pattern(pattern.pattern, analysis)
        elif pattern.pattern_type == "annotation":
            return self._check_annotation_pattern(pattern.pattern, analysis)
        else:
            logger.warning(f"Unknown pattern type: {pattern.pattern_type}")
            return 0.0

    def _check_string_pattern(self, pattern: str, analysis: ComponentAnalysis) -> float:
        """Check if a string pattern is present in a component analysis."""
        # Check in source code
        if pattern.lower() in analysis.source_code.lower():
            return 1.0

        # Check in docstrings
        if analysis.ast_node:
            docstring = ast.get_docstring(analysis.ast_node)
            if docstring and pattern.lower() in docstring.lower():
                return 1.0

        return 0.0

    def _check_regex_pattern(self, pattern: Any, analysis: ComponentAnalysis) -> float:
        """Check if a regex pattern is present in a component analysis."""
        # Check in source code
        if pattern.search(analysis.source_code):
            return 1.0

        # Check in docstrings
        if analysis.ast_node:
            docstring = ast.get_docstring(analysis.ast_node)
            if docstring and pattern.search(docstring):
                return 1.0

        return 0.0

    def _check_ast_pattern(self, pattern: Any, analysis: ComponentAnalysis) -> float:
        """Check if an AST pattern is present in a component analysis."""
        if not analysis.ast_node:
            return 0.0

        node_type, condition = pattern

        # Count matching nodes
        matching_nodes = 0
        total_nodes = 0

        for node in ast.walk(analysis.ast_node):
            if isinstance(node, node_type):
                total_nodes += 1
                if condition is None or condition(node):
                    matching_nodes += 1

        # Return ratio of matching nodes
        if total_nodes == 0:
            return 0.0

        return matching_nodes / total_nodes

    def _check_annotation_pattern(self, pattern: str, analysis: ComponentAnalysis) -> float:
        """Check if an annotation pattern is present in a component analysis."""
        # Look for special annotations in comments or docstrings
        annotation_marker = f"@intent:{pattern}"

        # Check in source code
        if annotation_marker in analysis.source_code:
            return 1.0

        # Check in docstrings
        if analysis.ast_node:
            docstring = ast.get_docstring(analysis.ast_node)
            if docstring and annotation_marker in docstring:
                return 1.0

        return 0.0


class IntentExtractor:
    """Extracts intent from docstrings and comments."""

    @staticmethod
    def extract_intent(node: ast.AST) -> List[str]:
        """Extract intent from a node's docstring and comments."""
        intents = []

        # Extract docstring
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node):
            docstring = ast.get_docstring(node)
            if docstring:
                # Look for intent annotations
                for line in docstring.split('\n'):
                    line = line.strip()
                    if line.startswith('@intent:'):
                        intent = line[8:].strip()
                        intents.append(intent)
                    elif line and not line.startswith('@') and not line.startswith(':'):
                        # Also include regular docstring lines as potential intents
                        intents.append(line)

        return intents


class ASTAnalyzer:
    """Analyzer for AST nodes."""

    @staticmethod
    def analyze_function(node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        result = {
            "name": node.name,
            "args": len(node.args.args),
            "returns": node.returns is not None,
            "body_size": len(node.body),
            "has_docstring": ast.get_docstring(node) is not None,
            "calls": [],
            "attributes_accessed": [],
            "attributes_modified": [],
            "return_values": [],
        }

        # Analyze function body
        for item in ast.walk(node):
            # Find function calls
            if isinstance(item, ast.Call):
                if isinstance(item.func, ast.Name):
                    result["calls"].append(item.func.id)
                elif isinstance(item.func, ast.Attribute):
                    if isinstance(item.func.value, ast.Name):
                        result["calls"].append(f"{item.func.value.id}.{item.func.attr}")

            # Find attribute access
            if isinstance(item, ast.Attribute):
                if isinstance(item.value, ast.Name):
                    attr = f"{item.value.id}.{item.attr}"
                    # Check if it's being modified
                    if isinstance(item.ctx, ast.Store):
                        result["attributes_modified"].append(attr)
                    else:
                        result["attributes_accessed"].append(attr)

            # Find return values
            if isinstance(item, ast.Return) and item.value:
                if isinstance(item.value, ast.Name):
                    result["return_values"].append(item.value.id)
                elif isinstance(item.value, ast.Constant):
                    result["return_values"].append(str(item.value.value))

        return result


class DependencyAnalyzer:
    """Analyzer for dependencies between components."""

    @staticmethod
    def analyze_imports(node: ast.Module) -> List[str]:
        """Analyze imports in a module."""
        imports = []

        for item in node.body:
            if isinstance(item, ast.Import):
                for name in item.names:
                    imports.append(name.name)
            elif isinstance(item, ast.ImportFrom):
                if item.module:
                    for name in item.names:
                        imports.append(f"{item.module}.{name.name}")

        return imports


class SemanticMatrixBuilder:
    """Builder for semantic matrices."""

    def __init__(self, components: List[str], intents: List[str], project_dir: str = ".",
                components_config: Dict[str, Optional[str]] = None, intent_registry: IntentRegistry = None):
        """Initialize the builder.

        Args:
            components: List of component names to analyze
            intents: List of intents to analyze
            project_dir: Root directory of the project
            components_config: Dictionary mapping component names to file paths (relative to project_dir)
                              If a value is None, the builder will try to infer the file path
            intent_registry: Registry of intents to detect in code
        """
        self.components = components
        self.intents = intents
        self.project_dir = Path(project_dir)
        self.components_config = components_config or {}
        self.intent_registry = intent_registry or create_intent_registry_from_config(intents)
        self.intent_detector = IntentDetector(self.intent_registry)
        self.matrix = np.zeros((len(components), len(intents)))
        self.component_analyses = {}

    def analyze_component(self, component: str, file_path: Path) -> ComponentAnalysis:
        """Analyze a component."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Create component analysis
            analysis = ComponentAnalysis(
                name=component,
                file_path=file_path,
                ast_node=tree,
                source_code=source_code
            )

            # Extract dependencies
            analysis.dependencies = DependencyAnalyzer.analyze_imports(tree)

            # Find the component in the AST
            component_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == component:
                    component_node = node
                    break
                elif isinstance(node, ast.FunctionDef) and node.name == component:
                    component_node = node
                    break

            if component_node:
                # Analyze component
                if isinstance(component_node, ast.ClassDef):
                    # Analyze class
                    analysis.metrics["methods"] = len([n for n in component_node.body if isinstance(n, ast.FunctionDef)])
                    analysis.metrics["attributes"] = len([n for n in component_node.body if isinstance(n, ast.Assign)])

                    # Analyze methods
                    for method in [n for n in component_node.body if isinstance(n, ast.FunctionDef)]:
                        method_analysis = ASTAnalyzer.analyze_function(method)

                        # Check for context mutation
                        if "context" in [arg.arg for arg in method.args.args]:
                            context_modified = any(attr.startswith("context.") for attr in method_analysis["attributes_modified"])
                            if context_modified:
                                analysis.issues.append({
                                    "type": "context_mutation",
                                    "method": method.name,
                                    "message": f"Method '{method.name}' mutates the context object"
                                })

                        # Check for StepResult usage
                        if "StepResult" in method_analysis["calls"] or any("StepResult" in call for call in method_analysis["calls"]):
                            analysis.issues.append({
                                "type": "step_result_usage",
                                "method": method.name,
                                "message": f"Method '{method.name}' uses StepResult"
                            })
                elif isinstance(component_node, ast.FunctionDef):
                    # Analyze function
                    function_analysis = ASTAnalyzer.analyze_function(component_node)

                    # Check for context mutation
                    if "context" in [arg.arg for arg in component_node.args.args]:
                        context_modified = any(attr.startswith("context.") for attr in function_analysis["attributes_modified"])
                        if context_modified:
                            analysis.issues.append({
                                "type": "context_mutation",
                                "function": component_node.name,
                                "message": f"Function '{component_node.name}' mutates the context object"
                            })

                    # Check for StepResult usage
                    if "StepResult" in function_analysis["calls"] or any("StepResult" in call for call in function_analysis["calls"]):
                        analysis.issues.append({
                            "type": "step_result_usage",
                            "function": component_node.name,
                            "message": f"Function '{component_node.name}' uses StepResult"
                        })

            # Detect intents using the intent detector
            analysis.intent_alignments = self.intent_detector.detect_intents(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing component {component}: {e}")
            return ComponentAnalysis(name=component, file_path=file_path)

    def build_matrix(self) -> SemanticMatrix:
        """Build the semantic matrix."""
        for i, component in enumerate(self.components):
            # Determine file path based on component name
            file_path = self.get_file_path_for_component(component)

            if file_path and file_path.exists():
                analysis = self.analyze_component(component, file_path)
                self.component_analyses[component] = analysis

                # Update matrix based on intent alignments
                for j, intent in enumerate(self.intents):
                    self.matrix[i, j] = analysis.intent_alignments.get(intent, 0.0)
            else:
                logger.warning(f"Could not find file for component {component}")
                # Create a placeholder analysis
                self.component_analyses[component] = ComponentAnalysis(
                    name=component,
                    file_path=Path(f"unknown_path_for_{component}.py")
                )

        return SemanticMatrix(
            components=self.components,
            intents=self.intents,
            matrix=self.matrix,
            component_analyses=self.component_analyses
        )

    def get_file_path_for_component(self, component: str) -> Optional[Path]:
        """Get the file path for a component.

        First checks if the component has a path specified in components_config.
        If not, tries to infer the path based on common patterns.
        """
        # Check if the component has a path specified in components_config
        if component in self.components_config and self.components_config[component]:
            return self.project_dir / self.components_config[component]

        # Try to find the file by searching for the component name
        for ext in [".py", ".pyx", ".pyi"]:
            # Try snake_case version of the component name
            snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in component]).lstrip('_')
            potential_paths = list(self.project_dir.glob(f"**/{snake_case}{ext}"))
            if potential_paths:
                return potential_paths[0]

            # Try lowercase version of the component name
            potential_paths = list(self.project_dir.glob(f"**/{component.lower()}{ext}"))
            if potential_paths:
                return potential_paths[0]

            # Try exact case of the component name
            potential_paths = list(self.project_dir.glob(f"**/{component}{ext}"))
            if potential_paths:
                return potential_paths[0]

        # If all else fails, return None
        return None


def visualize_matrix(matrix: SemanticMatrix, output_file: Optional[str] = None) -> None:
    """Visualize the semantic matrix."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(matrix.matrix, cmap="YlGn")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Intent Alignment Score", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(matrix.intents)))
    ax.set_yticks(np.arange(len(matrix.components)))
    ax.set_xticklabels(matrix.intents)
    ax.set_yticklabels(matrix.components)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add title and labels
    ax.set_title("Semantic Matrix: Component-Intent Alignment")

    # Add text annotations
    for i in range(len(matrix.components)):
        for j in range(len(matrix.intents)):
            text = ax.text(j, i, f"{matrix.matrix[i, j]:.2f}",
                          ha="center", va="center", color="black" if matrix.matrix[i, j] > 0.5 else "white")

    fig.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def generate_report(matrix: SemanticMatrix) -> str:
    """Generate a report from the semantic matrix."""
    report = []
    report.append("# Semantic Matrix Analysis Report")
    report.append("")

    # Overall alignment
    overall_alignment = np.mean(matrix.matrix)
    report.append(f"## Overall Intent Alignment: {overall_alignment:.2f}")
    report.append("")

    # Component analysis
    report.append("## Component Analysis")
    report.append("")

    for component, analysis in matrix.component_analyses.items():
        report.append(f"### {component}")
        report.append(f"- File: {analysis.file_path}")

        # Intent alignments
        if analysis.intent_alignments:
            report.append("- Intent Alignments:")
            for intent, score in sorted(analysis.intent_alignments.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - {intent}: {score:.2f}")

        if analysis.metrics:
            report.append("- Metrics:")
            for metric, value in analysis.metrics.items():
                report.append(f"  - {metric}: {value}")

        if analysis.dependencies:
            report.append("- Dependencies:")
            for dependency in analysis.dependencies:
                report.append(f"  - {dependency}")

        if analysis.issues:
            report.append("- Issues:")
            for issue in analysis.issues:
                report.append(f"  - {issue['message']}")

        report.append("")

    # Intent analysis
    report.append("## Intent Analysis")
    report.append("")

    for i, intent in enumerate(matrix.intents):
        intent_alignment = np.mean(matrix.matrix[:, i])
        report.append(f"### {intent}")
        report.append(f"- Overall Alignment: {intent_alignment:.2f}")

        # Components with high alignment
        high_alignment = [(matrix.components[j], matrix.matrix[j, i]) for j in range(len(matrix.components)) if matrix.matrix[j, i] > 0.7]
        if high_alignment:
            report.append("- Components with High Alignment:")
            for component, score in sorted(high_alignment, key=lambda x: x[1], reverse=True):
                report.append(f"  - {component}: {score:.2f}")

        # Components with low alignment
        low_alignment = [(matrix.components[j], matrix.matrix[j, i]) for j in range(len(matrix.components)) if matrix.matrix[j, i] < 0.3 and matrix.matrix[j, i] > 0]
        if low_alignment:
            report.append("- Components with Low Alignment:")
            for component, score in sorted(low_alignment, key=lambda x: x[1]):
                report.append(f"  - {component}: {score:.2f}")

        # Components with no alignment
        no_alignment = [(matrix.components[j], 0.0) for j in range(len(matrix.components)) if matrix.matrix[j, i] == 0]
        if no_alignment:
            report.append("- Components with No Alignment:")
            for component, _ in no_alignment:
                report.append(f"  - {component}")

        report.append("")

    # Pattern matching details
    report.append("## Pattern Matching Details")
    report.append("")
    report.append("This analysis used the following patterns to detect intents:")
    report.append("")

    # Get the intent registry from the first component analysis
    if matrix.component_analyses:
        first_component = next(iter(matrix.component_analyses.values()))
        if hasattr(first_component, 'intent_detector') and first_component.intent_detector:
            intent_registry = first_component.intent_detector.registry
            for intent_name in matrix.intents:
                intent = intent_registry.get_intent(intent_name)
                if intent:
                    report.append(f"### {intent_name}")
                    report.append(f"- Description: {intent.description}")
                    report.append("- Patterns:")
                    for pattern in intent.patterns:
                        pattern_desc = f"{pattern.name}: {pattern.description}"
                        if pattern.is_negative:
                            pattern_desc += " (negative pattern)"
                        report.append(f"  - {pattern_desc}")
                    report.append("")

    return "\n".join(report)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Semantic Matrix Analyzer for Python codebases")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a Python codebase")
    analyze_parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    analyze_parser.add_argument("--project-dir", type=str, default=".", help="Path to project directory")
    analyze_parser.add_argument("--components", type=str, help="Comma-separated list of components to analyze")
    analyze_parser.add_argument("--intents", type=str, help="Comma-separated list of intents to analyze")
    analyze_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    analyze_parser.add_argument("--output-prefix", type=str, default="semantic_matrix", help="Prefix for output files")
    analyze_parser.add_argument("--format", type=str, choices=["png", "svg", "pdf"], default="png", help="Output format for visualization")

    # Generate config command
    config_parser = subparsers.add_parser("generate-config", help="Generate a sample configuration file")
    config_parser.add_argument("--output", type=str, default="semantic_matrix_config.json", help="Output file path")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)


def generate_sample_config(output_path: str) -> None:
    """Generate a sample configuration file."""
    # Get all available intents from the plugins
    plugin_registry = load_intent_plugins()
    available_intents = [intent.name for intent in plugin_registry.get_all_intents()]

    sample_config = {
        "project_dir": ".",
        "components": {
            "Component1": "path/to/component1.py",
            "Component2": "path/to/component2.py",
            "Component3": "path/to/component3.py"
        },
        "intents": available_intents,
        "output": {
            "directory": "output",
            "prefix": "semantic_matrix",
            "format": "png"
        }
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_config, f, indent=2)
        logger.info(f"Sample configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating sample configuration: {e}")
        sys.exit(1)


class IntentPlugin:
    """Base class for intent plugins."""

    @staticmethod
    def get_intents() -> List[Intent]:
        """Get the intents defined by this plugin."""
        return []


class GenericIntentPlugin(IntentPlugin):
    """Generic intent plugin with common software engineering intents."""

    @staticmethod
    def get_intents() -> List[Intent]:
        """Get generic software engineering intents."""
        intents = []

        # Immutability intent
        immutability = Intent(
            name="Immutability",
            description="Using immutable data structures and patterns"
        )
        immutability.add_string_pattern(
            name="immutable_pattern",
            description="Reference to immutability",
            pattern="immutable",
            weight=1.0
        )
        immutability.add_string_pattern(
            name="frozen_pattern",
            description="Using frozen dataclasses",
            pattern="frozen",
            weight=0.8
        )
        immutability.add_regex_pattern(
            name="frozen_dataclass_pattern",
            description="Defining frozen dataclasses",
            pattern=r"@dataclass\s*\(\s*frozen\s*=\s*True\s*\)",
            weight=1.0
        )
        immutability.add_string_pattern(
            name="readonly_pattern",
            description="Using readonly properties",
            pattern="readonly",
            weight=0.7
        )
        immutability.add_string_pattern(
            name="const_pattern",
            description="Using const variables",
            pattern="const ",
            weight=0.7
        )
        immutability.add_regex_pattern(
            name="final_pattern",
            description="Using final variables",
            pattern=r"final\s+[a-zA-Z_][a-zA-Z0-9_]*",
            weight=0.7
        )
        intents.append(immutability)

        # Dependency Injection intent
        di = Intent(
            name="Dependency Injection",
            description="Using dependency injection patterns"
        )
        di.add_string_pattern(
            name="inject_pattern",
            description="Reference to injection",
            pattern="inject",
            weight=1.0
        )
        di.add_string_pattern(
            name="dependency_injection_pattern",
            description="Reference to dependency injection",
            pattern="dependency injection",
            weight=1.0
        )
        di.add_string_pattern(
            name="di_pattern",
            description="Reference to DI",
            pattern="DI",
            weight=0.5
        )
        di.add_regex_pattern(
            name="constructor_injection_pattern",
            description="Constructor injection pattern",
            pattern=r"def\s+__init__\s*\(\s*self\s*,\s*[^)]*\)\s*:",
            weight=0.5
        )
        intents.append(di)

        # Factory Pattern intent
        factory = Intent(
            name="Factory Pattern",
            description="Using factory patterns for object creation"
        )
        factory.add_string_pattern(
            name="factory_pattern",
            description="Reference to factory pattern",
            pattern="factory",
            weight=1.0
        )
        factory.add_string_pattern(
            name="create_pattern",
            description="Methods for creating objects",
            pattern="create_",
            weight=0.7
        )
        factory.add_regex_pattern(
            name="factory_method_pattern",
            description="Factory method pattern",
            pattern=r"@classmethod\s+def\s+create",
            weight=0.8
        )
        intents.append(factory)

        # Singleton Pattern intent
        singleton = Intent(
            name="Singleton Pattern",
            description="Using singleton pattern for single instances"
        )
        singleton.add_string_pattern(
            name="singleton_pattern",
            description="Reference to singleton pattern",
            pattern="singleton",
            weight=1.0
        )
        singleton.add_regex_pattern(
            name="instance_check_pattern",
            description="Checking for existing instance",
            pattern=r"if\s+cls\._instance\s+is\s+None",
            weight=0.8
        )
        singleton.add_regex_pattern(
            name="private_constructor_pattern",
            description="Private constructor pattern",
            pattern=r"_[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*None",
            weight=0.6
        )
        intents.append(singleton)

        # Observer Pattern intent
        observer = Intent(
            name="Observer Pattern",
            description="Using observer pattern for event handling"
        )
        observer.add_string_pattern(
            name="observer_pattern",
            description="Reference to observer pattern",
            pattern="observer",
            weight=1.0
        )
        observer.add_string_pattern(
            name="subscribe_pattern",
            description="Methods for subscribing to events",
            pattern="subscribe",
            weight=0.8
        )
        observer.add_string_pattern(
            name="notify_pattern",
            description="Methods for notifying observers",
            pattern="notify",
            weight=0.8
        )
        observer.add_string_pattern(
            name="listener_pattern",
            description="Reference to listeners",
            pattern="listener",
            weight=0.7
        )
        observer.add_string_pattern(
            name="event_pattern",
            description="Reference to events",
            pattern="event",
            weight=0.6
        )
        intents.append(observer)

        # Strategy Pattern intent
        strategy = Intent(
            name="Strategy Pattern",
            description="Using strategy pattern for algorithm selection"
        )
        strategy.add_string_pattern(
            name="strategy_pattern",
            description="Reference to strategy pattern",
            pattern="strategy",
            weight=1.0
        )
        strategy.add_regex_pattern(
            name="abstract_method_pattern",
            description="Abstract method pattern",
            pattern=r"@abstractmethod",
            weight=0.7
        )
        intents.append(strategy)

        return intents


class PluginRegistry:
    """Registry for intent plugins."""

    def __init__(self):
        """Initialize the registry."""
        self.plugins: List[IntentPlugin] = []

    def register_plugin(self, plugin: IntentPlugin) -> None:
        """Register a plugin."""
        self.plugins.append(plugin)

    def get_all_intents(self) -> List[Intent]:
        """Get all intents from all registered plugins."""
        intents = []
        for plugin in self.plugins:
            intents.extend(plugin.get_intents())
        return intents


def load_intent_plugins() -> PluginRegistry:
    """Load all available intent plugins."""
    registry = PluginRegistry()

    # Register built-in plugins
    registry.register_plugin(GenericIntentPlugin())

    # Load custom plugins from plugins directory if it exists
    plugins_dir = Path("plugins")
    if plugins_dir.exists() and plugins_dir.is_dir():
        for plugin_file in plugins_dir.glob("*_plugin.py"):
            try:
                # Import the plugin module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find plugin classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, IntentPlugin) and
                            obj is not IntentPlugin):
                            registry.register_plugin(obj())
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file}: {e}")

    return registry


def create_intent_registry_from_names(intent_names: List[str]) -> IntentRegistry:
    """Create an intent registry from a list of intent names."""
    # Load all available plugins
    plugin_registry = load_intent_plugins()

    # Get all available intents
    all_intents = {intent.name: intent for intent in plugin_registry.get_all_intents()}

    # Create a new registry with only the requested intents
    registry = IntentRegistry()

    for intent_name in intent_names:
        if intent_name in all_intents:
            registry.register_intent(all_intents[intent_name])
        else:
            # Create a simple intent with no patterns
            registry.register_intent(Intent(
                name=intent_name,
                description=f"Custom intent: {intent_name}"
            ))

    return registry


def create_intent_registry_from_config(intents: List[str]) -> IntentRegistry:
    """Create an intent registry from a list of intent names in a config file."""
    return create_intent_registry_from_names(intents)


def main():
    """Main entry point for the script."""
    args = parse_args()

    if args.command == "generate-config":
        generate_sample_config(args.output)
        return

    if args.command == "analyze":
        # Load configuration
        if args.config:
            config = load_config(args.config)
            project_dir = config.get("project_dir", ".")
            components_config = config.get("components", {})
            intents = config.get("intents", [])
            output_dir = config.get("output", {}).get("directory", ".")
            output_prefix = config.get("output", {}).get("prefix", "semantic_matrix")
            output_format = config.get("output", {}).get("format", "png")
        else:
            project_dir = args.project_dir
            components_config = {}

            if args.components:
                components = args.components.split(",")
                # Create a dictionary with component names as keys and None as values
                # The SemanticMatrixBuilder will try to infer the file paths
                components_config = {comp.strip(): None for comp in components}
            else:
                # Default to analyzing all Python files in the project
                components_config = {}
                for py_file in Path(project_dir).glob("**/*.py"):
                    if py_file.is_file():
                        # Use the filename without extension as the component name
                        component_name = py_file.stem
                        # Use the relative path from the project directory
                        rel_path = py_file.relative_to(project_dir)
                        components_config[component_name] = str(rel_path)

            if args.intents:
                intents = [intent.strip() for intent in args.intents.split(",")]
            else:
                # Get all available intents from the plugins
                plugin_registry = load_intent_plugins()
                intents = [intent.name for intent in plugin_registry.get_all_intents()]

            output_dir = args.output_dir
            output_prefix = args.output_prefix
            output_format = args.format

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract component names
        components = list(components_config.keys())

        # Create intent registry
        intent_registry = create_intent_registry_from_config(intents)

        # Build semantic matrix
        builder = SemanticMatrixBuilder(components, intents, project_dir, components_config, intent_registry)
        matrix = builder.build_matrix()

        # Visualize matrix
        visualization_path = os.path.join(output_dir, f"{output_prefix}.{output_format}")
        visualize_matrix(matrix, visualization_path)

        # Generate report
        report = generate_report(matrix)

        # Write report to file
        report_path = os.path.join(output_dir, f"{output_prefix}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Generate JSON data for further processing
        json_data = {
            "components": matrix.components,
            "intents": matrix.intents,
            "matrix": matrix.matrix.tolist(),
            "component_analyses": {
                name: {
                    "name": analysis.name,
                    "file_path": str(analysis.file_path),
                    "issues": analysis.issues,
                    "metrics": analysis.metrics,
                    "dependencies": analysis.dependencies,
                    "intent_alignments": analysis.intent_alignments
                }
                for name, analysis in matrix.component_analyses.items()
            }
        }

        json_path = os.path.join(output_dir, f"{output_prefix}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Semantic matrix visualization saved to {visualization_path}")
        logger.info(f"Semantic matrix report saved to {report_path}")
        logger.info(f"Semantic matrix data saved to {json_path}")
    else:
        logger.error("No command specified. Use 'analyze' or 'generate-config'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
