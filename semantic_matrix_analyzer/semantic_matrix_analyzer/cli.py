"""
Command-line interface for Semantic Matrix Analyzer.

This module provides the command-line interface for the analyzer.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from semantic_matrix_analyzer.agent import AgentDrivenAnalyzer, UserIntent
from semantic_matrix_analyzer.agent.config import AgentConfig, generate_default_config
from semantic_matrix_analyzer.core import SemanticMatrixBuilder, intent_registry_global
from semantic_matrix_analyzer.plugins import plugin_manager


def setup_logging(verbose: bool = False) -> None:
    """Set up logging.

    Args:
        verbose: Whether to enable verbose logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Semantic Matrix Analyzer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a codebase")
    analyze_parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    analyze_parser.add_argument("--project-dir", type=str, default=".", help="Path to project directory")
    analyze_parser.add_argument("--components", type=str, help="Comma-separated list of components to analyze")
    analyze_parser.add_argument("--intents", type=str, help="Comma-separated list of intents to analyze")
    analyze_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    analyze_parser.add_argument("--output-prefix", type=str, default="semantic_matrix", help="Prefix for output files")
    analyze_parser.add_argument("--format", type=str, default="png", choices=["png", "svg", "pdf"], help="Output format for visualization")
    analyze_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Agent-driven analysis options
    analyze_parser.add_argument("--agent-driven", action="store_true", help="Use agent-driven analysis")
    analyze_parser.add_argument("--concerns", type=str, help="Comma-separated list of concerns to focus on")
    analyze_parser.add_argument("--threshold", type=float, default=0.5, help="Selection threshold for agent-driven analysis")
    analyze_parser.add_argument("--exclude-files", type=str, help="Comma-separated list of files to exclude")
    analyze_parser.add_argument("--exclude-patterns", type=str, help="Comma-separated list of patterns to exclude")

    # generate-config command
    config_parser = subparsers.add_parser("generate-config", help="Generate a sample configuration file")
    config_parser.add_argument("--output", type=str, default="semantic_matrix_config.json", help="Output file path")
    config_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # list-intents command
    list_intents_parser = subparsers.add_parser("list-intents", help="List available intents")
    list_intents_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # list-plugins command
    list_plugins_parser = subparsers.add_parser("list-plugins", help="List available plugins")
    list_plugins_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # generate-agent-config command
    agent_config_parser = subparsers.add_parser("generate-agent-config", help="Generate a default agent configuration file")
    agent_config_parser.add_argument("--output", type=str, required=True, help="Output file path")
    agent_config_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file is not valid JSON.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sample_config(output_path: str) -> None:
    """Generate a sample configuration file.

    Args:
        output_path: Path to save the configuration file.
    """
    # Get all available intents from the plugins
    plugin_manager.discover_plugins([Path("plugins")])
    plugin_manager.initialize_plugins()

    available_intents = []
    for plugin in plugin_manager.get_intent_plugins():
        for intent in plugin.get_intents():
            available_intents.append(intent.name)

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
        },
        "agent_driven": {
            "enabled": True,
            "concerns": ["error handling", "thread safety", "performance"],
            "threshold": 0.5,
            "exclude_files": ["path/to/exclude1.py", "path/to/exclude2.py"],
            "exclude_patterns": ["pattern1", "pattern2"]
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)

    logging.info(f"Sample configuration saved to {output_path}")


def visualize_matrix(matrix, output_path: str) -> None:
    """Visualize a semantic matrix.

    Args:
        matrix: The semantic matrix to visualize.
        output_path: Path to save the visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(matrix.matrix, cmap="YlGnBu")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Intent Alignment", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(matrix.intents)))
    ax.set_yticks(np.arange(len(matrix.components)))
    ax.set_xticklabels(matrix.intents)
    ax.set_yticklabels(matrix.components)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(matrix.components)):
        for j in range(len(matrix.intents)):
            text = ax.text(j, i, f"{matrix.matrix[i, j]:.2f}",
                          ha="center", va="center", color="black")

    ax.set_title("Semantic Matrix")
    fig.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.close()


def generate_report(matrix, agent_driven: bool = False, selected_files: List = None) -> str:
    """Generate a report from the semantic matrix.

    Args:
        matrix: The semantic matrix.
        agent_driven: Whether agent-driven analysis was used.
        selected_files: List of (file_path, score) tuples for agent-driven analysis.

    Returns:
        The report as a string.
    """
    report = []
    report.append("# Semantic Matrix Analysis Report")
    report.append("")

    # Add agent-driven analysis information if applicable
    if agent_driven and selected_files:
        report.append("## Agent-Driven Analysis")
        report.append("")
        report.append(f"Analysis was performed using agent-driven file selection.")
        report.append(f"Selected {len(selected_files)} files for analysis based on relevance, information value, and effort required.")
        report.append("")
        report.append("### Top Files by Selection Score")
        report.append("")
        report.append("| File | Score | Relevance | Info Value | Effort |")
        report.append("| ---- | ----- | --------- | ---------- | ------ |")

        # Show top 10 files
        for selection in sorted(selected_files, key=lambda x: x.score, reverse=True)[:10]:
            report.append(
                f"| {selection.file_path} | {selection.score:.2f} | "
                f"{selection.relevance:.2f} | {selection.information_value:.2f} | "
                f"{selection.effort:.2f} |"
            )

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

    return "\n".join(report)


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    # Discover and initialize plugins
    plugin_manager.discover_plugins([Path("plugins")])
    plugin_manager.initialize_plugins()

    if args.command == "generate-config":
        generate_sample_config(args.output)
        return

    if args.command == "generate-agent-config":
        generate_default_config(args.output)
        logging.info(f"Default agent configuration saved to {args.output}")
        return

    if args.command == "list-intents":
        print("Available intents:")
        for intent_name in sorted(intent_registry_global.get_intent_names()):
            intent = intent_registry_global.get_intent(intent_name)
            print(f"- {intent_name}: {intent.description}")
        return

    if args.command == "list-plugins":
        print("Available plugins:")
        for plugin in plugin_manager.plugins.values():
            print(f"- {plugin.name} (v{plugin.version}): {plugin.description}")
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

            # Agent-driven options
            agent_driven_config = config.get("agent_driven", {})
            args.agent_driven = agent_driven_config.get("enabled", False)

            if args.agent_driven:
                args.concerns = ",".join(agent_driven_config.get("concerns", []))
                args.threshold = agent_driven_config.get("threshold", 0.5)
                args.exclude_files = ",".join(agent_driven_config.get("exclude_files", []))
                args.exclude_patterns = ",".join(agent_driven_config.get("exclude_patterns", []))
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
                # Get all available intents
                intents = intent_registry_global.get_intent_names()

            output_dir = args.output_dir
            output_prefix = args.output_prefix
            output_format = args.format

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract component names
        components = list(components_config.keys())

        # Check if agent-driven analysis is enabled
        if args.agent_driven:
            # Create user intent
            concerns = []
            if args.concerns:
                concerns = [concern.strip() for concern in args.concerns.split(",")]

            excluded_files = []
            if args.exclude_files:
                excluded_files = [Path(file.strip()) for file in args.exclude_files.split(",")]

            excluded_patterns = []
            if args.exclude_patterns:
                excluded_patterns = [pattern.strip() for pattern in args.exclude_patterns.split(",")]

            user_intent = UserIntent(
                primary_concerns=concerns,
                component_mentions=components,
                excluded_files=excluded_files,
                excluded_patterns=excluded_patterns
            )

            # Create agent-driven analyzer
            if args.config:
                # Load configuration from file
                logging.info(f"Loading agent configuration from {args.config}")
                analyzer = AgentDrivenAnalyzer(
                    codebase_path=project_dir,
                    user_intent=user_intent,
                    config_path=args.config
                )
            else:
                # Use default configuration
                analyzer = AgentDrivenAnalyzer(codebase_path=project_dir, user_intent=user_intent)
                analyzer.selection_threshold = args.threshold

            # Collect file metrics
            logging.info("Collecting file metrics...")
            analyzer.collect_file_metrics()

            # Select files for analysis
            logging.info("Selecting files for analysis...")
            selected_files = analyzer.select_files_for_analysis()

            if not selected_files:
                logging.error("No files selected for analysis. Try lowering the threshold.")
                sys.exit(1)

            logging.info(f"Selected {len(selected_files)} files for analysis:")
            for selection in selected_files[:10]:  # Show top 10
                logging.info(
                    f"  {selection.file_path} (score: {selection.score:.2f}, "
                    f"relevance: {selection.relevance:.2f}, "
                    f"info_value: {selection.information_value:.2f}, "
                    f"effort: {selection.effort:.2f})"
                )

            if len(selected_files) > 10:
                logging.info(f"  ... and {len(selected_files) - 10} more")

            # Update components_config with selected files
            components_config = {}
            for selection in selected_files:
                # Use the filename without extension as the component name
                component_name = selection.file_path.stem
                # Use the relative path from the project directory
                rel_path = selection.file_path.relative_to(project_dir)
                components_config[component_name] = str(rel_path)

            # Update components list
            components = list(components_config.keys())

            # Build semantic matrix with selected files
            builder = SemanticMatrixBuilder(components, intents, project_dir, components_config)
            matrix = builder.build_matrix()

            # Update analyzer with results
            for component, analysis in matrix.component_analyses.items():
                file_path = analysis.file_path
                # Consider a file useful if it has any intent alignments > 0.5
                was_useful = any(score > 0.5 for score in analysis.intent_alignments.values())
                analyzer.update_from_feedback(file_path, was_useful)

                if was_useful:
                    logging.info(f"File {file_path} was useful for analysis.")
                else:
                    logging.debug(f"File {file_path} was not useful for analysis.")
        else:
            # Build semantic matrix with all files
            builder = SemanticMatrixBuilder(components, intents, project_dir, components_config)
            matrix = builder.build_matrix()

        # Visualize matrix
        visualization_path = os.path.join(output_dir, f"{output_prefix}.{output_format}")
        visualize_matrix(matrix, visualization_path)

        # Generate report
        if args.agent_driven:
            report = generate_report(matrix, agent_driven=True, selected_files=selected_files)
        else:
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

        # Add agent-driven information if applicable
        if args.agent_driven:
            json_data["agent_driven"] = {
                "enabled": True,
                "threshold": args.threshold,
                "selected_files": [
                    {
                        "file_path": str(selection.file_path),
                        "score": selection.score,
                        "relevance": selection.relevance,
                        "information_value": selection.information_value,
                        "effort": selection.effort
                    }
                    for selection in selected_files
                ]
            }

            if args.concerns:
                json_data["agent_driven"]["concerns"] = [
                    concern.strip() for concern in args.concerns.split(",")
                ]

        json_path = os.path.join(output_dir, f"{output_prefix}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logging.info(f"Semantic matrix visualization saved to {visualization_path}")
        logging.info(f"Semantic matrix report saved to {report_path}")
        logging.info(f"Semantic matrix data saved to {json_path}")
    else:
        logging.error("No command specified. Use 'analyze', 'generate-config', 'list-intents', or 'list-plugins'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
