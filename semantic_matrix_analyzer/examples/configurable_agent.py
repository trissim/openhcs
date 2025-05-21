#!/usr/bin/env python3
"""
Example of configurable agent-driven analysis.

This script demonstrates how to use the configurable agent-driven analyzer
to intelligently select which files to analyze based on relevance, information value,
and effort required, with customizable parameters and functions.
"""

import argparse
import logging
import sys
from pathlib import Path

from semantic_matrix_analyzer.agent import AgentDrivenAnalyzer, UserIntent
from semantic_matrix_analyzer.agent.config import AgentConfig, generate_default_config
from semantic_matrix_analyzer.core import SemanticMatrixBuilder, intent_registry_global
from semantic_matrix_analyzer.plugins import plugin_manager


def setup_logging(verbose: bool = False) -> None:
    """Set up logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Configurable agent-driven analysis example")
    
    parser.add_argument("--project-dir", type=str, required=True, help="Path to project directory")
    parser.add_argument("--concerns", type=str, help="Comma-separated list of concerns to focus on")
    parser.add_argument("--components", type=str, help="Comma-separated list of components to focus on")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--generate-config", type=str, help="Generate a default configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Generate default configuration if requested
    if args.generate_config:
        generate_default_config(args.generate_config)
        logging.info(f"Generated default configuration file: {args.generate_config}")
        return
    
    # Discover and initialize plugins
    plugin_manager.discover_plugins([Path("plugins")])
    plugin_manager.initialize_plugins()
    
    # Create user intent
    concerns = []
    if args.concerns:
        concerns = [concern.strip() for concern in args.concerns.split(",")]
    
    components = []
    if args.components:
        components = [component.strip() for component in args.components.split(",")]
    
    user_intent = UserIntent(
        primary_concerns=concerns,
        component_mentions=components
    )
    
    # Create agent-driven analyzer
    if args.config:
        # Load configuration from file
        logging.info(f"Loading configuration from {args.config}")
        analyzer = AgentDrivenAnalyzer(
            codebase_path=args.project_dir,
            user_intent=user_intent,
            config_path=args.config
        )
    else:
        # Use default configuration
        logging.info("Using default configuration")
        analyzer = AgentDrivenAnalyzer(
            codebase_path=args.project_dir,
            user_intent=user_intent
        )
    
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
    for selection in selected_files:
        logging.info(
            f"  {selection.file_path} (score: {selection.score:.2f}, "
            f"relevance: {selection.relevance:.2f}, "
            f"info_value: {selection.information_value:.2f}, "
            f"effort: {selection.effort:.2f})"
        )
    
    # Create components_config for SemanticMatrixBuilder
    components_config = {}
    for selection in selected_files:
        # Use the filename without extension as the component name
        component_name = selection.file_path.stem
        # Use the relative path from the project directory
        rel_path = selection.file_path.relative_to(Path(args.project_dir))
        components_config[component_name] = str(rel_path)
    
    # Get all available intents
    intents = intent_registry_global.get_intent_names()
    
    # Build semantic matrix with selected files
    builder = SemanticMatrixBuilder(
        components=list(components_config.keys()),
        intents=intents,
        project_dir=args.project_dir,
        components_config=components_config
    )
    
    logging.info("Building semantic matrix...")
    matrix = builder.build_matrix()
    
    # Print results
    logging.info("Analysis results:")
    for component, analysis in matrix.component_analyses.items():
        logging.info(f"Component: {component}")
        for intent, score in analysis.intent_alignments.items():
            if score > 0.0:
                logging.info(f"  Intent: {intent}, Score: {score:.2f}")
    
    # Update analyzer with results
    for component, analysis in matrix.component_analyses.items():
        file_path = analysis.file_path
        # Consider a file useful if it has any intent alignments > 0.5
        was_useful = any(score > 0.5 for score in analysis.intent_alignments.values())
        analyzer.update_from_feedback(file_path, was_useful)
        
        if was_useful:
            logging.info(f"File {file_path} was useful for analysis.")
        else:
            logging.info(f"File {file_path} was not useful for analysis.")


if __name__ == "__main__":
    main()
