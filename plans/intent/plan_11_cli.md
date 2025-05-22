# plan_11_cli.md
## Component: Command-Line Interface

### Objective
Create a command-line interface (CLI) for the Structural Intent Analysis system that allows users to analyze codebases, customize the analysis with configuration files, and generate reports in different formats.

### Plan
1. Create a CLI using argparse or click
2. Implement commands for analyzing codebases
3. Add options for specifying configuration files
4. Add options for controlling output format and destination
5. Add options for filtering and limiting results
6. Implement progress reporting and error handling
7. Create documentation and examples

### Findings
A command-line interface makes the Structural Intent Analysis system more accessible and easier to use, especially for integration into existing workflows and CI/CD pipelines. It also provides a convenient way to customize the analysis with configuration files.

### Implementation Draft

```python
#!/usr/bin/env python3
"""
Command-line interface for the Structural Intent Analysis system.
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.config.configuration import Configuration
from semantic_matrix_analyzer.intent.analyzers.configurable_intent_analyzer import ConfigurableIntentAnalyzer
from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, DependencyExtractor


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
    parser = argparse.ArgumentParser(
        description="Analyze a codebase to extract intent from code structure."
    )
    
    # Required arguments
    parser.add_argument(
        "path",
        help="Path to the codebase to analyze."
    )
    
    # Configuration options
    parser.add_argument(
        "-c", "--config",
        help="Path to a configuration file."
    )
    parser.add_argument(
        "--python",
        action="store_true",
        help="Use Python-specific configuration."
    )
    parser.add_argument(
        "--java",
        action="store_true",
        help="Use Java-specific configuration."
    )
    parser.add_argument(
        "--microservices",
        action="store_true",
        help="Use microservices-specific configuration."
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal configuration."
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use comprehensive configuration."
    )
    
    # Analysis options
    parser.add_argument(
        "--no-names",
        action="store_true",
        help="Disable name analysis."
    )
    parser.add_argument(
        "--no-types",
        action="store_true",
        help="Disable type hint analysis."
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Disable structural analysis."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold (0.0 to 1.0)."
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results."
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Path to the output file."
    )
    parser.add_argument(
        "-f", "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format."
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output."
    )
    
    # Miscellaneous options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information."
    )
    
    return parser.parse_args()


def get_configuration(args: argparse.Namespace) -> Configuration:
    """Get the configuration based on command-line arguments.
    
    Args:
        args: The parsed command-line arguments.
        
    Returns:
        A Configuration object.
    """
    # Start with default configuration
    config = Configuration()
    
    # Load configuration from file if specified
    if args.config:
        config = Configuration.from_file(args.config)
    
    # Apply language-specific configuration
    if args.python:
        from semantic_matrix_analyzer.intent.config.example_configs import PYTHON_CONFIG
        config = Configuration(PYTHON_CONFIG)
    elif args.java:
        from semantic_matrix_analyzer.intent.config.example_configs import JAVA_CONFIG
        config = Configuration(JAVA_CONFIG)
    
    # Apply architecture-specific configuration
    if args.microservices:
        from semantic_matrix_analyzer.intent.config.example_configs import MICROSERVICES_CONFIG
        config = Configuration(MICROSERVICES_CONFIG)
    
    # Apply analysis level configuration
    if args.minimal:
        from semantic_matrix_analyzer.intent.config.example_configs import MINIMAL_CONFIG
        config = Configuration(MINIMAL_CONFIG)
    elif args.comprehensive:
        from semantic_matrix_analyzer.intent.config.example_configs import COMPREHENSIVE_CONFIG
        config = Configuration(COMPREHENSIVE_CONFIG)
    
    # Apply command-line overrides
    if args.no_names:
        config.set("name_analysis.enabled", False)
    
    if args.no_types:
        config.set("type_analysis.enabled", False)
    
    if args.no_structure:
        config.set("structural_analysis.enabled", False)
    
    if args.min_confidence is not None:
        config.set("integration.min_confidence", args.min_confidence)
    
    if args.max_results is not None:
        config.set("integration.max_results", args.max_results)
    
    if args.format:
        config.set("integration.report_format", args.format)
    
    return config


def find_python_files(path: Path) -> List[Path]:
    """Find Python files in a directory.
    
    Args:
        path: The directory to search.
        
    Returns:
        A list of Python file paths.
    """
    if path.is_file():
        if path.suffix.lower() == ".py":
            return [path]
        return []
    
    python_files = []
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(os.path.join(root, file)))
    
    return python_files


def find_java_files(path: Path) -> List[Path]:
    """Find Java files in a directory.
    
    Args:
        path: The directory to search.
        
    Returns:
        A list of Java file paths.
    """
    if path.is_file():
        if path.suffix.lower() == ".java":
            return [path]
        return []
    
    java_files = []
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and target
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "target"]
        
        for file in files:
            if file.endswith(".java"):
                java_files.append(Path(os.path.join(root, file)))
    
    return java_files


def build_dependency_graph(file_paths: List[Path]) -> DependencyGraph:
    """Build a dependency graph for a list of files.
    
    Args:
        file_paths: A list of file paths.
        
    Returns:
        A DependencyGraph object.
    """
    dependency_graph = DependencyGraph()
    dependency_extractor = DependencyExtractor()
    
    for file_path in file_paths:
        try:
            nodes, edges = dependency_extractor.extract_dependencies(file_path)
            
            for node in nodes:
                dependency_graph.add_node(node)
            
            for edge in edges:
                dependency_graph.add_edge(edge)
        except Exception as e:
            logging.warning(f"Error extracting dependencies from {file_path}: {e}")
    
    return dependency_graph


def analyze_codebase(path: Path, config: Configuration) -> Dict[str, Any]:
    """Analyze a codebase to extract intent.
    
    Args:
        path: The path to the codebase.
        config: The configuration to use.
        
    Returns:
        A report of the analysis.
    """
    # Find files to analyze
    if config.get("name_analysis.enabled", True) or config.get("type_analysis.enabled", True):
        if config.get("java", False):
            file_paths = find_java_files(path)
        else:
            file_paths = find_python_files(path)
        
        logging.info(f"Found {len(file_paths)} files to analyze")
    else:
        file_paths = []
    
    # Build dependency graph if structural analysis is enabled
    dependency_graph = None
    if config.get("structural_analysis.enabled", True) and file_paths:
        logging.info("Building dependency graph...")
        dependency_graph = build_dependency_graph(file_paths)
        logging.info(f"Built dependency graph with {len(dependency_graph.nodes)} nodes and {len(dependency_graph.edges)} edges")
    
    # Create analyzer
    analyzer = ConfigurableIntentAnalyzer(config)
    
    # Analyze codebase
    logging.info("Analyzing codebase...")
    start_time = time.time()
    report = analyzer.analyze_codebase(file_paths, dependency_graph)
    end_time = time.time()
    
    # Add analysis metadata
    report["metadata"] = {
        "path": str(path),
        "files_analyzed": len(file_paths),
        "analysis_time": end_time - start_time
    }
    
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    return report


def save_report(report: Dict[str, Any], analyzer: ConfigurableIntentAnalyzer, args: argparse.Namespace) -> None:
    """Save the analysis report.
    
    Args:
        report: The analysis report.
        analyzer: The analyzer used to generate the report.
        args: The parsed command-line arguments.
    """
    # Format the report
    formatted_report = analyzer.format_report(report, args.format)
    
    # Save to file if specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_report)
        logging.info(f"Report saved to {args.output}")
    else:
        # Print to console
        print(formatted_report)


def main() -> None:
    """Main entry point for the CLI."""
    # Parse command-line arguments
    args = parse_args()
    
    # Show version information if requested
    if args.version:
        from semantic_matrix_analyzer import __version__
        print(f"Semantic Matrix Analyzer v{__version__}")
        return
    
    # Set up logging
    setup_logging(args.verbose)
    
    try:
        # Get configuration
        config = get_configuration(args)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return
        
        # Analyze codebase
        path = Path(args.path)
        report = analyze_codebase(path, config)
        
        # Create analyzer for formatting
        analyzer = ConfigurableIntentAnalyzer(config)
        
        # Save report
        save_report(report, analyzer, args)
    
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
```
