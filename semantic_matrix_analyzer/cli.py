#!/usr/bin/env python3
"""
Main command-line interface for the Semantic Matrix Analyzer.
"""

import argparse
import logging
import sys
from pathlib import Path

from semantic_matrix_analyzer import __version__


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
        description="Semantic Matrix Analyzer - A tool for analyzing code structure and intent."
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Intent analysis command
    intent_parser = subparsers.add_parser(
        "intent",
        help="Analyze a codebase to extract intent from code structure."
    )
    intent_parser.add_argument(
        "path",
        help="Path to the codebase to analyze."
    )
    intent_parser.add_argument(
        "-c", "--config",
        help="Path to a configuration file."
    )
    intent_parser.add_argument(
        "--python",
        action="store_true",
        help="Use Python-specific configuration."
    )
    intent_parser.add_argument(
        "--java",
        action="store_true",
        help="Use Java-specific configuration."
    )
    intent_parser.add_argument(
        "--microservices",
        action="store_true",
        help="Use microservices-specific configuration."
    )
    intent_parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal configuration."
    )
    intent_parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use comprehensive configuration."
    )
    intent_parser.add_argument(
        "--no-names",
        action="store_true",
        help="Disable name analysis."
    )
    intent_parser.add_argument(
        "--no-types",
        action="store_true",
        help="Disable type hint analysis."
    )
    intent_parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Disable structural analysis."
    )
    intent_parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold (0.0 to 1.0)."
    )
    intent_parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results."
    )
    intent_parser.add_argument(
        "-o", "--output",
        help="Path to the output file."
    )
    intent_parser.add_argument(
        "-f", "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format."
    )
    intent_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output."
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information."
    )
    
    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Handle commands
    if args.command == "intent":
        # Import the intent CLI module
        from semantic_matrix_analyzer.intent.cli import (
            get_configuration, analyze_codebase, save_report
        )
        from semantic_matrix_analyzer.intent.analyzers.intent_analyzer import ConfigurableIntentAnalyzer
        
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
    
    elif args.command == "version" or args.command is None:
        # Show version information
        print(f"Semantic Matrix Analyzer v{__version__}")
    
    else:
        logging.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
