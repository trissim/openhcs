#!/usr/bin/env python3
"""
Semantic Matrix Analyzer CLI

This is the main entry point for the Semantic Matrix Analyzer (SMA) tool.
It provides a unified interface for all SMA functionality, including:
- Code analysis
- Error trace processing
- Intent extraction
- Configuration management

The SMA is designed for AI agents to assist humans in architecting, implementing,
and debugging code with maximum accuracy and minimum cognitive load through
the principle of separating judgment from execution via intent extraction.
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import SMA components
try:
    from semantic_matrix_analyzer.config_manager import ConfigManager
    from semantic_matrix_analyzer.analyzer import SemanticAnalyzer
    from semantic_matrix_analyzer.intent.extractor import IntentExtractor
except ImportError:
    # Handle case where package is not installed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from semantic_matrix_analyzer.config_manager import ConfigManager
    from semantic_matrix_analyzer.analyzer import SemanticAnalyzer
    from semantic_matrix_analyzer.intent.extractor import IntentExtractor

# ANSI color codes for terminal output
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "END": "\033[0m"
}

def color_text(text: str, color: str) -> str:
    """Add color to text for terminal output."""
    if color not in COLORS:
        return text
    return f"{COLORS[color]}{text}{COLORS['END']}"

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + color_text("=" * 80, "BOLD"))
    print(color_text(text.center(80), "BOLD"))
    print(color_text("=" * 80, "BOLD") + "\n")

def print_section(title: str) -> None:
    """Print a formatted section title."""
    print("\n" + color_text(f"--- {title} ", "BLUE") + color_text("-" * (76 - len(title)), "BLUE"))

def wrap_text(text: str, indent: int = 0) -> str:
    """Wrap text to fit terminal width with indentation."""
    width = 80 - indent
    wrapped = textwrap.fill(text, width=width)
    if indent > 0:
        indent_str = " " * indent
        wrapped = "\n".join(f"{indent_str}{line}" for line in wrapped.split("\n"))
    return wrapped

def print_agent_role() -> None:
    """Print information about the AI agent's role in using SMA."""
    print_section("AI AGENT ROLE")
    
    role_text = """
    As an AI agent, your role is to use the Semantic Matrix Analyzer to assist humans with coding tasks while minimizing their cognitive load. You should:
    
    1. Build comprehensive mental models of codebases
    2. Extract intent from code, error traces, and conversations
    3. Identify potentially fatal assumptions in your reasoning
    4. Report ambiguities to humans when clarification is needed
    5. Make precise, targeted fixes based on thorough analysis
    
    The SMA tool separates judgment (understanding what should be done) from execution (implementing the solution), allowing humans to focus on creativity while you handle semantic analysis.
    """
    
    print(wrap_text(role_text.strip(), indent=4))

def print_tool_intent() -> None:
    """Print information about the intent of the SMA tool."""
    print_section("TOOL INTENT")
    
    intent_text = """
    The Semantic Matrix Analyzer is designed to:
    
    1. Reduce cognitive load on human developers
    2. Improve code quality through intent-driven development
    3. Accelerate development by focusing human attention on creativity
    4. Provide deeper understanding of codebases and error traces
    5. Enable more accurate and targeted fixes
    
    This tool is now established as the principal methodology for AI-assisted development, based on the core principle of separating judgment from execution through intent extraction.
    """
    
    print(wrap_text(intent_text.strip(), indent=4))

def print_usage_examples() -> None:
    """Print usage examples for the SMA CLI."""
    print_section("USAGE EXAMPLES")
    
    examples = [
        ("Analyze code for intent extraction:", 
         "sma-cli analyze --code-file path/to/file.py"),
        
        ("Process an error trace:", 
         "sma-cli error-trace --input-file error.txt"),
        
        ("Extract intent from conversation:", 
         "sma-cli extract-intent --text \"User's concerns about error handling\""),
        
        ("Update configuration weights:", 
         "sma-cli config update --weight error_detection=0.9"),
        
        ("Analyze a specific component for an intent:", 
         "sma-cli analyze --component \"Authentication\" --intent \"Error Handling\""),
        
        ("Generate a comprehensive report:", 
         "sma-cli analyze --project-dir /path/to/project --report"),
    ]
    
    for title, command in examples:
        print(f"  {color_text(title, 'BOLD')}")
        print(f"    {color_text('$', 'GREEN')} {command}\n")

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the SMA CLI."""
    # Create main parser with description
    description = """
    Semantic Matrix Analyzer (SMA) - AI-driven code analysis tool
    
    The SMA is designed for AI agents to assist humans in architecting, implementing,
    and debugging code with maximum accuracy and minimum cognitive load through
    the principle of separating judgment from execution via intent extraction.
    """
    
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add global options
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    
    parser.add_argument(
        "--verbose", "-v",
        help="Increase output verbosity",
        action="count",
        default=0
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (default: stdout)",
        type=str
    )
    
    parser.add_argument(
        "--format", "-f",
        help="Output format (json, yaml, markdown, text)",
        choices=["json", "yaml", "markdown", "text"],
        default="text"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code for intent extraction")
    analyze_parser.add_argument(
        "--code-file",
        help="Path to code file to analyze",
        type=str
    )
    analyze_parser.add_argument(
        "--project-dir",
        help="Path to project directory",
        type=str
    )
    analyze_parser.add_argument(
        "--component",
        help="Component to analyze",
        type=str
    )
    analyze_parser.add_argument(
        "--intent",
        help="Intent to analyze for",
        type=str
    )
    analyze_parser.add_argument(
        "--report",
        help="Generate a comprehensive report",
        action="store_true"
    )
    
    # Error trace command
    error_parser = subparsers.add_parser("error-trace", help="Process an error trace")
    error_parser.add_argument(
        "--input-file",
        help="Path to error trace file",
        type=str
    )
    error_parser.add_argument(
        "--text",
        help="Error trace text",
        type=str
    )
    error_parser.add_argument(
        "--build-model",
        help="Build a comprehensive mental model",
        action="store_true"
    )
    
    # Extract intent command
    intent_parser = subparsers.add_parser("extract-intent", help="Extract intent from conversation")
    intent_parser.add_argument(
        "--input-file",
        help="Path to conversation file",
        type=str
    )
    intent_parser.add_argument(
        "--text",
        help="Conversation text",
        type=str
    )
    intent_parser.add_argument(
        "--output-file",
        help="Path to output file for extracted intents",
        type=str
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Configuration command")
    
    # Config view command
    view_parser = config_subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument(
        "--section",
        help="Section to view (e.g., 'analysis.weights')",
        type=str
    )
    
    # Config update command
    update_parser = config_subparsers.add_parser("update", help="Update configuration")
    update_parser.add_argument(
        "--weight",
        help="Update weight (format: name=value)",
        action="append"
    )
    update_parser.add_argument(
        "--pattern",
        help="Update pattern (format: category.name=value)",
        action="append"
    )
    update_parser.add_argument(
        "--add-token",
        help="Add token (format: category=token)",
        action="append"
    )
    update_parser.add_argument(
        "--remove-token",
        help="Remove token (format: category=token)",
        action="append"
    )
    update_parser.add_argument(
        "--add-key",
        help="Add key (format: category=key)",
        action="append"
    )
    update_parser.add_argument(
        "--remove-key",
        help="Remove key (format: category=key)",
        action="append"
    )
    
    # Config reset command
    reset_parser = config_subparsers.add_parser("reset", help="Reset configuration to defaults")
    reset_parser.add_argument(
        "--confirm",
        help="Confirm reset",
        action="store_true"
    )
    
    return parser

def handle_analyze_command(args: argparse.Namespace) -> None:
    """Handle the analyze command."""
    print_header("CODE ANALYSIS")
    print("Analyzing code for intent extraction...")
    
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))

def handle_error_trace_command(args: argparse.Namespace) -> None:
    """Handle the error-trace command."""
    print_header("ERROR TRACE PROCESSING")
    print("Processing error trace...")
    
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))

def handle_extract_intent_command(args: argparse.Namespace) -> None:
    """Handle the extract-intent command."""
    print_header("INTENT EXTRACTION")
    print("Extracting intent from conversation...")
    
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))

def handle_config_command(args: argparse.Namespace) -> None:
    """Handle the config command."""
    print_header("CONFIGURATION MANAGEMENT")
    
    # Create config manager
    config_manager = ConfigManager(args.config)
    
    if args.config_command == "view":
        # View configuration
        config = config_manager.get_config()
        section = config
        
        if args.section:
            # Get nested section
            parts = args.section.split(".")
            current = config
            
            for part in parts:
                if part not in current:
                    print(color_text(f"Error: Section '{part}' not found in configuration", "RED"))
                    return
                current = current[part]
                
            section = current
        
        # Print configuration
        if args.format == "json":
            print(json.dumps(section, indent=2))
        elif args.format == "yaml":
            import yaml
            print(yaml.dump(section, default_flow_style=False))
        else:
            print_section("CONFIGURATION")
            print(json.dumps(section, indent=2))
            
    elif args.config_command == "update":
        # Update weights
        if args.weight:
            for weight_arg in args.weight:
                try:
                    name, value = weight_arg.split("=")
                    config_manager.set_weight(name, float(value))
                    print(f"Updated weight '{name}' to {value}")
                except ValueError:
                    print(color_text(f"Error: Invalid weight format '{weight_arg}'. Use name=value", "RED"))
        
        # Update patterns
        if args.pattern:
            for pattern_arg in args.pattern:
                try:
                    path, value = pattern_arg.split("=")
                    category, name = path.split(".")
                    
                    # Try to convert value to appropriate type
                    try:
                        # Try as number
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # Keep as string
                        pass
                        
                    config_manager.set_pattern(category, name, value)
                    print(f"Updated pattern '{category}.{name}' to {value}")
                except ValueError:
                    print(color_text(f"Error: Invalid pattern format '{pattern_arg}'. Use category.name=value", "RED"))
        
        # Add tokens
        if args.add_token:
            for token_arg in args.add_token:
                try:
                    category, token = token_arg.split("=")
                    config_manager.add_token(category, token)
                    print(f"Added token '{token}' to category '{category}'")
                except ValueError:
                    print(color_text(f"Error: Invalid token format '{token_arg}'. Use category=token", "RED"))
        
        # Remove tokens
        if args.remove_token:
            for token_arg in args.remove_token:
                try:
                    category, token = token_arg.split("=")
                    config_manager.remove_token(category, token)
                    print(f"Removed token '{token}' from category '{category}'")
                except ValueError:
                    print(color_text(f"Error: Invalid token format '{token_arg}'. Use category=token", "RED"))
        
        # Add keys
        if args.add_key:
            for key_arg in args.add_key:
                try:
                    category, key = key_arg.split("=")
                    config_manager.add_key(category, key)
                    print(f"Added key '{key}' to category '{category}'")
                except ValueError:
                    print(color_text(f"Error: Invalid key format '{key_arg}'. Use category=key", "RED"))
        
        # Remove keys
        if args.remove_key:
            for key_arg in args.remove_key:
                try:
                    category, key = key_arg.split("=")
                    config_manager.remove_key(category, key)
                    print(f"Removed key '{key}' from category '{category}'")
                except ValueError:
                    print(color_text(f"Error: Invalid key format '{key_arg}'. Use category=key", "RED"))
                    
    elif args.config_command == "reset":
        # Reset configuration
        if not args.confirm:
            print(color_text("Warning: This will reset all configuration to defaults.", "YELLOW"))
            print(color_text("Use --confirm to proceed.", "YELLOW"))
            return
            
        config_manager.reset_to_defaults()
        print(color_text("Configuration reset to defaults.", "GREEN"))
        
    else:
        print(color_text("Error: No configuration command specified", "RED"))
        print("Use one of: view, update, reset")

def main() -> None:
    """Main entry point for the SMA CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no arguments or just --help, print extended help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]):
        parser.print_help()
        print("\n")
        print_tool_intent()
        print_agent_role()
        print_usage_examples()
        return
    
    # Execute command
    if args.command == "analyze":
        handle_analyze_command(args)
    elif args.command == "error-trace":
        handle_error_trace_command(args)
    elif args.command == "extract-intent":
        handle_extract_intent_command(args)
    elif args.command == "config":
        handle_config_command(args)
    else:
        print(color_text("Error: No command specified", "RED"))
        print("Use one of: analyze, error-trace, extract-intent, config")
        sys.exit(1)

if __name__ == "__main__":
    main()
