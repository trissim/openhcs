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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import SMA components
# For development environment, use relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from semantic_matrix_analyzer.config_manager import ConfigManager
from semantic_matrix_analyzer.analyzer import SemanticAnalyzer
from semantic_matrix_analyzer.intent.extractor import IntentExtractor
from semantic_matrix_analyzer.auto_config import generate_initial_config

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

def find_python_files(project_dir: str) -> List[Path]:
    """
    Find all Python files in a project directory.

    Args:
        project_dir: Path to project directory

    Returns:
        List of paths to Python files
    """
    project_path = Path(project_dir)
    python_files = []

    for path in project_path.rglob("*.py"):
        # Skip hidden directories and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        python_files.append(path)

    return python_files

def analyze_file_structure(project_dir: str) -> Dict[str, Any]:
    """
    Analyze the file structure of a project.

    Args:
        project_dir: Path to project directory

    Returns:
        Dictionary with file structure analysis
    """
    project_path = Path(project_dir)
    python_files = find_python_files(project_dir)

    # Count files by directory
    dir_counts = {}
    for file_path in python_files:
        rel_path = file_path.relative_to(project_path)
        parent = str(rel_path.parent)
        if parent == ".":
            parent = "root"
        dir_counts[parent] = dir_counts.get(parent, 0) + 1

    # Find top-level directories
    top_dirs = set()
    for file_path in python_files:
        rel_path = file_path.relative_to(project_path)
        if len(rel_path.parts) > 1:
            top_dirs.add(rel_path.parts[0])

    # Find potential package structure
    packages = []
    for dir_name in top_dirs:
        init_path = project_path / dir_name / "__init__.py"
        if init_path.exists():
            packages.append(dir_name)

    return {
        "total_files": len(python_files),
        "directory_counts": dir_counts,
        "top_level_directories": sorted(list(top_dirs)),
        "potential_packages": packages
    }

def analyze_imports(python_files: List[Path]) -> Dict[str, Any]:
    """
    Analyze imports in Python files.

    Args:
        python_files: List of paths to Python files

    Returns:
        Dictionary with import analysis
    """
    import_counts = {}
    internal_imports = set()
    external_imports = set()

    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()

                # Simple regex-based import analysis
                import_lines = [line.strip() for line in content.split('\n')
                               if line.strip().startswith(('import ', 'from '))
                               and not line.strip().startswith('#')]

                for line in import_lines:
                    # Extract the main module being imported
                    if line.startswith('import '):
                        modules = line[7:].split(',')
                        for module in modules:
                            module = module.strip().split(' as ')[0].split('.')[0]
                            import_counts[module] = import_counts.get(module, 0) + 1

                            # Determine if internal or external
                            if any(file.stem == module for file in python_files):
                                internal_imports.add(module)
                            else:
                                external_imports.add(module)

                    elif line.startswith('from '):
                        # Handle "from x import y" syntax
                        parts = line.split(' import ')
                        if len(parts) == 2:
                            module = parts[0][5:].split('.')[0]
                            import_counts[module] = import_counts.get(module, 0) + 1

                            # Determine if internal or external
                            if any(file.stem == module for file in python_files):
                                internal_imports.add(module)
                            else:
                                external_imports.add(module)
            except Exception as e:
                # Skip files that can't be read
                continue

    # Sort imports by frequency
    top_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "top_imports": top_imports,
        "internal_modules": sorted(list(internal_imports)),
        "external_dependencies": sorted(list(external_imports))
    }

def generate_project_snapshot(
    project_dir: str,
    depth: int = 3,
    focus: str = "all",
    analyzer: Optional[Any] = None,
    output_format: str = "markdown"
) -> Union[str, Dict[str, Any]]:
    """
    Generate a snapshot of a project's architecture.

    Args:
        project_dir: Path to project directory
        depth: Depth of analysis (1-5, where 5 is most detailed)
        focus: Focus area (structure, semantics, intent, architecture, all)
        analyzer: Optional analyzer instance
        output_format: Output format (json, yaml, markdown, text)

    Returns:
        Project snapshot as string or dictionary
    """
    start_time = time.time()

    # Find Python files
    python_files = find_python_files(project_dir)

    # Initialize snapshot data
    snapshot = {
        "project": {
            "path": project_dir,
            "name": Path(project_dir).name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_depth": depth,
            "focus_areas": focus
        },
        "structure": {},
        "semantics": {},
        "intent": {},
        "architecture": {}
    }

    # Analyze file structure
    if focus in ["structure", "all"]:
        snapshot["structure"] = analyze_file_structure(project_dir)

    # Analyze imports
    if focus in ["structure", "architecture", "all"] and depth >= 2:
        snapshot["structure"]["imports"] = analyze_imports(python_files)

    # Placeholder for semantic analysis
    if focus in ["semantics", "all"] and depth >= 2:
        snapshot["semantics"] = {
            "status": "placeholder",
            "message": "Semantic analysis would analyze code patterns, naming conventions, and code quality."
        }

    # Placeholder for intent analysis
    if focus in ["intent", "all"] and depth >= 3:
        snapshot["intent"] = {
            "status": "placeholder",
            "message": "Intent analysis would extract the intended behavior from code, comments, and documentation."
        }

    # Placeholder for architectural analysis
    if focus in ["architecture", "all"] and depth >= 3:
        snapshot["architecture"] = {
            "status": "placeholder",
            "message": "Architectural analysis would identify design patterns, component relationships, and system structure."
        }

    # Add configuration information if analyzer has config
    if hasattr(analyzer, 'config') and analyzer.config:
        config = analyzer.config
        snapshot["configuration"] = {
            "auto_generated": True,
            "weights": config.get("analysis", {}).get("weights", {}),
            "patterns": config.get("analysis", {}).get("patterns", {}).get("naming_conventions", {}),
            "tokens": config.get("analysis", {}).get("tokens", {}),
            "keys": config.get("analysis", {}).get("keys", {})
        }

    # Add analysis metadata
    snapshot["meta"] = {
        "analysis_time": f"{time.time() - start_time:.2f} seconds",
        "files_analyzed": len(python_files),
        "sma_version": "0.1.0"
    }

    # Format output
    if output_format == "json":
        return snapshot
    elif output_format == "yaml":
        return snapshot
    else:
        # Generate markdown or text output
        return format_snapshot_as_markdown(snapshot)

def format_snapshot_as_markdown(snapshot: Dict[str, Any]) -> str:
    """
    Format a project snapshot as Markdown.

    Args:
        snapshot: Project snapshot dictionary

    Returns:
        Markdown-formatted snapshot
    """
    lines = []

    # Project header
    lines.append(f"# Project Snapshot: {snapshot['project']['name']}")
    lines.append(f"Generated: {snapshot['project']['timestamp']}")
    lines.append("")

    # Structure section
    if snapshot.get("structure"):
        lines.append("## Structure")

        structure = snapshot["structure"]
        lines.append(f"- **Total Python Files**: {structure.get('total_files', 'N/A')}")

        if structure.get("top_level_directories"):
            lines.append("- **Top-Level Directories**:")
            for dir_name in structure.get("top_level_directories", []):
                lines.append(f"  - {dir_name}")

        if structure.get("potential_packages"):
            lines.append("- **Python Packages**:")
            for package in structure.get("potential_packages", []):
                lines.append(f"  - {package}")

        if structure.get("imports"):
            imports = structure["imports"]

            if imports.get("top_imports"):
                lines.append("- **Top Imports**:")
                for module, count in imports.get("top_imports", [])[:10]:
                    lines.append(f"  - {module} ({count} occurrences)")

            if imports.get("external_dependencies"):
                lines.append("- **External Dependencies**:")
                for dep in imports.get("external_dependencies", [])[:10]:
                    lines.append(f"  - {dep}")

        lines.append("")

    # Semantics section
    if snapshot.get("semantics") and snapshot["semantics"].get("status") != "placeholder":
        lines.append("## Semantics")
        # Add semantics content here
        lines.append("")

    # Intent section
    if snapshot.get("intent") and snapshot["intent"].get("status") != "placeholder":
        lines.append("## Intent")
        # Add intent content here
        lines.append("")

    # Architecture section
    if snapshot.get("architecture") and snapshot["architecture"].get("status") != "placeholder":
        lines.append("## Architecture")
        # Add architecture content here
        lines.append("")

    # Metadata
    lines.append("## Analysis Metadata")
    lines.append(f"- **Analysis Time**: {snapshot['meta']['analysis_time']}")
    lines.append(f"- **Files Analyzed**: {snapshot['meta']['files_analyzed']}")
    lines.append(f"- **SMA Version**: {snapshot['meta']['sma_version']}")
    lines.append("")

    # Configuration information
    lines.append("## Configuration")
    lines.append("An initial configuration has been auto-generated based on codebase analysis:")
    lines.append("- Naming patterns extracted from code")
    lines.append("- Common tokens identified in docstrings and comments")
    lines.append("- Intent indicators derived from function and class names")
    lines.append("- Error patterns detected in exception handling")
    lines.append("")

    # Display configuration details if available
    if snapshot.get("configuration"):
        config = snapshot["configuration"]

        if config.get("weights"):
            lines.append("### Weights")
            for name, value in config["weights"].items():
                lines.append(f"- **{name}**: {value}")
            lines.append("")

        if config.get("patterns"):
            lines.append("### Naming Patterns")
            for name, pattern in config["patterns"].items():
                lines.append(f"- **{name}**: `{pattern}`")
            lines.append("")

        if config.get("tokens"):
            lines.append("### Tokens")
            for category, tokens in config["tokens"].items():
                if tokens:
                    lines.append(f"- **{category}**: {', '.join(tokens)}")
            lines.append("")

        if config.get("keys"):
            lines.append("### Keys")
            for category, keys in config["keys"].items():
                if keys:
                    lines.append(f"- **{category}**: {', '.join(keys)}")
            lines.append("")

    lines.append("View the full configuration with:")
    lines.append("```")
    lines.append("sma-cli config view")
    lines.append("```")
    lines.append("")

    # Configuration guidance
    lines.append("## Next Steps for AI Agents")
    lines.append("1. Share this snapshot with the human to confirm your understanding")
    lines.append("2. Update configuration based on human feedback:")
    lines.append("   ```")
    lines.append("   sma-cli config update --weight code_structure=0.8")
    lines.append("   sma-cli config update --weight naming_conventions=0.7")
    lines.append("   ```")
    lines.append("3. Continue analysis with the improved configuration")
    lines.append("4. Remember: This is a human-in-the-loop learning process where you tune")
    lines.append("   configuration parameters based on feedback to improve your understanding")
    lines.append("5. Each time you receive confirmation on ambiguities, update the configuration")
    lines.append("   to reflect this new knowledge")

    return "\n".join(lines)

def print_usage_examples() -> None:
    """Print usage examples for the SMA CLI."""
    print_section("USAGE EXAMPLES")

    examples = [
        ("Get an immediate project overview:",
         "sma-cli snapshot --project-dir ."),

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

    QUICK START FOR AI AGENTS:
    1. Run 'sma-cli snapshot --project-dir .' to get an immediate architectural overview
    2. Share the snapshot with the human to confirm your understanding
    3. Update configuration based on human feedback: 'sma-cli config update --weight <param>=<value>'
    4. Continue analysis with improved configuration as a human-in-the-loop learning process
    5. Use other SMA commands to analyze specific aspects of the codebase
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

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Get an immediate architectural overview of a project")
    snapshot_parser.add_argument(
        "--project-dir",
        help="Path to project directory",
        type=str,
        required=True
    )
    snapshot_parser.add_argument(
        "--depth",
        help="Depth of analysis (1-5, where 5 is most detailed)",
        type=int,
        default=3,
        choices=range(1, 6)
    )
    snapshot_parser.add_argument(
        "--focus",
        help="Focus area (structure, semantics, intent, architecture, all)",
        choices=["structure", "semantics", "intent", "architecture", "all"],
        default="all"
    )

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

def handle_snapshot_command(args: argparse.Namespace) -> None:
    """
    Handle the snapshot command.

    This command provides an immediate architectural overview of a project,
    giving AI agents a quick understanding of the codebase's structure,
    semantics, intent, and architecture.
    """
    print_header("PROJECT SNAPSHOT")
    print(f"Generating architectural overview of project at: {args.project_dir}")
    print(f"Analysis depth: {args.depth}")
    print(f"Focus areas: {args.focus}")

    # Generate initial configuration if none exists
    print("Checking for configuration...")
    config = generate_initial_config(args.project_dir)
    if config:
        print(color_text("Using auto-generated configuration based on codebase analysis", "GREEN"))

    # Create analyzer with the configuration
    analyzer = SemanticAnalyzer(config)

    # Generate snapshot
    snapshot = generate_project_snapshot(
        project_dir=args.project_dir,
        depth=args.depth,
        focus=args.focus,
        analyzer=analyzer,
        output_format=args.format
    )

    # Print snapshot
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == "json":
                json.dump(snapshot, f, indent=2)
            elif args.format == "yaml":
                import yaml
                yaml.dump(snapshot, f, default_flow_style=False)
            else:
                f.write(snapshot)
        print(f"Snapshot saved to {args.output}")
    else:
        if args.format == "json":
            print(json.dumps(snapshot, indent=2))
        elif args.format == "yaml":
            import yaml
            print(yaml.dump(snapshot, default_flow_style=False))
        else:
            print(snapshot)

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
    if args.command == "snapshot":
        handle_snapshot_command(args)
    elif args.command == "analyze":
        handle_analyze_command(args)
    elif args.command == "error-trace":
        handle_error_trace_command(args)
    elif args.command == "extract-intent":
        handle_extract_intent_command(args)
    elif args.command == "config":
        handle_config_command(args)
    else:
        print(color_text("Error: No command specified", "RED"))
        print("Use one of: snapshot, analyze, error-trace, extract-intent, config")
        sys.exit(1)

if __name__ == "__main__":
    main()
