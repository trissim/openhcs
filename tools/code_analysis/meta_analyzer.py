#!/usr/bin/env python3
"""
Meta Analyzer - A tool that combines multiple code analysis tools.

This tool provides meta-commands that run multiple analysis tools in sequence
and generate comprehensive reports for specific use cases.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum, auto

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Path to the tools directory
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

class MetaCommand(Enum):
    """Meta-commands that combine multiple analysis tools."""
    COMPREHENSIVE = "comprehensive"  # Run all analysis tools
    ARCHITECTURE = "architecture"    # Analyze architecture (interfaces, call graph)
    SEMANTICS = "semantics"          # Analyze semantics (roles, state mutations)
    IMPORTS = "imports"              # Analyze imports and fix issues
    QUALITY = "quality"              # Analyze code quality (complexity, maintainability)
    ASYNC = "async"                  # Analyze async/await patterns

def run_tool(tool_path: str, args: List[str]) -> int:
    """
    Run a tool with the given arguments.

    Args:
        tool_path: Path to the tool
        args: Arguments to pass to the tool

    Returns:
        Return code from the tool
    """
    cmd = [sys.executable, tool_path] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def ensure_directory(directory: str):
    """
    Ensure a directory exists.

    Args:
        directory: Directory to ensure exists
    """
    os.makedirs(directory, exist_ok=True)

def run_comprehensive_analysis(path: str, output_dir: str, exclude: List[str] = None) -> int:
    """
    Run comprehensive analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run call graph analyzer
    call_graph_tool = os.path.join(TOOLS_DIR, "call_graph_analyzer.py")
    call_graph_output = os.path.join(output_dir, "call_graph_analysis.md")
    if run_tool(call_graph_tool, [path, "-o", call_graph_output] + exclude_args) != 0:
        print("Warning: Call graph analysis failed")

    # Run semantic role analyzer
    semantic_tool = os.path.join(TOOLS_DIR, "semantic_role_analyzer.py")
    semantic_output = os.path.join(output_dir, "semantic_role_analysis.md")
    if run_tool(semantic_tool, [path, "-o", semantic_output] + exclude_args) != 0:
        print("Warning: Semantic role analysis failed")

    # Run interface classifier
    interface_tool = os.path.join(TOOLS_DIR, "interface_classifier.py")
    interface_output = os.path.join(output_dir, "interface_analysis.md")
    if run_tool(interface_tool, [path, "-o", interface_output] + exclude_args) != 0:
        print("Warning: Interface analysis failed")

    # Run import validator
    import_tool = os.path.join(TOOLS_DIR, "../import_analysis/import_validator.py")
    import_output = os.path.join(output_dir, "import_analysis.md")
    if run_tool(import_tool, [path, "-o", import_output] + exclude_args) != 0:
        print("Warning: Import analysis failed")

    # Run code analyzer CLI for dependencies
    code_analyzer_tool = os.path.join(TOOLS_DIR, "code_analyzer_cli.py")
    # Check if path is a file or directory and use appropriate command
    if os.path.isfile(path):
        dependency_output = os.path.join(output_dir, f"file_dependencies_{os.path.basename(path)}.md")
        if run_tool(code_analyzer_tool, ["file-dependencies", path, "-o", dependency_output]) != 0:
            print("Warning: File dependency analysis failed")
    else:
        dependency_output = os.path.join(output_dir, f"module_dependency_graph_{os.path.basename(path)}.md")
        if run_tool(code_analyzer_tool, ["dependencies", path, "-o", dependency_output]) != 0:
            print("Warning: Directory dependency analysis failed")

    # Run code analyzer CLI for snapshot
    snapshot_output = os.path.join(output_dir, f"{os.path.basename(path)}_codebase_snapshot.csv")
    if run_tool(code_analyzer_tool, ["snapshot", "-t", path, "-o", snapshot_output]) != 0:
        print("Warning: Snapshot analysis failed")

    # Run code analyzer CLI for async patterns
    if run_tool(code_analyzer_tool, ["async-patterns", path, "--output-dir", output_dir]) != 0:
        print("Warning: Async pattern analysis failed")

    print(f"\nComprehensive analysis complete. Reports written to {output_dir}")
    return 0

def run_architecture_analysis(path: str, output_dir: str, exclude: List[str] = None) -> int:
    """
    Run architecture analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run call graph analyzer
    call_graph_tool = os.path.join(TOOLS_DIR, "call_graph_analyzer.py")
    call_graph_output = os.path.join(output_dir, "call_graph_analysis.md")
    if run_tool(call_graph_tool, [path, "-o", call_graph_output] + exclude_args) != 0:
        print("Warning: Call graph analysis failed")

    # Run interface classifier
    interface_tool = os.path.join(TOOLS_DIR, "interface_classifier.py")
    interface_output = os.path.join(output_dir, "interface_analysis.md")
    if run_tool(interface_tool, [path, "-o", interface_output] + exclude_args) != 0:
        print("Warning: Interface analysis failed")

    # Run code analyzer CLI for dependencies
    code_analyzer_tool = os.path.join(TOOLS_DIR, "code_analyzer_cli.py")
    # Check if path is a file or directory and use appropriate command
    if os.path.isfile(path):
        if run_tool(code_analyzer_tool, ["file-dependencies", path, "--output-dir", output_dir]) != 0:
            print("Warning: File dependency analysis failed")
    else:
        if run_tool(code_analyzer_tool, ["dependencies", path, "--output-dir", output_dir]) != 0:
            print("Warning: Directory dependency analysis failed")

    # Run code analyzer CLI for async patterns
    if run_tool(code_analyzer_tool, ["async-patterns", path, "--output-dir", output_dir]) != 0:
        print("Warning: Async pattern analysis failed")

    print(f"\nArchitecture analysis complete. Reports written to {output_dir}")
    return 0

def run_semantics_analysis(path: str, output_dir: str, exclude: List[str] = None) -> int:
    """
    Run semantics analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run semantic role analyzer
    semantic_tool = os.path.join(TOOLS_DIR, "semantic_role_analyzer.py")
    semantic_output = os.path.join(output_dir, "semantic_role_analysis.md")
    if run_tool(semantic_tool, [path, "-o", semantic_output] + exclude_args) != 0:
        print("Warning: Semantic role analysis failed")

    # Run code analyzer CLI for matrix
    code_analyzer_tool = os.path.join(TOOLS_DIR, "code_analyzer_cli.py")
    if run_tool(code_analyzer_tool, ["matrix", path, "--output-dir", output_dir]) != 0:
        print("Warning: Matrix analysis failed")

    print(f"\nSemantics analysis complete. Reports written to {output_dir}")
    return 0

def run_imports_analysis(path: str, output_dir: str, exclude: List[str] = None, fix: bool = False) -> int:
    """
    Run imports analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude
        fix: Whether to fix import issues

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run import validator
    import_tool = os.path.join(TOOLS_DIR, "../import_analysis/import_validator.py")
    import_output = os.path.join(output_dir, "import_analysis.md")
    if run_tool(import_tool, [path, "-o", import_output] + exclude_args) != 0:
        print("Warning: Import analysis failed")

    # Fix import issues if requested
    if fix:
        fix_tool = os.path.join(TOOLS_DIR, "../import_analysis/fix_imports.py")
        if run_tool(fix_tool, [path] + exclude_args) != 0:
            print("Warning: Import fixing failed")

    print(f"\nImports analysis complete. Reports written to {output_dir}")
    return 0

def run_quality_analysis(path: str, output_dir: str, exclude: List[str] = None) -> int:
    """
    Run code quality analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run code analyzer CLI for snapshot
    code_analyzer_tool = os.path.join(TOOLS_DIR, "code_analyzer_cli.py")
    if run_tool(code_analyzer_tool, ["snapshot", path, "--output-dir", output_dir]) != 0:
        print("Warning: Snapshot analysis failed")

    print(f"\nCode quality analysis complete. Reports written to {output_dir}")
    return 0

def run_async_analysis(path: str, output_dir: str, exclude: List[str] = None) -> int:
    """
    Run async/await pattern analysis on a path.

    Args:
        path: Path to analyze
        output_dir: Directory to write reports to
        exclude: List of patterns to exclude

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    ensure_directory(output_dir)

    # Build exclude args
    exclude_args = []
    if exclude:
        for pattern in exclude:
            exclude_args.extend(["--exclude", pattern])

    # Run code analyzer CLI for async patterns
    code_analyzer_tool = os.path.join(TOOLS_DIR, "code_analyzer_cli.py")
    if run_tool(code_analyzer_tool, ["async-patterns", path, "--output-dir", output_dir]) != 0:
        print("Warning: Async pattern analysis failed")

    print(f"\nAsync pattern analysis complete. Reports written to {output_dir}")
    return 0

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Meta Analyzer - Run multiple code analysis tools in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis on a directory
  ./meta_analyzer.py comprehensive openhcs/tui

  # Run architecture analysis on a directory
  ./meta_analyzer.py architecture openhcs/core

  # Run semantics analysis on a directory
  ./meta_analyzer.py semantics openhcs/processing

  # Run imports analysis and fix issues
  ./meta_analyzer.py imports openhcs/tui --fix

  # Run code quality analysis on a directory
  ./meta_analyzer.py quality openhcs

  # Run async/await pattern analysis on a directory
  ./meta_analyzer.py async openhcs/tui
"""
    )

    parser.add_argument(
        "command",
        choices=[cmd.value for cmd in MetaCommand],
        help="Meta-command to run"
    )

    parser.add_argument(
        "path",
        help="File or directory to analyze"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="reports/code_analysis",
        help="Directory to write reports to (default: reports/code_analysis)"
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching pattern (can be used multiple times)"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix issues (only applicable to some commands)"
    )

    args = parser.parse_args()

    # Run the appropriate meta-command
    if args.command == MetaCommand.COMPREHENSIVE.value:
        return run_comprehensive_analysis(args.path, args.output_dir, args.exclude)
    elif args.command == MetaCommand.ARCHITECTURE.value:
        return run_architecture_analysis(args.path, args.output_dir, args.exclude)
    elif args.command == MetaCommand.SEMANTICS.value:
        return run_semantics_analysis(args.path, args.output_dir, args.exclude)
    elif args.command == MetaCommand.IMPORTS.value:
        return run_imports_analysis(args.path, args.output_dir, args.exclude, args.fix)
    elif args.command == MetaCommand.QUALITY.value:
        return run_quality_analysis(args.path, args.output_dir, args.exclude)
    elif args.command == MetaCommand.ASYNC.value:
        return run_async_analysis(args.path, args.output_dir, args.exclude)
    else:
        print(f"Error: Unknown command {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
