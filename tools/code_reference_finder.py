#!/usr/bin/env python3
"""
Code Reference Finder - A modular tool for finding references in Python code.

This tool analyzes Python code to find references to specific symbols, classes,
functions, or patterns. It can be used to identify potential issues with imports,
find usages of deprecated APIs, or locate specific code patterns.

Features:
- AST-based analysis for accurate symbol references
- String-based pattern matching for broader searches
- Import analysis to detect missing or unused imports
- Flexible output formats (text, JSON, CSV)
- Configurable search scope and filters
"""

import os
import re
import sys
import ast
import json
import csv
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"

@dataclass
class Reference:
    """A reference to a symbol or pattern in code."""
    line: int
    column: Optional[int] = None
    context: str = ""
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "line": self.line,
            "column": self.column,
            "context": self.context,
            "text": self.text
        }

@dataclass
class FileAnalysisResult:
    """Result of analyzing a file for references."""
    file_path: str
    imports: Set[str] = field(default_factory=set)
    imports_target: bool = False
    ast_references: List[Reference] = field(default_factory=list)
    string_references: List[Reference] = field(default_factory=list)
    error: Optional[str] = None

    def has_references(self) -> bool:
        """Check if the file has any references."""
        return bool(self.ast_references or self.string_references)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "imports": list(self.imports),
            "imports_target": self.imports_target,
            "ast_references": [ref.to_dict() for ref in self.ast_references],
            "string_references": [ref.to_dict() for ref in self.string_references],
            "error": self.error
        }

class ReferenceAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze imports and find references to symbols."""

    def __init__(self, target_symbols: List[str]):
        self.target_symbols = target_symbols
        self.imports = set()
        self.imports_target = False
        self.references = []
        self.current_function = None
        self.current_class = None

    def visit_Import(self, node):
        """Process import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
            # Check if any target symbol is in the import
            for symbol in self.target_symbols:
                if symbol in alias.name:
                    self.imports_target = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Process from-import statements."""
        if node.module:
            for alias in node.names:
                import_path = f"{node.module}.{alias.name}"
                self.imports.add(import_path)
                # Check if any target symbol is imported
                if alias.name in self.target_symbols:
                    self.imports_target = True
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Track current class."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Track current function."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Track current async function."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Name(self, node):
        """Find references to target symbols."""
        if node.id in self.target_symbols:
            context = f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
            self.references.append(Reference(
                line=node.lineno,
                column=node.col_offset,
                context=context or "module level",
                text=node.id
            ))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Find attribute references that might match target symbols."""
        if isinstance(node.value, ast.Name) and node.attr in self.target_symbols:
            context = f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
            self.references.append(Reference(
                line=node.lineno,
                column=node.col_offset,
                context=context or "module level",
                text=f"{node.value.id}.{node.attr}"
            ))
        self.generic_visit(node)

def find_string_references(file_path: str, patterns: List[str]) -> List[Reference]:
    """
    Find string references to patterns in a file.

    Args:
        file_path: Path to the file to analyze
        patterns: List of patterns to search for

    Returns:
        List of Reference objects
    """
    references = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for pattern in patterns:
                    if pattern in line:
                        # Find column offset
                        col_offset = line.find(pattern)
                        references.append(Reference(
                            line=i,
                            column=col_offset,
                            text=line.strip(),
                        ))
                        break  # Only add one reference per line
    except Exception as e:
        # Just return empty list if file can't be read
        pass
    return references

def analyze_file(file_path: str, target_symbols: List[str], include_strings: bool = True) -> FileAnalysisResult:
    """
    Analyze a file for references to target symbols.

    Args:
        file_path: Path to the file to analyze
        target_symbols: List of symbols to search for
        include_strings: Whether to include string references

    Returns:
        FileAnalysisResult object
    """
    result = FileAnalysisResult(file_path=file_path)

    # Find string references first (works even if there are syntax errors)
    if include_strings:
        result.string_references = find_string_references(file_path, target_symbols)

    # Try to parse and analyze the AST
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)

        analyzer = ReferenceAnalyzer(target_symbols)
        analyzer.visit(tree)

        result.imports = analyzer.imports
        result.imports_target = analyzer.imports_target
        result.ast_references = analyzer.references

    except SyntaxError as e:
        result.error = f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        result.error = f"Error: {type(e).__name__}: {e}"

    return result

def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """
    Find all Python files in a directory recursively.

    Args:
        directory: Directory to search
        exclude_patterns: List of glob patterns to exclude

    Returns:
        List of file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*"]

    python_files = []
    for root, _, files in os.walk(directory):
        # Check if this directory should be excluded
        if any(re.match(pattern, root) for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if this file should be excluded
                if not any(re.match(pattern, file_path) for pattern in exclude_patterns):
                    python_files.append(file_path)

    return python_files

def format_results_text(results: List[FileAnalysisResult], project_root: str) -> str:
    """
    Format analysis results as text.

    Args:
        results: List of FileAnalysisResult objects
        project_root: Project root directory for relative paths

    Returns:
        Formatted text
    """
    if not results:
        return "No references found."

    lines = []
    lines.append(f"Found {len(results)} files with references:")

    for result in results:
        rel_path = os.path.relpath(result.file_path, project_root)
        lines.append(f"\n{rel_path}:")

        if result.error:
            lines.append(f"  Error: {result.error}")

        lines.append(f"  Imports target: {result.imports_target}")

        if result.ast_references:
            lines.append("  AST References:")
            for ref in result.ast_references:
                lines.append(f"    Line {ref.line} in {ref.context}: {ref.text}")

        if result.string_references:
            lines.append("  String References:")
            for ref in result.string_references:
                # Truncate long lines
                text = ref.text[:80] + ('...' if len(ref.text) > 80 else '')
                lines.append(f"    Line {ref.line}: {text}")

    return "\n".join(lines)

def format_results_json(results: List[FileAnalysisResult]) -> str:
    """
    Format analysis results as JSON.

    Args:
        results: List of FileAnalysisResult objects

    Returns:
        JSON string
    """
    return json.dumps([result.to_dict() for result in results], indent=2)

def format_results_csv(results: List[FileAnalysisResult], project_root: str) -> str:
    """
    Format analysis results as CSV.

    Args:
        results: List of FileAnalysisResult objects
        project_root: Project root directory for relative paths

    Returns:
        CSV string
    """
    if not results:
        return "file_path,imports_target,ast_references,string_references,error"

    output = []
    writer = csv.StringIO()
    csv_writer = csv.writer(writer)

    # Write header
    csv_writer.writerow(["file_path", "imports_target", "ast_references", "string_references", "error"])

    # Write data
    for result in results:
        rel_path = os.path.relpath(result.file_path, project_root)
        csv_writer.writerow([
            rel_path,
            result.imports_target,
            len(result.ast_references),
            len(result.string_references),
            result.error or ""
        ])

    return writer.getvalue()

def format_results_markdown(results: List[FileAnalysisResult], project_root: str) -> str:
    """
    Format analysis results as Markdown.

    Args:
        results: List of FileAnalysisResult objects
        project_root: Project root directory for relative paths

    Returns:
        Markdown string
    """
    if not results:
        return "# Code Reference Analysis\n\nNo references found."

    lines = []
    lines.append("# Code Reference Analysis")
    lines.append(f"\nFound {len(results)} files with references:\n")

    for result in results:
        rel_path = os.path.relpath(result.file_path, project_root)
        lines.append(f"## {rel_path}")

        if result.error:
            lines.append(f"**Error:** {result.error}")

        lines.append(f"**Imports target:** {result.imports_target}")

        if result.ast_references:
            lines.append("\n### AST References")
            lines.append("| Line | Context | Reference |")
            lines.append("| ---- | ------- | --------- |")
            for ref in result.ast_references:
                lines.append(f"| {ref.line} | {ref.context} | `{ref.text}` |")

        if result.string_references:
            lines.append("\n### String References")
            lines.append("| Line | Reference |")
            lines.append("| ---- | --------- |")
            for ref in result.string_references:
                # Truncate and escape long lines
                text = ref.text[:80] + ('...' if len(ref.text) > 80 else '')
                text = text.replace("|", "\\|")
                lines.append(f"| {ref.line} | `{text}` |")

        lines.append("")  # Add blank line between files

    return "\n".join(lines)

def analyze_directory(directory: str, target_symbols: List[str],
                     exclude_patterns: List[str] = None,
                     include_strings: bool = True) -> List[FileAnalysisResult]:
    """
    Analyze all Python files in a directory for references to target symbols.

    Args:
        directory: Directory to search
        target_symbols: List of symbols to search for
        exclude_patterns: List of glob patterns to exclude
        include_strings: Whether to include string references

    Returns:
        List of FileAnalysisResult objects
    """
    python_files = find_python_files(directory, exclude_patterns)

    results = []
    for file_path in python_files:
        result = analyze_file(file_path, target_symbols, include_strings)
        if result.has_references():
            results.append(result)

    return results

def analyze_files(file_paths: List[str], target_symbols: List[str],
                 include_strings: bool = True) -> List[FileAnalysisResult]:
    """
    Analyze specific files for references to target symbols.

    Args:
        file_paths: List of file paths to analyze
        target_symbols: List of symbols to search for
        include_strings: Whether to include string references

    Returns:
        List of FileAnalysisResult objects
    """
    results = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            result = analyze_file(file_path, target_symbols, include_strings)
            if result.has_references():
                results.append(result)

    return results

def analyze_plan_file(plan_file: str, target_symbols: List[str],
                     exclude_patterns: List[str] = None,
                     include_strings: bool = True) -> List[FileAnalysisResult]:
    """
    Extract file paths from a plan file and analyze them.

    Args:
        plan_file: Path to the plan file
        target_symbols: List of symbols to search for
        exclude_patterns: List of glob patterns to exclude
        include_strings: Whether to include string references

    Returns:
        List of FileAnalysisResult objects
    """
    # Extract file paths from the plan file
    file_paths = []
    try:
        with open(plan_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for Python file paths in the content
        # This regex matches common Python file path patterns
        path_patterns = [
            r'(?:^|\s|[\'"`(])(?:\.{0,2}/|/)?(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.py(?:[\'"`)\s]|$)',
            r'(?:^|\s)(?:openhcs|semantic_matrix_analyzer)/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.py(?:\s|$)'
        ]

        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Clean up the path
                path = match.strip()
                path = re.sub(r'^[\'"`(]|[\'"`)\s]$', '', path)

                # Skip paths that are likely not actual file paths
                if 'import ' in path or 'from ' in path:
                    continue

                # Resolve relative paths
                if path.startswith('./') or path.startswith('../'):
                    base_dir = os.path.dirname(os.path.abspath(plan_file))
                    path = os.path.normpath(os.path.join(base_dir, path))
                elif not path.startswith('/'):
                    # Assume paths are relative to project root
                    path = os.path.join(PROJECT_ROOT, path)

                if os.path.exists(path):
                    file_paths.append(path)

    except Exception as e:
        print(f"Error extracting paths from plan file: {e}")
        return []

    # Analyze the extracted files
    return analyze_files(file_paths, target_symbols, include_strings)

def format_results(results: List[FileAnalysisResult], output_format: OutputFormat, project_root: str) -> str:
    """
    Format analysis results according to the specified output format.

    Args:
        results: List of FileAnalysisResult objects
        output_format: Output format
        project_root: Project root directory for relative paths

    Returns:
        Formatted results
    """
    if output_format == OutputFormat.JSON:
        return format_results_json(results)
    elif output_format == OutputFormat.CSV:
        return format_results_csv(results, project_root)
    elif output_format == OutputFormat.MARKDOWN:
        return format_results_markdown(results, project_root)
    else:  # OutputFormat.TEXT
        return format_results_text(results, project_root)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Code Reference Finder - Find references to symbols in Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find references to SpecialKey in the openhcs directory
  ./code_reference_finder.py -s SpecialKey openhcs

  # Find references to multiple symbols in specific files
  ./code_reference_finder.py -s SpecialKey Backend -f path/to/file1.py path/to/file2.py

  # Analyze files mentioned in a plan file
  ./code_reference_finder.py -s SpecialKey --plan path/to/plan.md

  # Output results in JSON format
  ./code_reference_finder.py -s SpecialKey -o json openhcs

  # Exclude test files
  ./code_reference_finder.py -s SpecialKey --exclude "*test*" openhcs

  # Find missing imports (like List, Dict from typing)
  ./code_reference_finder.py -s List Dict -o markdown openhcs --output-file reports/missing_imports.md
"""
    )

    # Target specification
    target_group = parser.add_argument_group("Target Specification")
    target_group.add_argument(
        "-s", "--symbols",
        nargs="+",
        required=True,
        help="Symbols to search for"
    )

    # Input specification
    input_group = parser.add_argument_group("Input Specification")
    input_source = input_group.add_mutually_exclusive_group(required=True)
    input_source.add_argument(
        "directories",
        nargs="*",
        default=[],
        help="Directories to search (default: current directory)"
    )
    input_source.add_argument(
        "-f", "--files",
        nargs="+",
        help="Specific files to analyze"
    )
    input_source.add_argument(
        "--plan",
        help="Extract file paths from a plan file and analyze them"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output-format",
        choices=[f.value for f in OutputFormat],
        default="text",
        help="Output format (default: text)"
    )
    output_group.add_argument(
        "--output-file",
        help="Write output to a file instead of stdout"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--no-strings",
        action="store_true",
        help="Don't include string references (only AST-based references)"
    )
    analysis_group.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching glob pattern (can be used multiple times)"
    )

    args = parser.parse_args()

    # Convert arguments
    target_symbols = args.symbols
    output_format = OutputFormat(args.output_format)
    include_strings = not args.no_strings
    exclude_patterns = args.exclude

    # Perform analysis based on input source
    if args.plan:
        results = analyze_plan_file(args.plan, target_symbols, exclude_patterns, include_strings)
    elif args.files:
        results = analyze_files(args.files, target_symbols, include_strings)
    else:
        directories = args.directories or ["."]
        all_results = []
        for directory in directories:
            dir_results = analyze_directory(directory, target_symbols, exclude_patterns, include_strings)
            all_results.extend(dir_results)
        results = all_results

    # Format results
    formatted_results = format_results(results, output_format, PROJECT_ROOT)

    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        print(f"Results written to {args.output_file}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    main()