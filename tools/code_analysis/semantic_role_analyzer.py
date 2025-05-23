#!/usr/bin/env python3
"""
Semantic Role Analyzer - A tool to analyze semantic roles of functions and methods.

This tool:
1. Identifies semantic roles of functions (getter, setter, factory, etc.)
2. Detects state mutations
3. Classifies functions by their behavior patterns
4. Identifies related function groups (e.g., get/set pairs)
"""

import os
import sys
import ast
import re
import json
import argparse
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

class SemanticRole(Enum):
    """Semantic roles for functions and methods."""
    GETTER = auto()          # Returns a value without side effects
    SETTER = auto()          # Sets a value
    FACTORY = auto()         # Creates and returns a new object
    MUTATOR = auto()         # Modifies state
    VALIDATOR = auto()       # Validates input
    CONVERTER = auto()       # Converts between types
    PROCESSOR = auto()       # Processes data
    INITIALIZER = auto()     # Initializes state
    FINALIZER = auto()       # Cleans up resources
    EVENT_HANDLER = auto()   # Handles events
    UTILITY = auto()         # Utility function
    UNKNOWN = auto()         # Unknown role

class StateAccess(Enum):
    """Types of state access."""
    READ_ONLY = auto()       # Only reads state
    WRITE_ONLY = auto()      # Only writes state
    READ_WRITE = auto()      # Both reads and writes state
    NO_STATE_ACCESS = auto() # Doesn't access state

class FunctionSemantics:
    """Semantic information about a function or method."""
    def __init__(self, name: str, module: str, class_name: Optional[str] = None):
        self.name = name
        self.module = module
        self.class_name = class_name
        self.role = SemanticRole.UNKNOWN
        self.state_access = StateAccess.NO_STATE_ACCESS
        self.reads_attributes = set()
        self.writes_attributes = set()
        self.creates_objects = set()
        self.calls = set()
        self.parameters = []
        self.return_annotation = None
        self.docstring = None
        self.lineno = 0
        self.related_functions = []  # Related functions (e.g., getter/setter pairs)

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the function."""
        if self.class_name:
            return f"{self.module}.{self.class_name}.{self.name}"
        return f"{self.module}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.full_name} ({self.role.name})"

class SemanticAnalysisVisitor(ast.NodeVisitor):
    """AST visitor to analyze semantic roles of functions."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.functions = {}  # Maps full_name to FunctionSemantics
        self.current_function = None
        self.current_class = None

    def visit_ClassDef(self, node):
        """Process class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        """Process async function definitions."""
        self._process_function(node)

    def _process_function(self, node):
        """Common processing for function and async function definitions."""
        old_function = self.current_function

        # Create function semantics
        function_name = node.name
        function_semantics = FunctionSemantics(
            name=function_name,
            module=self.module_name,
            class_name=self.current_class
        )
        function_semantics.lineno = node.lineno

        # Extract parameters
        function_semantics.parameters = [arg.arg for arg in node.args.args]

        # Extract return annotation if present
        if node.returns:
            if isinstance(node.returns, ast.Name):
                function_semantics.return_annotation = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                function_semantics.return_annotation = f"{node.returns.value.id}.{node.returns.attr}"
            elif isinstance(node.returns, ast.Subscript):
                # This is a simplification; handling complex type annotations would require more work
                function_semantics.return_annotation = "complex_type"

        # Extract docstring if present
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            function_semantics.docstring = node.body[0].value.value

        # Store function semantics
        self.functions[function_semantics.full_name] = function_semantics
        self.current_function = function_semantics

        # Visit function body to analyze semantics
        self.generic_visit(node)

        # Determine semantic role based on collected information
        self._determine_semantic_role(function_semantics)

        # Restore previous function context
        self.current_function = old_function

    def visit_Attribute(self, node):
        """Process attribute access."""
        if self.current_function is None:
            self.generic_visit(node)
            return

        # Check for self attribute access in methods
        if isinstance(node.value, ast.Name) and node.value.id == 'self' and self.current_class:
            attr_name = node.attr

            # Determine if this is a read or write
            if isinstance(node.ctx, ast.Store):
                self.current_function.writes_attributes.add(attr_name)
            elif isinstance(node.ctx, ast.Load):
                self.current_function.reads_attributes.add(attr_name)

        self.generic_visit(node)

    def visit_Call(self, node):
        """Process function calls."""
        if self.current_function is None:
            self.generic_visit(node)
            return

        # Check for object creation
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check if this looks like a class name (capitalized)
            if func_name and func_name[0].isupper():
                self.current_function.creates_objects.add(func_name)

            # Record function call
            self.current_function.calls.add(func_name)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr

                # Record method call
                self.current_function.calls.add(f"{obj_name}.{method_name}")

        self.generic_visit(node)

    def _determine_semantic_role(self, func: FunctionSemantics):
        """Determine the semantic role of a function based on its characteristics."""
        name = func.name.lower()

        # Determine state access
        if func.writes_attributes and func.reads_attributes:
            func.state_access = StateAccess.READ_WRITE
        elif func.writes_attributes:
            func.state_access = StateAccess.WRITE_ONLY
        elif func.reads_attributes:
            func.state_access = StateAccess.READ_ONLY
        else:
            func.state_access = StateAccess.NO_STATE_ACCESS

        # Check for common naming patterns
        if name.startswith('get_') or name.startswith('fetch_') or name.startswith('retrieve_'):
            func.role = SemanticRole.GETTER
        elif name.startswith('set_') or name.startswith('update_'):
            func.role = SemanticRole.SETTER
        elif name.startswith('create_') or name.startswith('make_') or name.startswith('build_'):
            func.role = SemanticRole.FACTORY
        elif name.startswith('validate_') or name.startswith('check_') or name.startswith('is_'):
            func.role = SemanticRole.VALIDATOR
        elif name.startswith('convert_') or name.startswith('to_'):
            func.role = SemanticRole.CONVERTER
        elif name.startswith('process_') or name.startswith('handle_'):
            func.role = SemanticRole.PROCESSOR
        elif name == '__init__' or name.startswith('initialize_') or name.startswith('init_'):
            func.role = SemanticRole.INITIALIZER
        elif name == '__del__' or name.startswith('finalize_') or name.startswith('cleanup_'):
            func.role = SemanticRole.FINALIZER
        elif name.startswith('on_') or name.endswith('_handler'):
            func.role = SemanticRole.EVENT_HANDLER
        elif func.creates_objects:
            func.role = SemanticRole.FACTORY
        elif func.state_access == StateAccess.WRITE_ONLY or func.state_access == StateAccess.READ_WRITE:
            func.role = SemanticRole.MUTATOR
        elif func.state_access == StateAccess.READ_ONLY:
            func.role = SemanticRole.GETTER
        else:
            func.role = SemanticRole.UTILITY

def analyze_file(file_path: str) -> Dict[str, FunctionSemantics]:
    """
    Analyze a file to determine semantic roles of functions.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dictionary mapping function names to FunctionSemantics objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine module name from file path
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        module_name = os.path.splitext(rel_path.replace(os.path.sep, '.'))[0]

        # Parse the file
        tree = ast.parse(content, filename=file_path)

        # Analyze semantics
        visitor = SemanticAnalysisVisitor(module_name)
        visitor.visit(tree)

        return visitor.functions

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {}

def analyze_directory(directory: str, exclude_patterns: List[str] = None) -> Dict[str, FunctionSemantics]:
    """
    Analyze all Python files in a directory to determine semantic roles.

    Args:
        directory: Directory to analyze
        exclude_patterns: List of patterns to exclude

    Returns:
        Dictionary mapping function names to FunctionSemantics objects
    """
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*"]

    all_functions = {}

    for root, _, files in os.walk(directory):
        # Check if this directory should be excluded
        if any(pattern in root for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if this file should be excluded
                if not any(pattern in file_path for pattern in exclude_patterns):
                    functions = analyze_file(file_path)
                    all_functions.update(functions)

    # Find related functions (e.g., getter/setter pairs)
    _find_related_functions(all_functions)

    return all_functions

def _find_related_functions(functions: Dict[str, FunctionSemantics]):
    """
    Find related functions (e.g., getter/setter pairs).

    Args:
        functions: Dictionary mapping function names to FunctionSemantics objects
    """
    # Group functions by module and class
    grouped_functions = defaultdict(list)
    for func_name, func in functions.items():
        key = (func.module, func.class_name)
        grouped_functions[key].append(func)

    # Find related functions within each group
    for group in grouped_functions.values():
        for func in group:
            # Skip if not a getter or setter
            if func.role not in [SemanticRole.GETTER, SemanticRole.SETTER]:
                continue

            # Find related functions
            name = func.name
            if func.role == SemanticRole.GETTER and name.startswith('get_'):
                # Look for corresponding setter
                setter_name = 'set_' + name[4:]
                for other_func in group:
                    if other_func.name == setter_name:
                        func.related_functions.append(other_func.full_name)
                        other_func.related_functions.append(func.full_name)
            elif func.role == SemanticRole.SETTER and name.startswith('set_'):
                # Look for corresponding getter
                getter_name = 'get_' + name[4:]
                for other_func in group:
                    if other_func.name == getter_name:
                        func.related_functions.append(other_func.full_name)
                        other_func.related_functions.append(func.full_name)

def format_results_markdown(functions: Dict[str, FunctionSemantics]) -> str:
    """
    Format semantic analysis results as Markdown.

    Args:
        functions: Dictionary mapping function names to FunctionSemantics objects

    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Semantic Role Analysis")
    lines.append(f"\nAnalyzed {len(functions)} functions.\n")

    # Group functions by role
    functions_by_role = defaultdict(list)
    for func in functions.values():
        functions_by_role[func.role].append(func)

    # Report functions by role
    for role in SemanticRole:
        funcs = functions_by_role[role]
        if not funcs:
            continue

        lines.append(f"## {role.name} Functions")
        lines.append(f"Found {len(funcs)} functions with role {role.name}:")
        lines.append("```")
        for func in sorted(funcs, key=lambda f: f.full_name):
            lines.append(f"- {func.full_name}")
        lines.append("```")

    # Report state mutators
    mutators = [func for func in functions.values()
                if func.state_access in [StateAccess.WRITE_ONLY, StateAccess.READ_WRITE]]
    if mutators:
        lines.append("\n## State Mutators")
        lines.append("Functions that modify state:")
        lines.append("```")
        for func in sorted(mutators, key=lambda f: f.full_name):
            attrs = ", ".join(func.writes_attributes) if func.writes_attributes else "indirect mutation"
            lines.append(f"- {func.full_name}: {attrs}")
        lines.append("```")

    # Report getter/setter pairs
    related_funcs = [func for func in functions.values() if func.related_functions]
    if related_funcs:
        lines.append("\n## Related Function Pairs")
        lines.append("Functions with related counterparts (e.g., getter/setter pairs):")
        lines.append("```")
        reported_pairs = set()
        for func in sorted(related_funcs, key=lambda f: f.full_name):
            for related_func_name in func.related_functions:
                pair = tuple(sorted([func.full_name, related_func_name]))
                if pair not in reported_pairs:
                    lines.append(f"- {pair[0]} <-> {pair[1]}")
                    reported_pairs.add(pair)
        lines.append("```")

    return "\n".join(lines)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Semantic Role Analyzer - Analyze semantic roles of functions and methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        help="File or directory to analyze"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for the report (default: stdout)"
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching pattern (can be used multiple times)"
    )

    args = parser.parse_args()

    # Analyze the code
    if os.path.isfile(args.path):
        functions = analyze_file(args.path)
    elif os.path.isdir(args.path):
        functions = analyze_directory(args.path, args.exclude)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1

    # Generate the report
    report = format_results_markdown(functions)

    # Output the report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0

if __name__ == "__main__":
    sys.exit(main())
